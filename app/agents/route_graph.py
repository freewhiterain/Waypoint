"""交通图谱：地点实体之间以"交通耗时"为边的图，以及高德路线 provider。

- 节点复用 knowledgeentity（city + category + name 唯一），坐标存在实体 attributes 里。
- 边是 travelroute：两地之间某种交通方式的分钟数和公里数，带来源和查询时间。

RouteGraphProvider 只读图；AmapRouteProvider 在图里没有时向高德查询，查完写回图，
同一城市查过的地点对以后直接命中缓存。两者都实现 TravelTimeProvider，按顺序挂在
TravelTimeService 上，答不上的地点对自动交给后面的 provider。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert

from app.agents.travel_time import Place, TravelLeg
from app.config import settings
from app.mcp_core.reliability import ExternalServiceError, ResilientExecutor
from app.models.base import async_session_maker
from app.models.knowledge_graph import KnowledgeEntity, TravelRoute
from app.utils.logger import app_logger

DRIVING_MODE = "驾车"
AMAP_DISTANCE_URL = "https://restapi.amap.com/v3/distance"
AMAP_PLACE_URL = "https://restapi.amap.com/v3/place/text"
AMAP_DOC_URL = "https://lbs.amap.com/api/webservice/guide/api/direction"
# 图谱查询走本地数据库，数据库没起时不能拖住整个规划，超时就当作没命中。
_DB_TIMEOUT_SECONDS = 3


def _ordered(first: UUID, second: UUID) -> tuple[UUID, UUID]:
    return (first, second) if str(first) <= str(second) else (second, first)


class RouteGraphStore:
    def __init__(self, session_factory=async_session_maker):
        self.session_factory = session_factory

    async def _entity_ids(self, session, city: str, places: Sequence[Place]) -> dict[str, UUID]:
        if not places:
            return {}
        names = {place.name for place in places}
        rows = (
            await session.execute(
                select(KnowledgeEntity).where(KnowledgeEntity.city == city, KnowledgeEntity.name.in_(names))
            )
        ).scalars().all()
        by_identity = {(row.category, row.name): row.id for row in rows}
        return {
            place.id: by_identity[(place.category, place.name)]
            for place in places
            if (place.category, place.name) in by_identity
        }

    async def find_legs(
        self,
        city: str,
        pairs: Sequence[tuple[Place, Place]],
        *,
        max_age: timedelta,
    ) -> list[TravelLeg]:
        places = list({place.id: place for pair in pairs for place in pair}.values())
        async with self.session_factory() as session:
            entity_ids = await self._entity_ids(session, city, places)
            wanted = {
                _ordered(entity_ids[a.id], entity_ids[b.id]): (a, b)
                for a, b in pairs
                if a.id in entity_ids and b.id in entity_ids
            }
            if not wanted:
                return []
            cutoff = datetime.now(timezone.utc) - max_age
            rows = (
                await session.execute(
                    select(TravelRoute).where(
                        TravelRoute.retrieved_at >= cutoff,
                        or_(
                            *(
                                and_(TravelRoute.from_entity_id == low, TravelRoute.to_entity_id == high)
                                for low, high in wanted
                            )
                        ),
                    )
                )
            ).scalars().all()
        legs: dict[tuple[UUID, UUID], TravelLeg] = {}
        for row in rows:
            key = (row.from_entity_id, row.to_entity_id)
            origin, target = wanted[key]
            # 同一对地点有多种方式时取最快的。
            if key in legs and legs[key].minutes <= row.minutes:
                continue
            legs[key] = TravelLeg(
                from_id=origin.id,
                to_id=target.id,
                from_name=origin.name,
                to_name=target.name,
                minutes=row.minutes,
                distance_km=row.distance_km,
                mode=row.mode,
                source="route_graph",
                source_url=row.source_url,
                retrieved_at=row.retrieved_at,
            )
        return list(legs.values())

    async def coordinates(self, city: str, places: Sequence[Place]) -> dict[str, tuple[float, float]]:
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(KnowledgeEntity).where(
                        KnowledgeEntity.city == city,
                        KnowledgeEntity.name.in_({place.name for place in places}),
                    )
                )
            ).scalars().all()
        by_identity = {(row.category, row.name): row.attributes or {} for row in rows}
        found: dict[str, tuple[float, float]] = {}
        for place in places:
            attributes = by_identity.get((place.category, place.name), {})
            if "latitude" in attributes and "longitude" in attributes:
                found[place.id] = (float(attributes["latitude"]), float(attributes["longitude"]))
        return found

    async def _upsert_entity(self, session, city: str, place: Place, attributes: dict | None) -> UUID:
        values = {
            "city": city,
            "category": place.category,
            "name": place.name,
            "source_document": "travel_route",
            "attributes": attributes or {},
        }
        update = {"attributes": attributes} if attributes else {"name": place.name}
        statement = (
            insert(KnowledgeEntity)
            .values(**values)
            .on_conflict_do_update(constraint="uq_knowledge_entity_identity", set_=update)
            .returning(KnowledgeEntity.id)
        )
        return (await session.execute(statement)).scalar_one()

    async def save(
        self,
        city: str,
        places: Sequence[Place],
        coordinates: dict[str, tuple[float, float]],
        legs: Sequence[TravelLeg],
    ) -> None:
        async with self.session_factory() as session, session.begin():
            entity_ids: dict[str, UUID] = {}
            for place in places:
                coords = coordinates.get(place.id)
                attributes = {"latitude": coords[0], "longitude": coords[1]} if coords else None
                entity_ids[place.id] = await self._upsert_entity(session, city, place, attributes)
            for leg in legs:
                if leg.minutes is None or leg.mode is None:
                    continue
                low, high = _ordered(entity_ids[leg.from_id], entity_ids[leg.to_id])
                values = {
                    "from_entity_id": low,
                    "to_entity_id": high,
                    "mode": leg.mode,
                    "minutes": leg.minutes,
                    "distance_km": leg.distance_km,
                    "source": leg.source,
                    "source_url": leg.source_url,
                    "retrieved_at": leg.retrieved_at,
                }
                statement = (
                    insert(TravelRoute)
                    .values(**values)
                    .on_conflict_do_update(
                        constraint="uq_travel_route_identity",
                        set_={key: values[key] for key in ("minutes", "distance_km", "source", "source_url", "retrieved_at")},
                    )
                )
                await session.execute(statement)


class RouteGraphProvider:
    name = "route_graph"

    def __init__(self, store: RouteGraphStore, *, max_age: timedelta = timedelta(days=30)):
        self.store = store
        self.max_age = max_age

    async def legs(self, city: str, pairs: Sequence[tuple[Place, Place]]) -> list[TravelLeg]:
        try:
            return await asyncio.wait_for(
                self.store.find_legs(city, pairs, max_age=self.max_age), timeout=_DB_TIMEOUT_SECONDS
            )
        except Exception as exc:
            app_logger.warning(f"交通图谱查询失败，交给下一个数据源: {type(exc).__name__}: {exc}")
            return []


class AmapRouteProvider:
    """用高德"距离测量"接口算驾车耗时：一次请求可算多个起点到同一终点。"""

    name = "amap"

    def __init__(
        self,
        *,
        store: RouteGraphStore | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        executor: ResilientExecutor | None = None,
    ):
        self.store = store
        self.api_key = api_key if api_key is not None else settings.amap_api_key
        self.client = client
        self.executor = executor or ResilientExecutor(
            timeout_seconds=settings.external_timeout_seconds,
            max_retries=settings.external_max_retries,
        )

    async def _request(self, url: str, params: dict) -> dict:
        params = {"key": self.api_key, "output": "JSON", **params}
        if self.client is not None:
            response = await self.client.get(url, params=params)
        else:
            async with httpx.AsyncClient(timeout=settings.external_timeout_seconds, trust_env=False) as client:
                response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "1":
            raise ExternalServiceError(f"高德接口失败：{data.get('info', '未知错误')}")
        return data

    async def _geocode(self, city: str, place: Place) -> tuple[float, float] | None:
        async def operation():
            return await self._request(
                AMAP_PLACE_URL, {"keywords": place.name, "city": city, "citylimit": "true", "offset": 1}
            )

        data = await self.executor.execute(f"amap:poi:{city}:{place.name}", operation, ttl_seconds=86400)
        pois = data.get("pois") or []
        location = pois[0].get("location") if pois else None
        if not location or "," not in str(location):
            return None
        longitude, latitude = (float(value) for value in str(location).split(",", 1))
        return latitude, longitude

    async def _resolve_coordinates(self, city: str, places: Sequence[Place]) -> dict[str, tuple[float, float]]:
        coordinates = {
            place.id: (place.latitude, place.longitude)
            for place in places
            if place.latitude is not None and place.longitude is not None
        }
        missing = [place for place in places if place.id not in coordinates]
        if missing and self.store is not None:
            try:
                coordinates.update(
                    await asyncio.wait_for(self.store.coordinates(city, missing), timeout=_DB_TIMEOUT_SECONDS)
                )
            except Exception as exc:
                app_logger.warning(f"读取地点坐标缓存失败: {type(exc).__name__}: {exc}")
        for place in places:
            if place.id in coordinates:
                continue
            try:
                found = await self._geocode(city, place)
            except Exception as exc:
                app_logger.warning(f"高德地点定位失败 {place.name}: {type(exc).__name__}: {exc}")
                continue
            if found is not None:
                coordinates[place.id] = found
        return coordinates

    async def legs(self, city: str, pairs: Sequence[tuple[Place, Place]]) -> list[TravelLeg]:
        if not self.api_key:
            return []
        places = list({place.id: place for pair in pairs for place in pair}.values())
        coordinates = await self._resolve_coordinates(city, places)

        # 按终点分组：一次请求带上所有起点。
        by_target: dict[str, list[Place]] = {}
        targets: dict[str, Place] = {}
        for origin, target in pairs:
            if origin.id in coordinates and target.id in coordinates:
                by_target.setdefault(target.id, []).append(origin)
                targets[target.id] = target

        legs: list[TravelLeg] = []
        now = datetime.now(timezone.utc)
        for target_id, origins in by_target.items():
            target = targets[target_id]
            try:
                data = await self._request(
                    AMAP_DISTANCE_URL,
                    {
                        "origins": "|".join(_lnglat(coordinates[origin.id]) for origin in origins),
                        "destination": _lnglat(coordinates[target_id]),
                        "type": 1,
                    },
                )
            except Exception as exc:
                app_logger.warning(f"高德距离测量失败 → {target.name}: {type(exc).__name__}: {exc}")
                continue
            for row in data.get("results") or []:
                try:
                    origin = origins[int(row["origin_id"]) - 1]
                    seconds, meters = float(row["duration"]), float(row["distance"])
                except (KeyError, ValueError, IndexError):
                    continue
                legs.append(
                    TravelLeg(
                        from_id=origin.id,
                        to_id=target.id,
                        from_name=origin.name,
                        to_name=target.name,
                        minutes=max(math.ceil(seconds / 60), 1),
                        distance_km=round(meters / 1000, 1),
                        mode=DRIVING_MODE,
                        source=self.name,
                        source_url=AMAP_DOC_URL,
                        retrieved_at=now,
                    )
                )

        if legs and self.store is not None:
            try:
                await self.store.save(city, places, coordinates, legs)
            except Exception as exc:
                app_logger.warning(f"交通图谱写回失败，本次结果照常使用: {type(exc).__name__}: {exc}")
        return legs


def _lnglat(coordinates: tuple[float, float]) -> str:
    latitude, longitude = coordinates
    return f"{longitude:.6f},{latitude:.6f}"


__all__ = ["AmapRouteProvider", "RouteGraphProvider", "RouteGraphStore"]
