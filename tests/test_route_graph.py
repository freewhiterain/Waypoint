"""交通图谱缓存与高德路线 provider（HTTP 全部 mock，不访问真实高德）。"""

from datetime import timedelta

import httpx
import pytest

from app.agents.route_graph import AMAP_DISTANCE_URL, AMAP_PLACE_URL, AmapRouteProvider, RouteGraphProvider
from app.agents.travel_time import DistrictEstimateProvider, Place, TravelTimeService
from app.mcp_core.reliability import ResilientExecutor


POIS = {"熊猫基地": "104.146,30.733", "宽窄巷子": "104.055,30.669", "武侯祠": "104.048,30.646"}


class FakeStore:
    def __init__(self, coordinates=None, fail=False):
        self.saved = []
        self._coordinates = coordinates or {}
        self.fail = fail

    async def find_legs(self, city, pairs, *, max_age):
        if self.fail:
            raise ConnectionError("database down")
        return []

    async def coordinates(self, city, places):
        return {place.id: self._coordinates[place.id] for place in places if place.id in self._coordinates}

    async def save(self, city, places, coordinates, legs):
        self.saved.append((city, [place.name for place in places], dict(coordinates), list(legs)))


def amap_client(requests: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        url = str(request.url).split("?")[0]
        if url == AMAP_PLACE_URL:
            location = POIS.get(request.url.params["keywords"])
            return httpx.Response(200, json={"status": "1", "pois": [{"location": location}] if location else []})
        if url == AMAP_DISTANCE_URL:
            origins = request.url.params["origins"].split("|")
            return httpx.Response(200, json={"status": "1", "results": [
                {"origin_id": str(index + 1), "distance": str(5000 * (index + 1)), "duration": str(900 * (index + 1))}
                for index in range(len(origins))
            ]})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def places(*names, category="attractions"):
    return [Place(id=name, name=name, category=category, district="某区") for name in names]


@pytest.mark.asyncio
async def test_amap_geocodes_measures_driving_time_and_writes_back_to_graph():
    requests: list[httpx.Request] = []
    store = FakeStore()
    provider = AmapRouteProvider(
        store=store, api_key="test-key", client=amap_client(requests), executor=ResilientExecutor(max_retries=0)
    )
    panda, kuanzhai, wuhou = places("熊猫基地", "宽窄巷子", "武侯祠")

    legs = await provider.legs("成都", [(panda, wuhou), (kuanzhai, wuhou)])

    assert {(leg.from_name, leg.minutes, leg.distance_km) for leg in legs} == {
        ("熊猫基地", 15, 5.0),
        ("宽窄巷子", 30, 10.0),
    }
    assert all(leg.mode == "驾车" and leg.source == "amap" and leg.source_url for leg in legs)
    # 两段终点相同，只发一次距离测量请求。
    assert sum(1 for request in requests if str(request.url).startswith(AMAP_DISTANCE_URL)) == 1
    city, saved_places, coordinates, saved_legs = store.saved[0]
    assert city == "成都" and len(saved_legs) == 2
    assert coordinates["熊猫基地"] == pytest.approx((30.733, 104.146))


@pytest.mark.asyncio
async def test_amap_reuses_cached_coordinates_and_skips_unlocatable_places():
    requests: list[httpx.Request] = []
    store = FakeStore(coordinates={"熊猫基地": (30.733, 104.146)})
    provider = AmapRouteProvider(
        store=store, api_key="test-key", client=amap_client(requests), executor=ResilientExecutor(max_retries=0)
    )
    panda, unknown, wuhou = places("熊猫基地", "查无此地", "武侯祠")

    legs = await provider.legs("成都", [(panda, wuhou), (unknown, wuhou)])

    assert [leg.from_name for leg in legs] == ["熊猫基地"]
    place_lookups = [request.url.params["keywords"] for request in requests if str(request.url).startswith(AMAP_PLACE_URL)]
    assert "熊猫基地" not in place_lookups  # 坐标已在图谱里


@pytest.mark.asyncio
async def test_without_key_amap_answers_nothing():
    provider = AmapRouteProvider(api_key="")

    assert await provider.legs("成都", [tuple(places("a", "b"))]) == []


@pytest.mark.asyncio
async def test_graph_failure_falls_back_to_district_estimate():
    service = TravelTimeService([RouteGraphProvider(FakeStore(fail=True)), DistrictEstimateProvider()])

    legs = await service.compute("成都", places("a", "b"))

    assert [leg.source for leg in legs] == ["district_estimate"]


@pytest.mark.asyncio
async def test_route_graph_provider_passes_freshness_window():
    seen = {}

    class RecordingStore(FakeStore):
        async def find_legs(self, city, pairs, *, max_age):
            seen["max_age"] = max_age
            return []

    await RouteGraphProvider(RecordingStore(), max_age=timedelta(days=7)).legs("成都", [tuple(places("a", "b"))])

    assert seen["max_age"] == timedelta(days=7)
