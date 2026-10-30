"""联网搜索适配器：Tavily、自建 SearXNG，以及按顺序降级的组合。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Protocol

import httpx

from app.config import settings
from app.mcp_core.reliability import ExternalServiceError, ResilientExecutor
from app.schemas.planning import Evidence
from app.utils.logger import app_logger


class TavilySearchAdapter:
    def __init__(self, api_key: str | None = None, executor: ResilientExecutor | None = None):
        self.api_key = api_key if api_key is not None else settings.tavily_api_key
        self.executor = executor or ResilientExecutor(
            timeout_seconds=settings.external_timeout_seconds,
            max_retries=settings.external_max_retries,
        )

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        if not self.api_key:
            raise ExternalServiceError("未配置 TAVILY_API_KEY")

        async def operation():
            async with httpx.AsyncClient(timeout=settings.external_timeout_seconds, trust_env=False) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max(1, min(max_results, 10)),
                        "search_depth": "advanced",
                    },
                )
                response.raise_for_status()
                return response.json()

        data = await self.executor.execute(
            f"tavily:{query}:{max_results}",
            operation,
            ttl_seconds=1800,
        )
        now = datetime.now(timezone.utc)
        return [
            Evidence(
                content=str(item.get("content") or item.get("title") or ""),
                source=str(item.get("title") or "Tavily 搜索结果"),
                source_url=item.get("url"),
                retrieved_at=now,
                valid_from=now,
                valid_until=now + timedelta(hours=6),
                confidence=float(item.get("score") or 0.6),
                metadata={"provider": "tavily", "query": query},
            )
            for item in data.get("results", [])
            if item.get("content") or item.get("title")
        ]


class _Throttle:
    """进程内串行、两次请求至少间隔 interval 秒。

    一次规划里景点/美食/住宿子 Agent 并发搜索，十来个请求同时打到 SearXNG，
    上游的搜狗、夸克几乎立刻弹验证码并被 SearXNG 暂停很久。排队慢一点，换来能用。
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._last = 0.0

    async def wait(self, interval: float) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            delay = self._last + interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


_searxng_throttle = _Throttle()


class SearxngSearchAdapter:
    """自建 SearXNG 的 JSON 接口（需要在 SearXNG 设置里开启 json 格式）。免费、无额度限制。"""

    def __init__(
        self, base_url: str | None = None, executor: ResilientExecutor | None = None, engines: str | None = None
    ):
        self.base_url = (base_url if base_url is not None else settings.searxng_url).rstrip("/")
        self.engines = (engines if engines is not None else settings.searxng_engines).strip()
        self.executor = executor or ResilientExecutor(
            timeout_seconds=settings.external_timeout_seconds,
            max_retries=settings.external_max_retries,
        )

    async def _fetch(self, query: str, engines: str) -> list[dict]:
        params = {"q": query, "format": "json", "language": "zh-CN"}
        if engines:
            params["engines"] = engines

        async def operation():
            await _searxng_throttle.wait(settings.searxng_min_interval_seconds)
            async with httpx.AsyncClient(timeout=settings.external_timeout_seconds, trust_env=False) as client:
                response = await client.get(f"{self.base_url}/search", params=params)
                response.raise_for_status()
                return response.json()

        data = await self.executor.execute(f"searxng:{engines}:{query}", operation, ttl_seconds=1800)
        return [item for item in data.get("results", []) if item.get("content")]

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        if not self.base_url:
            raise ExternalServiceError("未配置 SEARXNG_URL")

        results = await self._fetch(query, self.engines)
        if not results and self.engines:
            # 指定的引擎被验证码暂停时退回 SearXNG 默认引擎；无关结果由上层按目的地过滤。
            results = await self._fetch(query, "")
        now = datetime.now(timezone.utc)
        return [
            Evidence(
                content=str(item["content"]),
                source=str(item.get("title") or "SearXNG 搜索结果"),
                source_url=item.get("url"),
                retrieved_at=now,
                valid_from=now,
                valid_until=now + timedelta(hours=6),
                # 元搜索没有相关度分数，按排名给一个递减的置信度。
                confidence=round(max(0.4, 0.7 - index * 0.05), 2),
                metadata={"provider": "web", "engine": item.get("engine"), "query": query},
            )
            for index, item in enumerate(results[: max(1, min(max_results, 10))])
        ]


class SearchAdapter(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[Evidence]: ...


class RedisSearchCache:
    """把搜索结果存进 Redis，重启后、多次规划同一城市时不再重复打搜索引擎。

    地点库外的城市全靠联网搜索，SearXNG 上游一被并发请求就弹验证码；同一个查询
    几小时内结果不会变。缓存时长不超过证据自己的有效期（valid_until），过期证据
    下游本来就会丢掉，存得再久也没用。Redis 不可用时静默跳过，一分钟后再试。
    """

    PREFIX = "waypoint:search:"
    MAX_TTL_SECONDS = 6 * 3600
    RETRY_AFTER_SECONDS = 60

    def __init__(self, url: str | None = None):
        self.url = url or settings.redis_url
        self._client = None
        self._down_until = 0.0

    def _key(self, query: str, max_results: int) -> str:
        return self.PREFIX + hashlib.sha1(f"{max_results}:{query}".encode()).hexdigest()

    async def _redis(self):
        if time.monotonic() < self._down_until:
            return None
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.from_url(self.url, socket_timeout=1, socket_connect_timeout=1)
        return self._client

    def _mark_down(self, exc: Exception) -> None:
        app_logger.warning(f"搜索缓存不可用，暂时直接搜索: {type(exc).__name__}: {exc}")
        self._down_until = time.monotonic() + self.RETRY_AFTER_SECONDS

    async def get(self, query: str, max_results: int) -> list[Evidence] | None:
        try:
            client = await self._redis()
            raw = await client.get(self._key(query, max_results)) if client else None
        except Exception as exc:
            self._mark_down(exc)
            return None
        return [Evidence.model_validate(item) for item in json.loads(raw)] if raw else None

    async def set(self, query: str, max_results: int, results: list[Evidence]) -> None:
        now = datetime.now(timezone.utc)
        expiries = [item.valid_until for item in results if item.valid_until is not None]
        ttl = min([self.MAX_TTL_SECONDS, *((expiry - now).total_seconds() for expiry in expiries)])
        if not results or ttl < 60:
            return
        try:
            client = await self._redis()
            if client:
                payload = json.dumps([item.model_dump(mode="json") for item in results], ensure_ascii=False)
                await client.set(self._key(query, max_results), payload, ex=int(ttl))
        except Exception as exc:
            self._mark_down(exc)


class FallbackSearchAdapter:
    """按顺序尝试多个搜索源：前一个报错（未配置、额度用完、超时）就换下一个。"""

    def __init__(self, adapters: list[SearchAdapter], cache: RedisSearchCache | None = None):
        self.adapters = adapters
        self.cache = cache

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        if self.cache is not None and (cached := await self.cache.get(query, max_results)) is not None:
            return cached
        last_error: Exception | None = None
        for adapter in self.adapters:
            try:
                results = await adapter.search(query, max_results)
            except Exception as exc:
                last_error = exc
                continue
            if self.cache is not None:
                await self.cache.set(query, max_results, results)
            return results
        raise ExternalServiceError(f"所有搜索源都不可用：{type(last_error).__name__ if last_error else '未配置'}")


def default_search_adapter() -> FallbackSearchAdapter | None:
    """按配置组装搜索链：Tavily 在前（质量更好），SearXNG 兜底。都没配时返回 None。"""
    adapters: list[SearchAdapter] = []
    if settings.tavily_api_key:
        adapters.append(TavilySearchAdapter())
    if settings.searxng_url:
        adapters.append(SearxngSearchAdapter())
    return FallbackSearchAdapter(adapters, cache=RedisSearchCache()) if adapters else None
