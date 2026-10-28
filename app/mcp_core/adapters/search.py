"""联网搜索适配器：Tavily、自建 SearXNG，以及按顺序降级的组合。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

import httpx

from app.config import settings
from app.mcp_core.reliability import ExternalServiceError, ResilientExecutor
from app.schemas.planning import Evidence


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


class FallbackSearchAdapter:
    """按顺序尝试多个搜索源：前一个报错（未配置、额度用完、超时）就换下一个。"""

    def __init__(self, adapters: list[SearchAdapter]):
        self.adapters = adapters

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        last_error: Exception | None = None
        for adapter in self.adapters:
            try:
                return await adapter.search(query, max_results)
            except Exception as exc:
                last_error = exc
        raise ExternalServiceError(f"所有搜索源都不可用：{type(last_error).__name__ if last_error else '未配置'}")


def default_search_adapter() -> FallbackSearchAdapter | None:
    """按配置组装搜索链：Tavily 在前（质量更好），SearXNG 兜底。都没配时返回 None。"""
    adapters: list[SearchAdapter] = []
    if settings.tavily_api_key:
        adapters.append(TavilySearchAdapter())
    if settings.searxng_url:
        adapters.append(SearxngSearchAdapter())
    return FallbackSearchAdapter(adapters) if adapters else None
