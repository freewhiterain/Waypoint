"""Safe normalization for read-only subagent tool results."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.mcp_core.client import get_mcp_client
from app.schemas.planning import Evidence, TaskType
from app.schemas.events import EvidenceSufficiency
from app.utils.logger import app_logger


class ProviderError(BaseModel):
    """A safe, typed provider failure without raw upstream payloads."""

    model_config = ConfigDict(frozen=True)

    provider: str
    code: Literal["unavailable", "invalid_response", "empty_result"]
    message: str = Field(min_length=1)


class NormalizedToolResult(list[Evidence | ProviderError]):
    """List-compatible normalized output with typed sufficiency metadata."""

    def __init__(
        self,
        values: list[Evidence | ProviderError] | None = None,
        *,
        sufficiency: EvidenceSufficiency,
    ) -> None:
        super().__init__(values or [])
        self.sufficiency = sufficiency


@dataclass(slots=True)
class ReadOnlyTool:
    """Adapter that prevents raw provider results from reaching a subagent."""

    name: str
    provider: str
    _handler: Callable[[Any], Any]

    async def ainvoke(self, payload: Any) -> NormalizedToolResult:
        try:
            result = self._handler(payload)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            return normalize_tool_result(self.provider, exc)
        return normalize_tool_result(self.provider, result)


def _error(provider: str, code: Literal["unavailable", "invalid_response", "empty_result"]) -> ProviderError:
    return ProviderError(provider=provider, code=code, message=f"{provider} provider is unavailable")


def _normalized(
    provider: str,
    values: list[Evidence | ProviderError],
    *,
    status: str | None = None,
    reason_code: str | None = None,
) -> NormalizedToolResult:
    evidence_count = sum(isinstance(item, Evidence) for item in values)
    error = next((item for item in values if isinstance(item, ProviderError)), None)
    if status in {"failed", "unavailable"} or error is not None and error.code != "empty_result":
        resolved_status = "failed"
    elif status in {"partial", "incomplete"}:
        resolved_status = "partial"
    elif evidence_count:
        resolved_status = "sufficient"
    else:
        resolved_status = "empty"
    resolved_reason = reason_code or {
        "sufficient": "provider_complete",
        "partial": "provider_partial",
        "empty": "provider_empty",
        "failed": "provider_failed",
    }[resolved_status]
    return NormalizedToolResult(
        values,
        sufficiency=EvidenceSufficiency(
            status=resolved_status,
            evidence_count=evidence_count,
            reason_code=resolved_reason,
        ),
    )


_HIDDEN_METADATA_KEYS = {
    "api_key", "authorization", "headers", "internal_trace", "password",
    "raw", "request", "response", "secret", "token",
}


def _sanitize_evidence(item: Evidence | Mapping[str, Any]) -> Evidence:
    if isinstance(item, Evidence):
        data = item.model_dump(mode="python")
    else:
        data = dict(item)
    metadata = data.get("metadata") or {}
    if isinstance(metadata, Mapping):
        data["metadata"] = {
            key: value
            for key, value in metadata.items()
            if str(key).casefold() not in _HIDDEN_METADATA_KEYS
        }
    else:
        data["metadata"] = {}
    return Evidence.model_validate(data)


def normalize_tool_result(provider: str, payload: Any) -> NormalizedToolResult:
    """Keep only validated evidence or a sanitized provider error."""

    if isinstance(payload, BaseException):
        return _normalized(provider, [_error(provider, "unavailable")])

    raw_items: Any = payload
    provider_status: str | None = None
    if isinstance(payload, BaseModel):
        raw_items = payload.model_dump(mode="python")
    if isinstance(raw_items, Mapping):
        provider_status = str(raw_items.get("status")) if raw_items.get("status") is not None else None
        if provider_status in {"failed", "unavailable"} or raw_items.get("error"):
            return _normalized(provider, [_error(provider, "unavailable")], status="failed")
        raw_items = raw_items.get("evidence", [])

    if isinstance(raw_items, Evidence):
        return _normalized(provider, [_sanitize_evidence(raw_items)], status=provider_status)
    if not isinstance(raw_items, list):
        return _normalized(provider, [_error(provider, "invalid_response")], status="failed")
    if not raw_items:
        return _normalized(provider, [_error(provider, "empty_result")], status=provider_status)

    evidence: list[Evidence] = []
    for item in raw_items:
        try:
            evidence.append(_sanitize_evidence(item))
        except Exception:
            return _normalized(provider, [_error(provider, "invalid_response")], status="failed")
    return _normalized(provider, evidence, status=provider_status)


async def _graph_evidence(
    graph: Any,
    destination: str,
    category: Any,
    query: str,
) -> list[Any]:
    """查图谱关系；失败只降级为"没有图谱补充"，不影响文档证据。

    图谱是补充信号：库里没数据、表还没建、连接抖动都属于常态。让它把整个
    local_rag provider 打成 failed，会连带触发不必要的 Deep Search 补搜。
    """
    if graph is None:
        return []
    try:
        return list(await graph.search_related_entities(destination, category, query))
    except Exception as exc:
        app_logger.warning(f"图谱证据不可用，仅使用文档证据：{type(exc).__name__}: {exc}")
        return []


def _merge_evidence(document_evidence: Any, graph_evidence: list[Any]) -> list[Any]:
    """文档证据在前，图谱补在后；按 content 去重。

    同一句话可能既在原文里又被 graph_extraction 抽成关系，去重避免它在 LLM
    prompt 的证据清单里出现两遍（重复证据会让模型误判该事实更可信）。
    """
    merged: list[Any] = []
    seen: set[str] = set()
    for item in [*list(document_evidence or []), *graph_evidence]:
        content = getattr(item, "content", None)
        key = content if isinstance(content, str) else repr(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


async def build_subagent_tools(
    worker: TaskType,
    policy: "ToolPolicy | None" = None,
    *,
    mcp_manager: Any | None = None,
    knowledge: Any | None = None,
    graph: Any | None = None,
    weather_adapter: Any | None = None,
    search_adapter: Any | None = None,
    deep_research_service: Any | None = None,
) -> list[Any]:
    """Build normalized MCP and non-MCP tools for one domain worker."""

    from app.schemas.tool_policy import ToolPolicy

    canonical = ToolPolicy.for_worker(worker)
    if policy is not None and policy.worker != worker:
        raise ValueError("Tool policy worker does not match requested worker")
    manager = mcp_manager
    if manager is None and settings.enable_external_tools:
        manager = await get_mcp_client()
    mcp_tools = await manager.get_allowed_tools(canonical.allowed_tools) if manager is not None else []
    tools: list[ReadOnlyTool] = []
    mcp_providers = {
        "get_weather": "weather_mcp",
        "get_transport": "transport_mcp",
        "search_hotels": "hotel_mcp",
        "get_hotel_availability": "hotel_mcp",
        "web_search": "search_mcp",
    }
    for tool in mcp_tools:
        name = getattr(tool, "name", "")
        provider = mcp_providers.get(name)
        if provider is None:
            continue

        async def invoke_mcp(payload: Any, *, source=tool) -> Any:
            method = getattr(source, "ainvoke", None) or getattr(source, "arun", None)
            if method is not None:
                return await method(payload)
            return source.invoke(payload)

        tools.append(ReadOnlyTool(name=name, provider=provider, _handler=invoke_mcp))

    if "local_rag" in canonical.allowed_tools:
        if knowledge is None:
            from app.rag.local_knowledge import get_local_knowledge_service
            knowledge = get_local_knowledge_service()
        if graph is None:
            from app.rag.graph_knowledge import get_graph_knowledge_service
            graph = get_graph_knowledge_service()

        async def invoke_rag(payload: Mapping[str, Any]) -> Any:
            destination = str(payload["destination"])
            category = payload.get("category", worker)
            query = str(payload["query"])
            document_evidence = knowledge.search_destination(destination, category, query)
            # 图谱与文档互补：不新增 provider 塞进 provider_order（那是降级链，
            # local_rag 一命中后面就 break，图谱永远轮不到），而是在这里合并，
            # 文档证据在前、图谱证据在后：[*doc, *graph]。
            graph_evidence = await _graph_evidence(graph, destination, category, query)
            return _merge_evidence(document_evidence, graph_evidence)

        tools.append(ReadOnlyTool(name="local_rag", provider="local_rag", _handler=invoke_rag))

    if "weather_fallback_api" in canonical.allowed_tools:
        if weather_adapter is None and settings.enable_external_tools:
            from app.mcp_core.adapters.weather import CombinedWeatherAdapter
            weather_adapter = CombinedWeatherAdapter()

        if weather_adapter is not None:
            async def invoke_weather(payload: Mapping[str, Any]) -> Any:
                return await weather_adapter.query(
                    str(payload["city"]), bool(payload.get("forecast", True))
                )

            tools.append(ReadOnlyTool(
                name="weather_fallback_api",
                provider="weather_fallback_api",
                _handler=invoke_weather,
            ))

    if "deep_research" in canonical.allowed_tools:
        if deep_research_service is None and (search_adapter is not None or settings.enable_external_tools):
            from app.research.deep_research import DeepResearchService
            if search_adapter is None:
                from app.mcp_core.adapters.search import TavilySearchAdapter
                search_adapter = TavilySearchAdapter()
            deep_research_service = DeepResearchService(search_adapter.search)

        if deep_research_service is not None:
            async def invoke_deep_research(payload: Mapping[str, Any]) -> Any:
                return await deep_research_service.research(
                    str(payload["query"]),
                    int(payload.get("max_results_per_query", 3)),
                )

            tools.append(ReadOnlyTool(
                name="deep_research",
                provider="deep_research",
                _handler=invoke_deep_research,
            ))

    return tools
