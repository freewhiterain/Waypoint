"""Planning factories: build the subagent registry and run the supervisor."""

from typing import Any

from app.config import DEGRADED_PLANNING_MARKER, DEGRADED_PLANNING_REASON, settings


def _planning_llm() -> Any | None:
    """规划用的 LLM；未配置 Key 或初始化失败时返回 None。"""
    if not settings.llm_api_key.strip():
        return None
    try:
        from app.agents.llm import get_llm

        return get_llm()
    except Exception:
        return None


def create_planning_registry() -> tuple[Any, str | None]:
    """Build the subagent registry used by real planning requests.

    没有可用的 LLM 时子 Agent 退回"直接用证据摘要"的确定性分析，
    同时返回降级原因，让调用方把结果标成 degraded。
    """
    from app.agents.subagents.registry import create_default_subagent_registry

    from app.places.catalog import get_place_catalog

    llm = _planning_llm()
    return create_default_subagent_registry(llm=llm, place_catalog=get_place_catalog()), None if llm is not None else DEGRADED_PLANNING_REASON


async def run_travel_planning(requirement, **kwargs):
    """Run planning through the configured production registry."""
    from app.agents.supervisor import run_travel_planning as supervisor_runner

    registry = kwargs.pop("registry", None)
    fallback_reason = kwargs.pop("fallback_reason", None)
    if registry is None:
        registry, fallback_reason = create_planning_registry()
    kwargs.setdefault("planning_llm", _planning_llm())

    draft = await supervisor_runner(requirement, registry=registry, **kwargs)
    if fallback_reason is None:
        return draft

    reason = draft.degraded_reason or fallback_reason
    marker = DEGRADED_PLANNING_MARKER if reason == DEGRADED_PLANNING_REASON else f"planning_degraded:{reason}"
    warnings = list(dict.fromkeys([*draft.warnings, marker]))
    return draft.model_copy(
        update={
            "status": "degraded",
            "degraded_reason": reason,
            "warnings": warnings,
        }
    )
