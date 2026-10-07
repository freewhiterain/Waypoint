"""Mode-aware planning factories."""

from typing import Any

from app.config import (
    DEGRADED_PLANNING_MARKER,
    DEGRADED_PLANNING_REASON,
    SUPPORTED_TRAVEL_AGENT_MODES,
    settings,
)


def _travel_agent_mode() -> str:
    mode = settings.travel_agent_mode.strip().lower()
    if mode not in SUPPORTED_TRAVEL_AGENT_MODES:
        raise ValueError(
            f"TRAVEL_AGENT_MODE={settings.travel_agent_mode} is unsupported; "
            "use supervisor or supervisor_subagents."
        )
    return mode


def create_planning_registry() -> tuple[Any, str | None]:
    """Select the registry used by real planning requests.

    The legacy registry remains the default-compatible path. Subagents require
    an LLM configuration; without one, fallback is explicit and observable.
    """
    from app.agents.subagents.registry import create_default_subagent_registry
    from app.agents.workers.registry import create_default_registry

    mode = _travel_agent_mode()
    if mode == "supervisor":
        return create_default_registry(), None

    if not settings.llm_api_key.strip():
        reason = DEGRADED_PLANNING_REASON
        if settings.allow_legacy_fallback:
            return create_default_registry(), reason
        return create_default_subagent_registry(llm=None), reason

    try:
        from app.agents.llm import get_llm

        llm = get_llm()
    except Exception:
        if settings.allow_legacy_fallback:
            return create_default_registry(), DEGRADED_PLANNING_REASON
        return create_default_subagent_registry(llm=None), DEGRADED_PLANNING_REASON
    return create_default_subagent_registry(llm=llm), None


def _planning_llm() -> Any | None:
    """行程选点用的 LLM；未配置 Key 或初始化失败时返回 None，改用规则选点。"""
    if not settings.llm_api_key.strip():
        return None
    try:
        from app.agents.llm import get_llm

        return get_llm()
    except Exception:
        return None


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


async def create_chat_agent():
    """Create a graph with the registry selected for the configured mode."""
    registry, _fallback_reason = create_planning_registry()
    from app.agents.supervisor import create_supervisor_graph

    return create_supervisor_graph(registry=registry)
