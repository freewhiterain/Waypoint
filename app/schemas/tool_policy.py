"""Explicit read-only tool policies for domain subagents."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.planning import TaskType


_POLICIES: dict[TaskType, frozenset[str]] = {
    "weather": frozenset({"weather_mcp", "weather_fallback_api"}),
    "transport": frozenset({"transport_mcp", "search_mcp"}),
    "attractions": frozenset({"local_rag", "search_mcp", "deep_research"}),
    "hotel": frozenset({"local_rag", "hotel_mcp", "search_mcp", "deep_research"}),
    "food": frozenset({"local_rag", "search_mcp", "deep_research"}),
}


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """The capabilities a domain subagent may request."""

    worker: TaskType
    allowed_tools: frozenset[str]
    allow_deep_research: bool

    @classmethod
    def for_worker(cls, worker: TaskType) -> "ToolPolicy":
        if worker not in _POLICIES:
            raise ValueError(f"Unsupported subagent worker: {worker}")
        allowed = _POLICIES[worker]
        return cls(
            worker=worker,
            allowed_tools=allowed,
            allow_deep_research="deep_research" in allowed,
        )

