"""Registry for domain subagent workers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from app.agents.subagents.attractions import AttractionsSubagent
from app.agents.subagents.base import DomainSubagent, ToolBuilder
from app.agents.subagents.food import FoodSubagent
from app.agents.subagents.hotel import HotelSubagent
from app.agents.subagents.tools import build_subagent_tools
from app.agents.subagents.transport import TransportSubagent
from app.agents.subagents.weather import WeatherSubagent
from app.schemas.planning import ResearchTask, TaskType, TravelRequirement
from app.schemas.research import SubagentResponse
from app.utils.callables import supports_keyword


class SubagentRegistry:
    """Run the registered domain subagent for a research task."""

    def __init__(self, workers: dict[TaskType, DomainSubagent]):
        self._workers = dict(workers)

    @property
    def workers(self) -> dict[TaskType, DomainSubagent]:
        return dict(self._workers)

    async def run(
        self,
        task: ResearchTask,
        requirement: TravelRequirement,
        *,
        event_callback=None,
        prior_results=None,
    ) -> SubagentResponse:
        worker = self._workers.get(task.task_type)
        if worker is None:
            return SubagentResponse(
                task_id=task.id,
                worker=task.task_type,
                status="failed",
                summary="No registered domain subagent is available.",
                warnings=["subagent_error:worker_unregistered"],
            )
        kwargs = {}
        if event_callback is not None and supports_keyword(worker.run, "event_callback"):
            kwargs["event_callback"] = event_callback
        if prior_results is not None and supports_keyword(worker.run, "prior_results"):
            kwargs["prior_results"] = prior_results
        try:
            return await worker.run(task, requirement, **kwargs)
        except Exception as exc:
            return SubagentResponse(
                task_id=task.id,
                worker=task.task_type,
                status="failed",
                summary="Domain subagent execution failed.",
                warnings=["subagent_error:worker_execution_failed"],
            )


def create_default_subagent_registry(
    *,
    build_tools: ToolBuilder | None = build_subagent_tools,
    llm: Any | None = None,
    deep_search: Callable[..., Any] | None = None,
) -> SubagentRegistry:
    """Create the default registry without touching the legacy worker registry."""

    shared_kwargs = {
        "tool_builder": build_tools,
        "llm": llm,
        "deep_search": deep_search,
    }
    return SubagentRegistry(
        {
            "attractions": AttractionsSubagent(**shared_kwargs),
            "weather": WeatherSubagent(**shared_kwargs),
            "transport": TransportSubagent(**shared_kwargs),
            "hotel": HotelSubagent(**shared_kwargs),
            "food": FoodSubagent(**shared_kwargs),
        }
    )


__all__ = ["SubagentRegistry", "create_default_subagent_registry"]
