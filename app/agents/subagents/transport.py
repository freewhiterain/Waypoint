"""Transport domain subagent：城市内部地点之间的交通。

规划流程里交通是第二批任务：拿到景点、住宿、美食的候选后，算出它们两两之间
怎么走、要多久，交给排日程决定先后顺序。每段交通都作为一个带证据的候选输出，
所以照常经过证据治理。

对话里单独问交通（research_transport 工具）时没有前置候选，仍走原来的检索流程。
"""

from __future__ import annotations

from collections.abc import Iterable

from app.agents.subagents.base import DomainSubagent
from app.agents.travel_time import (
    Place,
    TravelTimeService,
    default_travel_time_service,
    describe_leg,
    place_from_option,
)
from app.schemas.planning import Evidence, ResearchTask, TravelRequirement, WorkerResult
from app.schemas.research import EvidenceBoundCandidate, SubagentResponse

# 地点太多时两两组合会爆炸（n 个地点 n(n-1)/2 段），每类只取前几个候选。
PLACE_LIMITS = {"attractions": 8, "hotel": 2, "food": 4}


class TransportSubagent(DomainSubagent):
    def __init__(self, *, travel_time: TravelTimeService | None = None, **kwargs):
        super().__init__(
            worker="transport",
            provider_order=("transport_mcp", "search_mcp"),
            **kwargs,
        )
        self.travel_time = travel_time or default_travel_time_service()

    def build_query(self, task: ResearchTask, requirement: TravelRequirement) -> str:
        preferences = " ".join(requirement.transport_preferences)
        origin = requirement.origin or "origin pending"
        return f"{origin} to {requirement.destination} {task.query} {preferences}".strip()

    async def run(
        self,
        task: ResearchTask,
        requirement: TravelRequirement,
        *,
        event_callback=None,
        prior_results: Iterable[WorkerResult] | None = None,
    ) -> SubagentResponse:
        if prior_results is None:
            return await super().run(task, requirement, event_callback=event_callback)
        if task.task_type != self.worker:
            return self._failure_response(
                task,
                f"Task type {task.task_type} does not match subagent {self.worker}.",
            )

        places = collect_places(prior_results)
        if len(places) < 2:
            return SubagentResponse(
                task_id=task.id,
                worker=self.worker,
                status="unavailable",
                summary="Fewer than two places are available for intra-city transport.",
                warnings=["transport_needs_places"],
            )

        if event_callback is not None:
            await event_callback("subagent_tool_call", {"tool_name": "travel_time", "round_number": 1})
        legs = await self.travel_time.compute(requirement.destination, places)
        places_by_id = {place.id: place for place in places}

        evidence: list[Evidence] = []
        candidates: list[EvidenceBoundCandidate] = []
        for leg in legs:
            origin, target = places_by_id[leg.from_id], places_by_id[leg.to_id]
            text = describe_leg(leg, origin, target)
            evidence_id = f"{task.id}-{leg.leg_id}"
            evidence.append(
                Evidence(
                    id=evidence_id,
                    content=text,
                    source=leg.source,
                    retrieved_at=leg.retrieved_at,
                    confidence=0.8 if leg.minutes is not None else 0.3,
                    metadata={"source_type": "synthetic", "provider": leg.source},
                )
            )
            candidates.append(
                EvidenceBoundCandidate(
                    id=leg.leg_id,
                    name=f"{origin.name} → {target.name}",
                    category="travel_leg",
                    description=text,
                    attributes=leg.model_dump(mode="json"),
                    evidence_ids=[evidence_id],
                )
            )

        if event_callback is not None:
            await event_callback(
                "subagent_tool_completed",
                {
                    "tool_name": "travel_time",
                    "round_number": 1,
                    "status": "sufficient",
                    "evidence_count": len(evidence),
                },
            )
        timed = sum(1 for leg in legs if leg.minutes is not None)
        warnings = [] if timed == len(legs) else ["travel_time_estimated:no_route_data"]
        return SubagentResponse(
            task_id=task.id,
            worker=self.worker,
            status="completed" if timed == len(legs) else "partial",
            summary=f"{len(legs)} intra-city legs between {len(places)} places; {timed} with route times.",
            candidates=candidates,
            evidence=evidence,
            warnings=warnings,
        )


def collect_places(results: Iterable[WorkerResult]) -> list[Place]:
    places: list[Place] = []
    seen: set[str] = set()
    for result in results:
        limit = PLACE_LIMITS.get(result.worker)
        if limit is None:
            continue
        for option in result.options[:limit]:
            if option.id in seen:
                continue
            seen.add(option.id)
            places.append(place_from_option(option).model_copy(update={"category": result.worker}))
    return places


__all__ = ["TransportSubagent", "collect_places"]
