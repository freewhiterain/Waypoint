"""交通子 Agent 改为计算市内地点两两之间的交通，并作为第二批任务运行。"""

from datetime import date

import pytest

from app.agents.subagents.registry import create_default_subagent_registry
from app.agents.subagents.transport import TransportSubagent, collect_places
from app.agents.supervisor import run_travel_planning
from app.agents.travel_time import (
    DistrictEstimateProvider,
    Place,
    TravelLeg,
    TravelTimeService,
    place_from_option,
)
from app.schemas.planning import CandidateOption, ResearchTask, TravelRequirement, WorkerResult
from app.schemas.research import EvidenceBoundCandidate, SubagentResponse
from app.schemas.planning import Evidence


def _requirement() -> TravelRequirement:
    return TravelRequirement(destination="成都", departure_date=date(2026, 11, 14), days=2)


def _option(name: str, category: str, description: str, **attributes) -> CandidateOption:
    return CandidateOption(
        id=f"{category}-{name}",
        name=name,
        category=category,
        description=description,
        attributes=attributes,
        evidence_ids=["e"],
    )


def _result(worker: str, options: list[CandidateOption]) -> WorkerResult:
    return WorkerResult(task_id=f"{worker}-task", worker=worker, status="completed", summary="", options=options)


def test_place_district_comes_from_attributes_or_description():
    assert place_from_option(_option("熊猫基地", "attractions", "位于成华区。")).district == "成华区"
    assert place_from_option(_option("宽窄巷子", "attractions", "", location="青羊区")).district == "青羊区"
    assert place_from_option(_option("某店", "food", "很好吃")).district is None


@pytest.mark.asyncio
async def test_district_estimate_compares_districts_without_inventing_minutes():
    places = [
        Place(id="a", name="武侯祠", category="attractions", district="武侯区"),
        Place(id="b", name="锦里", category="attractions", district="武侯区"),
        Place(id="c", name="熊猫基地", category="attractions", district="成华区"),
        Place(id="d", name="某店", category="food"),
    ]

    legs = await TravelTimeService([DistrictEstimateProvider()]).compute("成都", places)

    assert len(legs) == 6  # 4 个地点两两组合
    by_pair = {frozenset((leg.from_id, leg.to_id)): leg for leg in legs}
    assert by_pair[frozenset("ab")].proximity == "same_district"
    assert by_pair[frozenset("ac")].proximity == "different_district"
    assert by_pair[frozenset("ad")].proximity == "unknown"
    assert all(leg.minutes is None for leg in legs)


class _PartialRouteProvider:
    """模拟图谱/高德：只答得上 a-b 这一对，其余交给兜底。"""

    name = "route_graph"

    async def legs(self, city, pairs):
        return [
            TravelLeg(from_id=a.id, to_id=b.id, from_name=a.name, to_name=b.name,
                      minutes=25, distance_km=8.0, mode="地铁", source=self.name)
            for a, b in pairs
            if {a.id, b.id} == {"a", "b"}
        ]


class _BrokenProvider:
    name = "amap"

    async def legs(self, city, pairs):
        raise TimeoutError("amap down")


@pytest.mark.asyncio
async def test_providers_are_tried_in_order_and_failures_fall_through():
    places = [Place(id=x, name=x, category="attractions", district="武侯区") for x in "abc"]
    service = TravelTimeService([_PartialRouteProvider(), _BrokenProvider(), DistrictEstimateProvider()])

    legs = await service.compute("成都", places)

    sources = {frozenset((leg.from_id, leg.to_id)): leg.source for leg in legs}
    assert sources == {
        frozenset("ab"): "route_graph",
        frozenset("ac"): "district_estimate",
        frozenset("bc"): "district_estimate",
    }


@pytest.mark.asyncio
async def test_transport_subagent_turns_prior_places_into_evidence_bound_legs():
    prior = [
        _result("attractions", [_option("熊猫基地", "attractions", "位于成华区。"),
                                _option("宽窄巷子", "attractions", "位于青羊区。")]),
        _result("hotel", [_option("武侯酒店", "hotel", "位于武侯区。")]),
        _result("weather", [_option("晴", "weather", "")]),  # 天气不是地点，不参与
    ]
    task = ResearchTask(task_type="transport", query="市内交通")

    response = await TransportSubagent(tool_builder=None).run(task, _requirement(), prior_results=prior)

    assert len(collect_places(prior)) == 3
    assert len(response.candidates) == 3
    assert {candidate.category for candidate in response.candidates} == {"travel_leg"}
    evidence_ids = {item.id for item in response.evidence}
    assert all(set(candidate.evidence_ids) <= evidence_ids for candidate in response.candidates)
    # 只有区级粗估、没有真实耗时，必须标成 partial 并给出提示。
    assert response.status == "partial"
    assert "travel_time_estimated:no_route_data" in response.warnings


@pytest.mark.asyncio
async def test_transport_without_places_is_unavailable():
    task = ResearchTask(task_type="transport", query="市内交通")

    response = await TransportSubagent(tool_builder=None).run(task, _requirement(), prior_results=[])

    assert response.status == "unavailable"


class _PlaceSubagent:
    def __init__(self, worker, names):
        self.worker, self.names = worker, names

    async def run(self, task, requirement):
        evidence = [Evidence(id=f"{self.worker}-{name}", content=f"{name}位于武侯区。", source="local",
                             metadata={"source_type": "local"}) for name in self.names]
        return SubagentResponse(
            task_id=task.id,
            worker=self.worker,
            status="completed",
            candidates=[
                EvidenceBoundCandidate(id=f"{self.worker}-{name}", name=name, category=self.worker,
                                       description=f"{name}位于武侯区。", evidence_ids=[f"{self.worker}-{name}"])
                for name in self.names
            ],
            evidence=evidence,
        )


@pytest.mark.asyncio
async def test_supervisor_runs_transport_after_places_and_keeps_its_legs():
    registry = create_default_subagent_registry(build_tools=None)
    workers = registry.workers
    workers["attractions"] = _PlaceSubagent("attractions", ["武侯祠", "锦里"])
    workers["hotel"] = _PlaceSubagent("hotel", ["武侯酒店"])
    workers["food"] = _PlaceSubagent("food", [])
    workers["weather"] = _PlaceSubagent("weather", [])
    registry = type(registry)(workers)

    draft = await run_travel_planning(_requirement(), registry=registry)

    transport = next(result for result in draft.worker_results if result.worker == "transport")
    assert len(transport.options) == 3  # 3 个地点两两之间
    assert all(option.attributes["proximity"] == "same_district" for option in transport.options)
