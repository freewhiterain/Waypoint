"""Phase 2 end-to-end coverage for the Chengdu local mock RAG workflow.

Covers the full chain declared in
docs/superpowers/plans/2026-07-23-trip-agent-phase2-mock-rag-implementation.md (Task 7):

    confirmed form -> Supervisor exactly once -> five category-scoped RAG
    queries -> five Worker Agent results -> evidence-backed mock draft

plus the two required degradation paths: missing category fixtures and
Worker Agent/LLM failure.
"""

from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agents.supervisor import run_travel_planning
from app.agents.workers import WorkerRegistry
from app.agents.workers.attractions import AttractionsWorker
from app.agents.workers.base import TravelWorker
from app.agents.workers.food import FoodWorker
from app.agents.workers.hotel import HotelWorker
from app.agents.workers.local_knowledge import LocalKnowledgeService
from app.agents.workers.transport import TransportWorker
from app.agents.workers.weather import WeatherWorker
from app.rag.document_loader import DocumentManager
from app.schemas.planning import ResearchTask, TravelRequirement, WorkerResult

from tests.test_trip_form_tool_flow import (
    InMemoryInvocationRepository,
    configure_endpoint,
    endpoint_client,
    invocation,
    parse_sse,
)


WORKER_CLASSES = {
    "attractions": AttractionsWorker,
    "weather": WeatherWorker,
    "transport": TransportWorker,
    "hotel": HotelWorker,
    "food": FoodWorker,
}


def chengdu_requirement(**overrides) -> TravelRequirement:
    fields = dict(
        origin="上海",
        destination="成都",
        departure_date=date(2026, 9, 1),
        days=3,
    )
    fields.update(overrides)
    return TravelRequirement(**fields)


def build_local_registry(knowledge: LocalKnowledgeService, llm=None) -> WorkerRegistry:
    return WorkerRegistry(
        {
            name: worker_cls(knowledge=knowledge, llm=llm)
            for name, worker_cls in WORKER_CLASSES.items()
        }
    )


def documents_excluding_category(category: str):
    return [
        document
        for document in DocumentManager().load_all_documents()
        if document.metadata.get("category") != category
    ]


class FailingLlm:
    """Structured LLM stand-in that always fails, forcing deterministic fallback."""

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        raise RuntimeError("model unavailable")


class ExplodingWorker(TravelWorker):
    """Worker stand-in that raises before producing any result."""

    async def run(self, task: ResearchTask, requirement: TravelRequirement) -> WorkerResult:
        raise RuntimeError("upstream dependency crashed")


@pytest.mark.asyncio
async def test_confirmed_form_runs_supervisor_once_across_five_category_scoped_workers(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    configure_endpoint(monkeypatch, repository)
    from app.agents import factory
    from app.api.v1 import tools

    # 这条用例断言的是本地 mock RAG 链路（transport 无出发地时 partial、
    # 全部证据 source_type=mock_markdown），只有 legacy WorkerRegistry 会这样。
    # TRAVEL_AGENT_MODE 的默认值后来改成了 supervisor_subagents，用例没有钉住
    # 模式，于是开始沉默地跑另一条注册表。按仓库既有约定显式钉住模式。
    monkeypatch.setattr(factory.settings, "travel_agent_mode", "supervisor")

    calls = []
    real_run_travel_planning = tools.run_travel_planning

    async def counting_run_travel_planning(requirement, **kwargs):
        calls.append(requirement)
        return await real_run_travel_planning(requirement, **kwargs)

    monkeypatch.setattr(tools, "run_travel_planning", counting_run_travel_planning, raising=False)

    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "成都", "departure_date": "2026-09-01", "days": 3},
    }

    async with endpoint_client(user) as client:
        response = await client.post("/api/v1/chat/tools/call-1/result", json=payload)
        duplicate_response = await client.post("/api/v1/chat/tools/call-1/result", json=payload)

    stream = parse_sse(response)
    duplicate_stream = parse_sse(duplicate_response)

    # Supervisor runs exactly once even though the confirmed form was submitted twice.
    assert len(calls) == 1
    assert [event["type"] for event in stream][-3:] == ["result", "token", "done"]
    assert "subagent_started" in [event["type"] for event in stream]
    assert [event["type"] for event in duplicate_stream] == ["result", "token", "done"]
    result_event = next(event for event in stream if event["type"] == "result")
    assert duplicate_stream[0]["payload"]["result"] == result_event["payload"]["result"]

    draft = result_event["payload"]["result"]
    worker_results = {result["worker"]: result for result in draft["worker_results"]}

    # All five category Workers ran and are clearly marked as local mock results.
    assert set(worker_results) == set(WORKER_CLASSES)
    assert all(result["is_mock"] for result in worker_results.values())
    assert all(result["status"] != "failed" for result in worker_results.values())

    # The trip form contract only carries destination/departure_date/days, so origin
    # stays unset; TransportWorker must degrade to "partial" without inventing options.
    assert worker_results["transport"]["status"] == "partial"
    assert worker_results["transport"]["evidence"] == []
    assert worker_results["transport"]["options"] == []

    # The other four categories have matching Chengdu fixtures and must retrieve evidence.
    for worker_name in ("attractions", "weather", "hotel", "food"):
        result = worker_results[worker_name]
        assert result["status"] == "completed"
        assert result["evidence"], f"{worker_name} worker must retrieve grounded evidence"

    assert draft["evidence"], "assembled draft must keep flattened evidence"
    assert all(item["metadata"].get("source_type") == "mock_markdown" for item in draft["evidence"])


@pytest.mark.asyncio
async def test_missing_category_fixture_and_llm_failure_degrade_without_losing_other_workers():
    """Task 7 Step 2: remove one category's fixture and force the structured LLM call to fail."""
    documents = documents_excluding_category("hotel")
    knowledge = LocalKnowledgeService(documents=documents)
    registry = build_local_registry(knowledge, llm=FailingLlm())

    draft = await run_travel_planning(chengdu_requirement(), registry=registry)
    worker_results = {result.worker: result for result in draft.worker_results}

    assert set(worker_results) == set(WORKER_CLASSES)
    assert all(result.status != "failed" for result in worker_results.values())

    # Removed fixture -> no evidence -> unavailable, with an explicit warning, no options.
    hotel_result = worker_results["hotel"]
    assert hotel_result.status == "unavailable"
    assert hotel_result.options == []
    assert hotel_result.evidence == []
    assert "No evidence is available for this analysis." in hotel_result.warnings

    # Categories that still have evidence must survive the structured-LLM failure by
    # falling back to a deterministic, evidence-grounded summary.
    for worker_name in ("attractions", "weather", "food"):
        result = worker_results[worker_name]
        assert result.evidence
        assert result.status == "completed"
        assert any("降级为证据摘要" in warning for warning in result.warnings)
        for option in result.options:
            # No unsupported concrete option: every fallback option cites its evidence
            # source and never fabricates a price.
            assert option.attributes.get("source")
            assert option.estimated_cost is None

    assert draft.warnings
    assert all(result.summary for result in worker_results.values())


@pytest.mark.asyncio
async def test_single_worker_exception_does_not_block_other_worker_results():
    knowledge = LocalKnowledgeService()
    registry = build_local_registry(knowledge)
    registry._workers["weather"] = ExplodingWorker()

    draft = await run_travel_planning(chengdu_requirement(), registry=registry)
    worker_results = {result.worker: result for result in draft.worker_results}

    assert set(worker_results) == set(WORKER_CLASSES)
    assert worker_results["weather"].status == "failed"
    assert "upstream dependency crashed" in worker_results["weather"].warnings[0]

    for worker_name in ("attractions", "hotel", "food"):
        result = worker_results[worker_name]
        assert result.status == "completed"
        assert result.evidence
        assert result.is_mock is True

    # The overall draft still assembles: itinerary generation is not blocked
    # by a single failed Worker.
    assert draft.itinerary
