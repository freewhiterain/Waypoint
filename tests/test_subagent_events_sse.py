import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from app.agents import supervisor as supervisor_module
from app.agents.subagents.base import DomainSubagent
from app.agents.subagents.registry import SubagentRegistry
from app.agents.subagents.tools import ReadOnlyTool
from app.api.v1 import tools
from app.governance.tool_invocations import ProcessingOutcome, ToolInvocationRecord
from app.research.deep_search import DeepSearchEvaluation, run_deep_search
from app.schemas.planning import Evidence
from app.schemas.research import ResearchConflict, SubagentResponse
from app.schemas.tools import ToolResultRequest


class InMemoryProcessingRepository:
    def __init__(self):
        self.record = ToolInvocationRecord(
            call_id="call-subagent-events",
            user_id="user-1",
            conversation_id="conversation-1",
            tool="collect_trip_requirements",
            status="pending",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    async def claim_processing(self, call_id, user_id, _lease_timeout):
        assert call_id == self.record.call_id
        assert user_id == self.record.user_id
        self.record.status = "processing"
        self.record.version += 1
        return ProcessingOutcome(
            record=self.record.model_copy(deep=True),
            claimed=True,
            claim_version=self.record.version,
        )

    async def finish_processing(self, call_id, user_id, expected_version, durable_result, session=None):
        assert call_id == self.record.call_id
        assert user_id == self.record.user_id
        assert expected_version == self.record.version
        self.record.status = "completed"
        self.record.result = durable_result
        self.record.version += 1
        return self.record.model_copy(deep=True)

    async def release_processing(self, call_id, user_id, expected_version):
        assert call_id == self.record.call_id
        assert user_id == self.record.user_id
        assert expected_version == self.record.version
        self.record.status = "pending"
        self.record.version += 1
        return self.record.model_copy(deep=True)

    async def renew_processing(self, call_id, user_id, expected_version):
        return (
            call_id == self.record.call_id
            and user_id == self.record.user_id
            and expected_version == self.record.version
            and self.record.status == "processing"
        )


class RecordingEventRepository:
    def __init__(self):
        self.events = []

    async def append(self, event):
        event.sequence = len(self.events) + 1
        self.events.append(event)
        return event

    async def list(self, task_id, user_id):
        return [
            event
            for event in self.events
            if event.task_id == task_id and event.user_id == user_id
        ]


class RecordingSession:
    def __init__(self):
        self.messages = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def begin(self):
        return self

    def add(self, message):
        self.messages.append(message)


class FakeDraft:
    def model_dump(self, *, mode):
        assert mode == "json"
        return {"destination": "Kyoto", "summary": "confirmed itinerary"}


class EmptyWorker:
    async def run(self, task, requirement):
        return SubagentResponse(
            task_id=task.id,
            worker=task.task_type,
            status="unavailable",
            summary="No evidence available.",
            warnings=["worker has no provider"],
        )


class FollowUpEvaluator:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, state):
        self.calls += 1
        return DeepSearchEvaluation(
            needs_follow_up=self.calls == 1,
            follow_up_query="official follow-up query",
            conflicts=(
                [
                    ResearchConflict(
                        fact_key="opening_status",
                        values=["open", "closed"],
                        evidence_ids=["search-1", "search-2"],
                        description="hidden conflict details",
                    )
                ]
                if self.calls == 2
                else []
            ),
        )


@asynccontextmanager
async def session_factory(session):
    yield session


def parse_sse(frame):
    return json.loads(frame.removeprefix("data: ").strip())


async def run_fake_planning_stream(monkeypatch):
    repository = InMemoryProcessingRepository()
    event_repository = RecordingEventRepository()
    session = RecordingSession()

    monkeypatch.setattr(tools, "PostgresToolInvocationRepository", lambda: repository)
    monkeypatch.setattr(tools, "PostgresEventRepository", lambda: event_repository)
    monkeypatch.setattr(tools, "async_session_maker", lambda: session)

    async def fake_checkpointer():
        return None

    async def fake_supervisor(requirement, **kwargs):
        await kwargs["event_service"].emit(
            task_id=kwargs["task_id"],
            user_id=kwargs["user_id"],
            conversation_id=kwargs["conversation_id"],
            event_type="worker_started",
            payload={
                "task_id": "research-attractions",
                "worker": "attractions",
                "hidden_reasoning": "do not stream this",
            },
        )
        await kwargs["event_service"].emit(
            task_id=kwargs["task_id"],
            user_id=kwargs["user_id"],
            conversation_id=kwargs["conversation_id"],
            event_type="evidence_collected",
            payload={
                "task_id": "research-attractions",
                "worker": "attractions",
                "count": 2,
                "evidence": [{"content": "secret source excerpt"}],
            },
        )
        await kwargs["event_service"].emit(
            task_id=kwargs["task_id"],
            user_id=kwargs["user_id"],
            conversation_id=kwargs["conversation_id"],
            event_type="worker_completed",
            payload={
                "task_id": "research-attractions",
                "worker": "attractions",
                "status": "completed",
                "summary": "hidden chain of thought",
                "evidence": [{"content": "secret evidence"}],
                "warnings": ["Deep Search stopped because max rounds was reached."],
                "conflicts": [{"details": "secret conflict evidence"}],
            },
        )
        return FakeDraft()

    monkeypatch.setattr(tools, "get_checkpointer", fake_checkpointer)
    monkeypatch.setattr(tools, "run_travel_planning", fake_supervisor)

    data = ToolResultRequest.model_validate(
        {
            "tool": "collect_trip_requirements",
            "status": "completed",
            "result": {
                "destination": "Kyoto",
                "departure_date": "2026-08-03",
                "days": 4,
            },
        }
    )
    return [
        parse_sse(frame)
        async for frame in tools.tool_result_stream(
            repository.record.call_id,
            data,
            repository.record.user_id,
            repository.record.model_copy(deep=True),
        )
    ]


async def run_real_planning_event_stream(monkeypatch):
    repository = InMemoryProcessingRepository()
    event_repository = RecordingEventRepository()
    session = RecordingSession()
    search_queries = []

    async def empty_local_rag(_payload):
        return []

    async def search_provider(payload):
        search_queries.append(payload["query"])
        index = len(search_queries)
        return [
            Evidence(
                id=f"search-{index}",
                content=f"search result {index}",
                source="web",
                metadata={
                    "fact_key": "opening_status",
                    "fact_value": "open" if index == 1 else "closed",
                },
            )
        ]

    evaluator = FollowUpEvaluator()

    async def deep_search(request, *, search, event_callback=None):
        kwargs = {"event_callback": event_callback} if event_callback is not None else {}
        return await run_deep_search(request, search=search, evaluator=evaluator, **kwargs)

    attractions = DomainSubagent(
        worker="attractions",
        provider_order=("local_rag",),
        tools=[
            ReadOnlyTool(name="local_rag", provider="local_rag", _handler=empty_local_rag),
            ReadOnlyTool(name="web_search", provider="search_mcp", _handler=search_provider),
        ],
        deep_search=deep_search,
    )
    workers = {
        "attractions": attractions,
        **{worker: EmptyWorker() for worker in ("weather", "transport", "hotel", "food")},
    }
    registry = SubagentRegistry(workers)

    monkeypatch.setattr(tools, "PostgresToolInvocationRepository", lambda: repository)
    monkeypatch.setattr(tools, "PostgresEventRepository", lambda: event_repository)
    monkeypatch.setattr(tools, "async_session_maker", lambda: session)

    async def fake_checkpointer():
        return None

    async def real_supervisor(requirement, **kwargs):
        return await supervisor_module.run_travel_planning(
            requirement,
            registry=registry,
            **kwargs,
        )

    monkeypatch.setattr(tools, "get_checkpointer", fake_checkpointer)
    monkeypatch.setattr(tools, "run_travel_planning", real_supervisor)
    monkeypatch.setattr(supervisor_module.settings, "llm_api_key", "")

    data = ToolResultRequest.model_validate(
        {
            "tool": "collect_trip_requirements",
            "status": "completed",
            "result": {
                "destination": "Kyoto",
                "departure_date": "2026-08-03",
                "days": 4,
            },
        }
    )
    return [
        parse_sse(frame)
        async for frame in tools.tool_result_stream(
            repository.record.call_id,
            data,
            repository.record.user_id,
            repository.record.model_copy(deep=True),
        )
    ]


@pytest.mark.asyncio
async def test_subagent_events_keep_monotonic_sequence_and_legacy_fields(monkeypatch):
    events = await run_fake_planning_stream(monkeypatch)

    assert [event["type"] for event in events] == [
        "subagent_started",
        "evidence_collected",
        "subagent_completed",
        "result",
        "token",
        "done",
    ]
    assert [event["sequence"] for event in events] == sorted(event["sequence"] for event in events)
    assert events[-1]["type"] == "done"
    assert events[4]["content"] == '{"destination": "Kyoto", "summary": "confirmed itinerary"}'


@pytest.mark.asyncio
async def test_subagent_events_expose_only_public_typed_metadata(monkeypatch):
    events = await run_fake_planning_stream(monkeypatch)
    research_events = events[:3]
    serialized = json.dumps(research_events)

    assert "hidden" not in serialized
    assert "secret" not in serialized
    assert research_events[0]["payload"] == {
        "task_id": "research-attractions",
        "worker": "attractions",
    }
    assert research_events[1]["payload"] == {
        "task_id": "research-attractions",
        "worker": "attractions",
        "evidence_count": 2,
    }
    assert research_events[2]["payload"] == {
        "task_id": "research-attractions",
        "worker": "attractions",
        "status": "completed",
        "evidence_count": 1,
        "conflict_count": 1,
        "warning_codes": ["max_rounds"],
    }


@pytest.mark.asyncio
async def test_real_subagent_tool_follow_up_and_conflict_events_are_emitted(monkeypatch):
    events = await run_real_planning_event_stream(monkeypatch)
    research_events = [
        event
        for event in events
        if event["type"] in {"subagent_tool_call", "follow_up_search", "research_conflict"}
    ]
    event_types = [event["type"] for event in research_events]

    assert "subagent_tool_call" in event_types
    assert "follow_up_search" in event_types
    assert "research_conflict" in event_types

    tool_event = next(event for event in research_events if event["type"] == "subagent_tool_call")
    assert tool_event["payload"]["worker"] == "attractions"
    assert tool_event["payload"]["tool_name"] == "local_rag"
    assert tool_event["payload"]["round_number"] == 1
    task_id = tool_event["payload"]["task_id"]

    follow_up_event = next(event for event in research_events if event["type"] == "follow_up_search")
    assert follow_up_event["payload"] == {
        "task_id": task_id,
        "worker": "attractions",
        "tool_name": "web_search",
        "round_number": 2,
    }

    conflict_event = next(event for event in research_events if event["type"] == "research_conflict")
    assert conflict_event["payload"] == {
        "task_id": task_id,
        "worker": "attractions",
        "conflict_count": 1,
    }
    serialized = json.dumps(research_events)
    assert "hidden" not in serialized
    assert "opening_status" not in serialized
