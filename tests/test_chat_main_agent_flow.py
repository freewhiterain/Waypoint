import inspect
import json
from pathlib import Path
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from sqlalchemy.dialects import postgresql

from app.api.v1 import chat
from app.schemas.slots import SlotUpdate
from app.schemas.tools import MainAgentDecision


class RecordingSession:
    def __init__(self, history, *, fail_assistant_add=False):
        self.history = [HistoryMessage(**item) for item in history]
        self.saved_messages = []
        self.persisted_invocations = []
        self._staged_messages = []
        self._staged_invocations = []
        self._in_transaction = False
        self.fail_assistant_add = fail_assistant_add

    def add(self, entity):
        if self._in_transaction:
            if getattr(entity, "role", None) == "assistant" and self.fail_assistant_add:
                raise RuntimeError("assistant metadata write failed")
            self._staged_messages.append(entity)
            return
        self.saved_messages.append(entity)

    async def commit(self):
        pass

    async def refresh(self, message):
        if message.id is None:
            message.id = uuid.uuid4()

    async def execute(self, _statement):
        return MessageResult(self.history)

    def begin(self):
        return RecordingTransaction(self)

    def stage_invocation(self, record):
        self._staged_invocations.append(record)


class RecordingTransaction:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        self.session._in_transaction = True
        return self.session

    async def __aexit__(self, exc_type, exc, traceback):
        self.session._in_transaction = False
        if exc_type is None:
            self.session.saved_messages.extend(self.session._staged_messages)
            self.session.persisted_invocations.extend(self.session._staged_invocations)
        self.session._staged_messages.clear()
        self.session._staged_invocations.clear()
        return False


class HistoryMessage:
    def __init__(self, role, content, message_id=None, created_at=None):
        self.role = role
        self.content = content
        self.id = message_id or uuid.uuid4()
        self.created_at = created_at or datetime.now(timezone.utc)


class MessageResult:
    def __init__(self, messages):
        self.messages = messages

    def scalars(self):
        return self

    def all(self):
        return self.messages


class DecisionAgent:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    async def decide(self, message, context, slots=None):
        self.calls.append((message, context))
        return self.decision


class SessionAwareRecordingRepository:
    async def create_in_session(self, session, record):
        session.stage_invocation(record)
        return record


def session_factory(session):
    @asynccontextmanager
    async def factory():
        yield session

    return factory


async def stream_events(message="message"):
    return [
        json.loads(frame.removeprefix("data: ").strip())
        async for frame in chat.generate_sse_stream(
            str(uuid.uuid4()), message, str(uuid.uuid4())
        )
    ]


def configure_stream(monkeypatch, decision, history=None, *, fail_assistant_add=False):
    session = RecordingSession(history or [], fail_assistant_add=fail_assistant_add)
    agent = DecisionAgent(decision)
    repository = SessionAwareRecordingRepository()
    monkeypatch.setattr(chat, "async_session_maker", session_factory(session))
    monkeypatch.setattr(chat, "MainAgentService", lambda: agent)
    monkeypatch.setattr(chat, "PostgresToolInvocationRepository", lambda: repository)
    return session, agent, repository


@pytest.fixture
def supervisor_calls(monkeypatch):
    from app.agents import supervisor

    calls = []

    async def unexpected_supervisor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Supervisor must not be called")

    monkeypatch.setattr(supervisor, "run_travel_planning", unexpected_supervisor)
    return calls


@pytest.mark.asyncio
async def test_affirmation_persists_one_form_call_and_emits_tool_call(monkeypatch, supervisor_calls):
    session, agent, _ = configure_stream(
        monkeypatch,
        MainAgentDecision(
            action="collect_trip_requirements",
            reason="affirmed proactive offer",
        ),
        history=[{"role": "assistant", "content": "Need help planning a trip?"}],
    )

    events = await stream_events()

    assert [event["type"] for event in events] == ["tool_call", "done"]
    assert len(session.persisted_invocations) == 1
    record = session.persisted_invocations[0]
    assert events[0]["call_id"] == record.call_id
    assert events[0]["tool"] == "collect_trip_requirements"
    assert events[0]["arguments"] == {"initial_values": {}}
    assert record.arguments == {"initial_values": {}}
    assert uuid.UUID(record.user_id)
    assert uuid.UUID(record.conversation_id)
    assert agent.calls == [("message", [{"role": "assistant", "content": "Need help planning a trip?"}])]
    assert [message.role for message in session.saved_messages] == ["user", "assistant"]
    assert session.saved_messages[1].extra_info["action"] == "collect_trip_requirements"
    assert session.saved_messages[1].extra_info["tool_call"]["call_id"] == record.call_id
    assert supervisor_calls == []


@pytest.mark.asyncio
async def test_direct_planning_prepopulates_destination_without_supervisor(monkeypatch, supervisor_calls):
    session, _, _ = configure_stream(
        monkeypatch,
        MainAgentDecision(
            action="collect_trip_requirements",
            reason="explicit planning request",
            slot_update=SlotUpdate(destination="Chengdu", days=3),
        ),
    )

    events = await stream_events("plan a trip to Chengdu")

    assert events[0]["type"] == "tool_call"
    # 目的地在用户原话里，填入；天数在原话里找不到依据，被丢弃，留给表单去问。
    assert events[0]["arguments"] == {"initial_values": {"destination": "Chengdu"}}
    record = session.persisted_invocations[0]
    assert record.arguments == events[0]["arguments"]
    source = inspect.getsource(chat.generate_sse_stream)
    assert "TripCoordinator" not in source
    assert "PostgresDraftRepository" not in source
    assert "TripDraftRecord" not in source
    assert "hard_missing" not in source
    assert "classify_intent" not in source
    assert supervisor_calls == []


def test_legacy_slot_and_coordinator_interfaces_are_removed():
    workspace = Path(__file__).resolve().parents[1]

    assert not (workspace / "app" / "services" / "intent.py").exists()
    assert not (workspace / "app" / "agents" / "coordinator.py").exists()

    from app.schemas.events import SSEEvent
    from app.schemas.planning import TravelRequirementDraft
    from app.services import planning

    assert "ask" not in SSEEvent.model_fields["type"].annotation.__args__
    assert not hasattr(TravelRequirementDraft, "hard_missing")
    assert not hasattr(planning, "DESTINATION_QUICK_OPTIONS")
    assert not hasattr(planning, "merge_drafts")


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["answer_open_question", "recommend_destination"])
async def test_rag_actions_call_only_rag_and_persist_their_action(monkeypatch, action, supervisor_calls):
    session, _, _ = configure_stream(
        monkeypatch,
        MainAgentDecision(action=action, reason="RAG response"),
    )
    calls = []

    async def answer(question):
        calls.append(question)
        return "RAG answer"

    monkeypatch.setattr(chat, "answer_open_question", answer)
    monkeypatch.setattr(
        chat,
        "PostgresToolInvocationRepository",
        lambda: (_ for _ in ()).throw(AssertionError("form repository must not be used")),
    )

    events = await stream_events()

    assert [event["type"] for event in events] == ["token", "done"]
    assert events[0]["content"] == "RAG answer"
    assert events[0]["payload"]["action"] == action
    assert calls == ["message"]
    assert session.saved_messages[-1].extra_info["action"] == action
    assert supervisor_calls == []


@pytest.mark.asyncio
async def test_direct_response_calls_neither_rag_nor_supervisor(monkeypatch, supervisor_calls):
    session, _, _ = configure_stream(
        monkeypatch,
        MainAgentDecision(
            action="direct_response",
            reason="conversational",
            response="Hello there",
        ),
    )

    async def unexpected_rag(_question):
        raise AssertionError("RAG must not be used")

    monkeypatch.setattr(chat, "answer_open_question", unexpected_rag)
    monkeypatch.setattr(
        chat,
        "PostgresToolInvocationRepository",
        lambda: (_ for _ in ()).throw(AssertionError("form repository must not be used")),
    )

    events = await stream_events()

    assert [event["type"] for event in events] == ["token", "done"]
    assert events[0]["content"] == "Hello there"
    assert events[0]["payload"]["action"] == "direct_response"
    assert session.saved_messages[-1].extra_info["action"] == "direct_response"
    assert supervisor_calls == []


@pytest.mark.asyncio
async def test_collect_form_rolls_back_invocation_and_assistant_metadata_on_failure(monkeypatch, supervisor_calls):
    session, _, _ = configure_stream(
        monkeypatch,
        MainAgentDecision(
            action="collect_trip_requirements",
            reason="explicit planning request",
            slot_update=SlotUpdate(destination="Chengdu", days=3),
        ),
        fail_assistant_add=True,
    )

    events = await stream_events()

    assert [event["type"] for event in events] == ["error", "done"]
    assert session.persisted_invocations == []
    assert [message.role for message in session.saved_messages] == ["user"]
    assert supervisor_calls == []


class ContextStatementSession:
    def __init__(self, rows):
        self.rows = rows
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return MessageResult(self.rows)


@pytest.mark.asyncio
async def test_recent_context_excludes_current_message_limits_and_orders_ties_deterministically():
    current_message_id = uuid.uuid4()
    same_timestamp = datetime(2026, 7, 22, tzinfo=timezone.utc)
    newest_first = [
        HistoryMessage("assistant", f"message-{index}", uuid.UUID(int=index), same_timestamp)
        for index in range(13, 1, -1)
    ]
    session = ContextStatementSession(newest_first)

    context = await chat._load_recent_context(session, str(uuid.uuid4()), current_message_id)

    sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "message.id !=" in sql
    assert "ORDER BY message.created_at DESC, message.id DESC" in sql
    assert "LIMIT" in sql
    assert len(context) == 12
    assert [item["content"] for item in context] == [f"message-{index}" for index in range(2, 14)]
    assert all(item["content"] != "current message" for item in context)
