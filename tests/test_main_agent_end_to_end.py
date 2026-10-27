import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.destination_recommender import DestinationRecommendation
from app.api.v1 import chat as chat_api, tools as tools_api
from app.services import trip_form as tools
from app.services import chat_flow as chat
from app.api.v1.conversations import create_conversation
from app.governance.events import InMemoryEventRepository
from app.governance.tool_invocations import (
    InMemoryToolInvocationRepository,
    ToolInvocationRecord,
)
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.conversation import ConversationCreate
from app.schemas.tools import ToolResultRequest


PROACTIVE_OFFER = "需要我帮你规划一下旅行吗？"
STORE = Path(__file__).resolve().parents[1] / "web" / "src" / "store" / "chat.ts"


class Rows:
    def __init__(self, items):
        self.items = list(items)

    def scalar_one_or_none(self):
        return self.items[0] if self.items else None

    def scalars(self):
        return self

    def all(self):
        return list(self.items)


class MemorySession:
    def __init__(self, history=None):
        self.history = list(history or [])
        self.conversations = []
        self.messages = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def begin(self):
        return self

    def add(self, entity):
        if isinstance(entity, Conversation):
            self.conversations.append(entity)
        elif isinstance(entity, Message):
            self.messages.append(entity)
        else:
            raise AssertionError(f"Unexpected entity: {type(entity).__name__}")

    async def flush(self):
        now = datetime.now(timezone.utc)
        for conversation in self.conversations:
            conversation.id = conversation.id or uuid.uuid4()
            conversation.extra_info = conversation.extra_info or {}
            conversation.created_at = conversation.created_at or now
            conversation.updated_at = conversation.updated_at or now
        for message in self.messages:
            message.id = message.id or uuid.uuid4()
            message.extra_info = message.extra_info or {}
            message.created_at = message.created_at or now

    async def commit(self):
        await self.flush()

    async def refresh(self, _entity):
        await self.flush()

    async def execute(self, _statement):
        return Rows(self.history)


class HistorySession(MemorySession):
    def __init__(self, conversation, messages):
        super().__init__()
        self.query_results = [[conversation], messages]

    async def execute(self, _statement):
        return Rows(self.query_results.pop(0))


class SessionToolRepository(InMemoryToolInvocationRepository):
    async def create_in_session(self, _session, record):
        return await self.create(record)


class FakePlan:
    def __init__(self, destination):
        self.destination = destination

    def model_dump(self, *, mode):
        assert mode == "json"
        return {"destination": self.destination, "summary": "confirmed itinerary"}


def session_factory(session):
    @asynccontextmanager
    async def factory():
        yield session

    return factory


def parse_frame(frame):
    if isinstance(frame, bytes):
        frame = frame.decode("utf-8")
    return json.loads(frame.removeprefix("data: ").strip())


async def collect_stream(stream):
    return [parse_frame(frame) async for frame in stream]


async def collect_response(response):
    return [parse_frame(frame) async for frame in response.body_iterator]


def configure_chat(monkeypatch, session, repository):
    monkeypatch.setattr(chat, "async_session_maker", session_factory(session))
    monkeypatch.setattr(chat, "PostgresToolInvocationRepository", lambda: repository)


def configure_tools(monkeypatch, session, repository, supervisor):
    monkeypatch.setattr(tools, "async_session_maker", lambda: session)
    monkeypatch.setattr(tools, "PostgresToolInvocationRepository", lambda: repository)
    monkeypatch.setattr(tools, "PostgresEventRepository", InMemoryEventRepository)
    monkeypatch.setattr(tools, "run_travel_planning", supervisor)

    async def fake_checkpointer():
        return None

    monkeypatch.setattr(tools, "get_checkpointer", fake_checkpointer)


def make_message(conversation_id, role, content, extra_info=None):
    return Message(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        role=role,
        content=content,
        extra_info=extra_info or {},
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_new_conversation_returns_and_frontend_renders_proactive_offer():
    session = MemorySession()
    user = SimpleNamespace(id=uuid.uuid4())

    response = await create_conversation(ConversationCreate(title="新行程"), user, session)

    assert response.initial_message.content == PROACTIVE_OFFER
    assert session.messages[0].extra_info == {"kind": "conversation_offer"}


@pytest.mark.asyncio
async def test_affirmation_after_offer_emits_form_tool_call(monkeypatch):
    conversation_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    history = [
        make_message(uuid.UUID(conversation_id), "assistant", PROACTIVE_OFFER)
    ]
    session = MemorySession(history)
    repository = SessionToolRepository()
    configure_chat(monkeypatch, session, repository)

    events = await collect_stream(
        chat.generate_sse_stream(conversation_id, "好的", user_id)
    )

    assert [event["type"] for event in events] == ["tool_call", "done"]
    assert events[0]["tool"] == "collect_trip_requirements"
    assert events[0]["arguments"] == {"initial_values": {}}
    stored = await repository.get_for_user(events[0]["call_id"], user_id)
    assert stored is not None
    assert stored.status == "pending"


@pytest.mark.asyncio
async def test_valid_tool_result_runs_supervisor_exactly_once_and_returns_final(monkeypatch):
    user = SimpleNamespace(id=uuid.uuid4())
    conversation_id = str(uuid.uuid4())
    repository = SessionToolRepository()
    await repository.create(
        ToolInvocationRecord(
            call_id="call-final",
            user_id=str(user.id),
            conversation_id=conversation_id,
            tool="collect_trip_requirements",
        )
    )
    session = MemorySession()
    supervisor_calls = []

    async def supervisor(requirement, **kwargs):
        supervisor_calls.append((requirement, kwargs))
        return FakePlan(requirement.destination)

    configure_tools(monkeypatch, session, repository, supervisor)
    request = ToolResultRequest.model_validate(
        {
            "tool": "collect_trip_requirements",
            "status": "completed",
            "result": {
                "destination": "成都",
                "departure_date": "2026-08-03",
                "days": 4,
            },
        }
    )

    first = await tools_api.submit_tool_result("call-final", request, user)
    first_events = await collect_response(first)
    duplicate = await tools_api.submit_tool_result("call-final", request, user)
    duplicate_events = await collect_response(duplicate)

    assert [event["type"] for event in first_events] == ["result", "token", "done"]
    assert first_events[0]["payload"]["result"]["destination"] == "成都"
    assert "confirmed itinerary" in first_events[1]["content"]
    assert len(supervisor_calls) == 1
    assert [event["type"] for event in duplicate_events] == [
        event["type"] for event in first_events
    ]
    assert [event["payload"] for event in duplicate_events] == [
        event["payload"] for event in first_events
    ]
    assert session.messages[-1].extra_info["tool_result"]["status"] == "completed"


@pytest.mark.asyncio
async def test_direct_chengdu_planning_request_emits_prefilled_form(monkeypatch):
    conversation_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    session = MemorySession()
    repository = SessionToolRepository()
    configure_chat(monkeypatch, session, repository)

    events = await collect_stream(
        chat.generate_sse_stream(conversation_id, "帮我规划成都旅行", user_id)
    )

    assert [event["type"] for event in events] == ["tool_call", "done"]
    assert events[0]["arguments"] == {"initial_values": {"destination": "成都"}}


@pytest.mark.asyncio
async def test_chengdu_open_question_uses_rag_only(monkeypatch):
    conversation_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    session = MemorySession()
    repository = SessionToolRepository()
    configure_chat(monkeypatch, session, repository)
    rag_calls = []

    async def rag_answer(question):
        rag_calls.append(question)
        return "成都近期适合逛博物馆和城市街区。"

    monkeypatch.setattr(chat, "answer_open_question", rag_answer)
    monkeypatch.setattr(
        chat,
        "PostgresToolInvocationRepository",
        lambda: (_ for _ in ()).throw(AssertionError("RAG must not open the form")),
    )

    events = await collect_stream(
        chat.generate_sse_stream(conversation_id, "成都有什么好玩的", user_id)
    )

    assert [event["type"] for event in events] == ["token", "done"]
    assert events[0]["payload"]["action"] == "answer_open_question"
    assert rag_calls == ["成都有什么好玩的"]
    assert repository.records == {}


@pytest.mark.asyncio
async def test_recommendation_saves_partial_values_then_selected_city_resumes(monkeypatch):
    user = SimpleNamespace(id=uuid.uuid4())
    conversation_id = str(uuid.uuid4())
    repository = SessionToolRepository()
    await repository.create(
        ToolInvocationRecord(
            call_id="call-recommend",
            user_id=str(user.id),
            conversation_id=conversation_id,
            tool="collect_trip_requirements",
        )
    )
    session = MemorySession()
    supervisor_calls = []

    async def supervisor(requirement, **_kwargs):
        supervisor_calls.append(requirement)
        return FakePlan(requirement.destination)

    async def recommend(_partial_values, _context=None):
        return DestinationRecommendation(answer="可以考虑成都或厦门。", cities=["成都"])

    configure_tools(monkeypatch, session, repository, supervisor)
    monkeypatch.setattr(tools, "recommend_destinations", recommend)
    recommendation = ToolResultRequest.model_validate(
        {
            "tool": "collect_trip_requirements",
            "status": "recommend_destination",
            "partial_values": {"departure_date": "2026-08-03", "days": 4},
        }
    )

    recommendation_response = await tools_api.submit_tool_result(
        "call-recommend", recommendation, user
    )
    recommendation_events = await collect_response(recommendation_response)
    pending = await repository.get_for_user("call-recommend", str(user.id))

    assert [event["type"] for event in recommendation_events] == [
        "token",
        "tool_result",
        "done",
    ]
    assert pending.status == "pending"
    assert pending.partial_values == {"departure_date": "2026-08-03", "days": 4}
    assert session.messages[-1].extra_info["tool_result"]["status"] == "awaiting_destination"

    completion = ToolResultRequest.model_validate(
        {
            "tool": "collect_trip_requirements",
            "status": "completed",
            "result": {
                "destination": "成都",
                "departure_date": "2026-08-03",
                "days": 4,
            },
        }
    )
    completion_response = await tools_api.submit_tool_result(
        "call-recommend", completion, user
    )
    completion_events = await collect_response(completion_response)

    assert completion_events[0]["type"] == "result"
    assert [call.destination for call in supervisor_calls] == ["成都"]


@pytest.mark.asyncio
async def test_history_refresh_contains_and_restores_pending_tool_state():
    user = SimpleNamespace(id=uuid.uuid4())
    conversation_id = uuid.uuid4()
    call_id = "call-history"
    conversation = Conversation(
        id=conversation_id,
        user_id=user.id,
        title="成都行程",
        status="active",
        extra_info={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    messages = [
        make_message(conversation_id, "assistant", PROACTIVE_OFFER),
        make_message(
            conversation_id,
            "assistant",
            "正在收集旅行需求。",
            {
                "tool_call": {
                    "call_id": call_id,
                    "tool": "collect_trip_requirements",
                    "arguments": {"initial_values": {}},
                }
            },
        ),
        make_message(
            conversation_id,
            "assistant",
            "可以考虑成都或厦门。",
            {
                "tool_result": {
                    "tool": "collect_trip_requirements",
                    "status": "awaiting_destination",
                    "partial_values": {"days": 4},
                }
            },
        ),
    ]
    history = await chat_api.get_chat_history(
        str(conversation_id), user, HistorySession(conversation, messages)
    )

    assert history["messages"][1]["extra_info"]["tool_call"]["call_id"] == call_id
    assert history["messages"][2]["extra_info"]["tool_result"] == {
        "tool": "collect_trip_requirements",
        "status": "awaiting_destination",
        "partial_values": {"days": 4},
    }
    store = STORE.read_text(encoding="utf-8")
    assert 'toolCall?.tool === "collect_trip_requirements"' in store
    assert 'result.status === "awaiting_destination"' in store
