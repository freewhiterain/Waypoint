import json
import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app.services.destination_recommender import DestinationRecommendation
from app.api.dependencies import get_current_user
from app.governance.tool_invocations import CompletionOutcome, ToolInvocationRecord
from app.main import app
from app.schemas.planning import TravelRequirement
from app.schemas.tools import ToolResultRequest


class InMemoryInvocationRepository:
    def __init__(self, records):
        self.records = {record.call_id: record.model_copy(deep=True) for record in records}
        self.lock = asyncio.Lock()

    async def get_for_user(self, call_id, user_id):
        record = self.records.get(call_id)
        if record is None or record.user_id != user_id:
            return None
        return record.model_copy(deep=True)

    async def update_partial(self, call_id, user_id, partial_values):
        record = self.records.get(call_id)
        if record is None or record.user_id != user_id:
            return None
        record.partial_values = {**record.partial_values, **partial_values}
        record.version += 1
        return record.model_copy(deep=True)

    async def complete_once(self, call_id, user_id, result):
        record = self.records.get(call_id)
        if record is None or record.user_id != user_id:
            return None
        if record.status == "completed":
            return CompletionOutcome(record=record.model_copy(deep=True), completed_now=False)
        record.status = "completed"
        record.result = result
        record.version += 1
        return CompletionOutcome(record=record.model_copy(deep=True), completed_now=True)

    async def claim_processing(self, call_id, user_id, lease_timeout):
        async with self.lock:
            record = self.records.get(call_id)
            if record is None or record.user_id != user_id:
                return None
            if record.status != "pending":
                from app.governance.tool_invocations import ProcessingOutcome

                return ProcessingOutcome(
                    record=record.model_copy(deep=True),
                    claimed=False,
                    claim_version=record.version,
                )
            record.status = "processing"
            record.version += 1
            record.updated_at = datetime.now(timezone.utc)
            from app.governance.tool_invocations import ProcessingOutcome

            return ProcessingOutcome(
                record=record.model_copy(deep=True),
                claimed=True,
                claim_version=record.version,
            )

    async def finish_processing(self, call_id, user_id, expected_version, durable_result, session=None):
        async with self.lock:
            record = self.records.get(call_id)
            if (
                record is None
                or record.user_id != user_id
                or record.status != "processing"
                or record.version != expected_version
            ):
                return None
            record.status = "completed"
            record.result = durable_result
            record.version += 1
            record.updated_at = datetime.now(timezone.utc)
            return record.model_copy(deep=True)

    async def release_processing(self, call_id, user_id, expected_version):
        async with self.lock:
            record = self.records.get(call_id)
            if (
                record is None
                or record.user_id != user_id
                or record.status != "processing"
                or record.version != expected_version
            ):
                return None
            record.status = "pending"
            record.version += 1
            record.updated_at = datetime.now(timezone.utc)
            return record.model_copy(deep=True)

    async def renew_processing(self, call_id, user_id, expected_version):
        async with self.lock:
            record = self.records.get(call_id)
            if (
                record is None
                or record.user_id != user_id
                or record.status != "processing"
                or record.version != expected_version
            ):
                return False
            record.updated_at = datetime.now(timezone.utc)
            return True


class RecordingSession:
    def __init__(self):
        self.messages = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def add(self, message):
        self.messages.append(message)

    async def commit(self):
        return None

    async def refresh(self, _message):
        return None

    def begin(self):
        return self


class RecordingEventRepository:
    def __init__(self):
        self.events = []

    async def append(self, event):
        event.sequence = len(self.events) + 1
        self.events.append(event)
        return event


class FakeDraft:
    def __init__(self, destination):
        self.destination = destination

    def model_dump(self, *, mode):
        assert mode == "json"
        return {"destination": self.destination, "summary": "confirmed itinerary"}


def invocation(*, user_id, status="pending", tool="collect_trip_requirements"):
    return ToolInvocationRecord(
        call_id="call-1",
        user_id=user_id,
        conversation_id=str(uuid4()),
        tool=tool,
        status=status,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def parse_sse(response):
    return [
        json.loads(frame.removeprefix("data: ").strip())
        for frame in response.text.split("\n\n")
        if frame.strip()
    ]


@asynccontextmanager
async def endpoint_client(user):
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def configure_endpoint(monkeypatch, repository):
    from app.services import trip_form as tools

    session = RecordingSession()
    events = RecordingEventRepository()
    monkeypatch.setattr(tools, "PostgresToolInvocationRepository", lambda: repository, raising=False)
    monkeypatch.setattr(tools, "PostgresEventRepository", lambda: events, raising=False)
    monkeypatch.setattr(tools, "async_session_maker", lambda: session, raising=False)

    async def checkpointer():
        return None

    monkeypatch.setattr(tools, "get_checkpointer", checkpointer, raising=False)
    return session, events


def test_trip_form_result_route_is_registered():
    paths = {route.path for route in app.routes}

    assert "/api/v1/chat/tools/{call_id}/result" in paths


@pytest.mark.asyncio
async def test_tool_result_rejects_another_users_call(monkeypatch):
    owner_id = str(uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=owner_id)])
    configure_endpoint(monkeypatch, repository)

    async with endpoint_client(SimpleNamespace(id=uuid4())) as client:
        response = await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={"tool": "collect_trip_requirements", "status": "completed", "result": {
                "destination": "Kyoto", "departure_date": "2026-08-03", "days": 4,
            }},
        )

    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        None,
        {"destination": "Kyoto", "days": 4},
        {"destination": "Kyoto", "departure_date": "invalid", "days": 4},
        {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 0},
        {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4, "extra": True},
    ],
)
async def test_completed_result_requires_valid_trip_fields(monkeypatch, result):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    configure_endpoint(monkeypatch, repository)

    async with endpoint_client(user) as client:
        response = await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={"tool": "collect_trip_requirements", "status": "completed", "result": result},
        )

    assert response.status_code == 422
    assert repository.records["call-1"].status == "pending"


@pytest.mark.asyncio
async def test_tool_result_rejects_wrong_stored_tool(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository(
        [invocation(user_id=str(user.id), tool="other_tool")]
    )
    configure_endpoint(monkeypatch, repository)

    async with endpoint_client(user) as client:
        response = await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={"tool": "collect_trip_requirements", "status": "recommend_destination"},
        )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_recommendation_requires_a_pending_call(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id), status="cancelled")])
    configure_endpoint(monkeypatch, repository)

    async with endpoint_client(user) as client:
        response = await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={"tool": "collect_trip_requirements", "status": "recommend_destination"},
        )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_recommendation_keeps_call_pending_and_returns_rag_answer(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    session, _ = configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    queries = []

    async def recommend(partial_values, _context=None):
        queries.append(partial_values)
        return DestinationRecommendation(
            answer="Kyoto works well for a four-day food-focused trip.", cities=["Kyoto"]
        )

    monkeypatch.setattr(tools, "recommend_destinations", recommend, raising=False)

    async with endpoint_client(user) as client:
        response = await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={
                "tool": "collect_trip_requirements",
                "status": "recommend_destination",
                "partial_values": {"days": 4, "interests": "food"},
            },
        )

    events = parse_sse(response)
    assert response.status_code == 200
    assert [event["type"] for event in events] == ["token", "tool_result", "done"]
    assert events[0]["content"] == "Kyoto works well for a four-day food-focused trip."
    assert events[1]["status"] == "awaiting_destination"
    assert events[1]["payload"]["suggested_destinations"] == ["Kyoto"]
    assert repository.records["call-1"].status == "pending"
    assert repository.records["call-1"].partial_values == {"days": 4, "interests": "food"}
    assert queries == [{"days": 4, "interests": "food"}]
    assert session.messages[-1].extra_info["tool_result"]["status"] == "awaiting_destination"


@pytest.mark.asyncio
async def test_completed_result_invokes_supervisor_once_with_confirmed_fields(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    session, events = configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    calls = []

    async def supervisor(requirement, **kwargs):
        calls.append((requirement, kwargs))
        await kwargs["event_service"].emit(
            task_id=kwargs["task_id"],
            user_id=kwargs["user_id"],
            conversation_id=kwargs["conversation_id"],
            event_type="task_completed",
        )
        return FakeDraft(requirement.destination)

    monkeypatch.setattr(tools, "run_travel_planning", supervisor, raising=False)
    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4},
    }

    async with endpoint_client(user) as client:
        response = await client.post("/api/v1/chat/tools/call-1/result", json=payload)
        duplicate_response = await client.post(
            "/api/v1/chat/tools/call-1/result", json=payload
        )

    stream = parse_sse(response)
    duplicate_stream = parse_sse(duplicate_response)
    assert [event["type"] for event in stream] == ["result", "token", "done"]
    assert len(calls) == 1
    requirement, kwargs = calls[0]
    assert isinstance(requirement, TravelRequirement)
    assert requirement.destination == "Kyoto"
    assert requirement.departure_date == date(2026, 8, 3)
    assert requirement.days == 4
    assert kwargs["task_id"] == "call-1"
    durable = repository.records["call-1"].result
    assert durable["confirmed_result"] == payload["result"]
    assert durable["task_id"] == "call-1"
    assert [
        (event["type"], event.get("payload")) for event in duplicate_stream
    ] == [
        (event["type"], event.get("payload")) for event in stream
    ]
    assert not [
        task for task in asyncio.all_tasks() if task.get_name().startswith("tool-result-heartbeat:")
    ]
    assert events.events[-1].event_type == "task_completed"
    assert session.messages[-1].extra_info["tool_result"]["status"] == "completed"
    assert session.messages[-1].extra_info["assistant_result"] == stream[0]["payload"]["result"]


@pytest.mark.asyncio
async def test_duplicate_completed_result_uses_stored_completion_without_supervisor(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    confirmed = {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4}
    durable = {
        "confirmed_result": confirmed,
        "task_id": "call-1",
        "assistant_result": {"destination": "Kyoto", "summary": "confirmed itinerary"},
        "assistant_markdown": '{"destination": "Kyoto", "summary": "confirmed itinerary"}',
    }
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id), status="completed")])
    repository.records["call-1"].result = durable
    configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    async def unexpected_supervisor(*_args, **_kwargs):
        raise AssertionError("duplicate completion must not invoke Supervisor")

    monkeypatch.setattr(tools, "run_travel_planning", unexpected_supervisor, raising=False)

    async with endpoint_client(user) as client:
        response = await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={"tool": "collect_trip_requirements", "status": "completed", "result": confirmed},
        )

    stream = parse_sse(response)
    assert response.status_code == 200
    assert stream[0]["type"] == "result"
    assert stream[0]["payload"]["result"] == durable["assistant_result"]


@pytest.mark.asyncio
async def test_supervisor_failure_releases_claim_and_sanitizes_retryable_error(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    async def failing_supervisor(*_args, **_kwargs):
        raise RuntimeError("database password leaked")

    monkeypatch.setattr(tools, "run_travel_planning", failing_supervisor, raising=False)
    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4},
    }

    async with endpoint_client(user) as client:
        response = await client.post("/api/v1/chat/tools/call-1/result", json=payload)

    events = parse_sse(response)
    assert [event["type"] for event in events] == ["error", "done"]
    assert events[0]["payload"]["retryable"] is True
    assert "database password leaked" not in json.dumps(events[0])
    assert repository.records["call-1"].status == "pending"

    async def successful_supervisor(*_args, **_kwargs):
        return FakeDraft("Kyoto")

    monkeypatch.setattr(tools, "run_travel_planning", successful_supervisor, raising=False)

    async with endpoint_client(user) as client:
        retry = await client.post("/api/v1/chat/tools/call-1/result", json=payload)

    assert parse_sse(retry)[0]["type"] == "result"
    assert not [
        task for task in asyncio.all_tasks() if task.get_name().startswith("tool-result-heartbeat:")
    ]


@pytest.mark.asyncio
async def test_active_processing_duplicate_returns_nonterminal_tool_result(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    active = invocation(user_id=str(user.id), status="processing")
    active.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    repository = InMemoryInvocationRepository([active])
    configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    async def unexpected_supervisor(*_args, **_kwargs):
        raise AssertionError("active processing duplicate must not invoke Supervisor")

    monkeypatch.setattr(tools, "run_travel_planning", unexpected_supervisor, raising=False)
    async with endpoint_client(user) as client:
        response = await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={
                "tool": "collect_trip_requirements",
                "status": "completed",
                "result": {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4},
            },
        )

    events = parse_sse(response)
    assert [event["type"] for event in events] == ["tool_result", "done"]
    assert events[0]["status"] == "processing"
    assert events[0]["payload"]["terminal"] is False


@pytest.mark.asyncio
async def test_heartbeat_keeps_long_supervisor_claim_owned(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    monkeypatch.setattr(tools, "PROCESSING_LEASE_TIMEOUT", timedelta(milliseconds=45))
    supervisor_started = asyncio.Event()
    supervisor_calls = []

    async def slow_supervisor(requirement, **_kwargs):
        supervisor_calls.append(requirement.destination)
        supervisor_started.set()
        await asyncio.sleep(0.12)
        return FakeDraft(requirement.destination)

    monkeypatch.setattr(tools, "run_travel_planning", slow_supervisor, raising=False)
    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4},
    }

    async with endpoint_client(user) as client:
        first_request = asyncio.create_task(
            client.post("/api/v1/chat/tools/call-1/result", json=payload)
        )
        await asyncio.wait_for(supervisor_started.wait(), timeout=1)
        await asyncio.sleep(0.07)
        duplicate = await client.post("/api/v1/chat/tools/call-1/result", json=payload)
        first = await first_request

    duplicate_events = parse_sse(duplicate)
    assert [event["type"] for event in duplicate_events] == ["tool_result", "done"]
    assert duplicate_events[0]["status"] == "processing"
    assert supervisor_calls == ["Kyoto"]


@pytest.mark.asyncio
async def test_lease_loss_cancels_blocking_supervisor_before_second_claim_side_effect(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    monkeypatch.setattr(tools, "PROCESSING_LEASE_TIMEOUT", timedelta(milliseconds=30))
    started = asyncio.Event()
    block = asyncio.Event()
    side_effects = []

    async def lease_lost(*_args, **_kwargs):
        return False

    monkeypatch.setattr(repository, "renew_processing", lease_lost)

    async def supervisor(requirement, **_kwargs):
        side_effects.append("first-start" if len(side_effects) == 0 else "second-start")
        if len(side_effects) == 1:
            started.set()
            try:
                await block.wait()
            except asyncio.CancelledError:
                side_effects.append("first-cancel")
                raise
        return FakeDraft(requirement.destination)

    monkeypatch.setattr(tools, "run_travel_planning", supervisor, raising=False)
    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4},
    }

    async with endpoint_client(user) as client:
        first_request = asyncio.create_task(
            client.post("/api/v1/chat/tools/call-1/result", json=payload)
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0.05)
        second_request = asyncio.create_task(
            client.post("/api/v1/chat/tools/call-1/result", json=payload)
        )
        first = await asyncio.wait_for(first_request, timeout=1)
        second = await asyncio.wait_for(second_request, timeout=1)

    assert parse_sse(first)[0]["payload"]["code"] == "processing_conflict"
    assert parse_sse(second)[0]["type"] == "result"
    assert side_effects == ["first-start", "first-cancel", "second-start"]
    assert not [
        task for task in asyncio.all_tasks() if task.get_name().startswith("tool-result-heartbeat:")
    ]


@pytest.mark.asyncio
async def test_heartbeat_lease_loss_prevents_finish_and_returns_retryable_conflict(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    monkeypatch.setattr(tools, "PROCESSING_LEASE_TIMEOUT", timedelta(milliseconds=30))

    async def lease_lost(*_args, **_kwargs):
        return False

    async def slow_supervisor(*_args, **_kwargs):
        await asyncio.sleep(0.04)
        return FakeDraft("Kyoto")

    monkeypatch.setattr(repository, "renew_processing", lease_lost)
    monkeypatch.setattr(tools, "run_travel_planning", slow_supervisor, raising=False)
    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4},
    }

    async with endpoint_client(user) as client:
        response = await client.post("/api/v1/chat/tools/call-1/result", json=payload)

    event = parse_sse(response)[0]
    assert event["type"] == "error"
    assert event["payload"]["code"] == "processing_conflict"
    assert event["payload"]["retryable"] is True
    assert repository.records["call-1"].status == "pending"
    assert not [
        task for task in asyncio.all_tasks() if task.get_name().startswith("tool-result-heartbeat:")
    ]


@pytest.mark.asyncio
async def test_heartbeat_exception_cancels_supervisor_and_releases_original_claim(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    monkeypatch.setattr(tools, "PROCESSING_LEASE_TIMEOUT", timedelta(milliseconds=30))
    release_calls = []
    original_release = repository.release_processing

    async def release(call_id, user_id, expected_version):
        release_calls.append((call_id, user_id, expected_version))
        return await original_release(call_id, user_id, expected_version)

    async def renewal_failure(*_args, **_kwargs):
        raise RuntimeError("renewal backend unavailable")

    supervisor_started = asyncio.Event()

    async def blocking_supervisor(*_args, **_kwargs):
        supervisor_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(repository, "release_processing", release)
    monkeypatch.setattr(repository, "renew_processing", renewal_failure)
    monkeypatch.setattr(tools, "run_travel_planning", blocking_supervisor, raising=False)
    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4},
    }

    async with endpoint_client(user) as client:
        await asyncio.wait_for(
            client.post("/api/v1/chat/tools/call-1/result", json=payload), timeout=1
        )

    assert release_calls == [("call-1", str(user.id), 2)]
    assert repository.records["call-1"].status == "pending"


@pytest.mark.asyncio
async def test_client_cancellation_releases_claim_and_stops_heartbeat(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    session, _ = configure_endpoint(monkeypatch, repository)
    from app.services import trip_form as tools

    started = asyncio.Event()

    async def never_finishes(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(tools, "run_travel_planning", never_finishes, raising=False)
    data = ToolResultRequest.model_validate(
        {
            "tool": "collect_trip_requirements",
            "status": "completed",
            "result": {"destination": "Kyoto", "departure_date": "2026-08-03", "days": 4},
        }
    )
    record = repository.records["call-1"].model_copy(deep=True)

    async def consume():
        async for _frame in tools.tool_result_stream("call-1", data, str(user.id), record):
            pass

    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert repository.records["call-1"].status == "pending"
    assert session.messages == []
    assert not [
        task for task in asyncio.all_tasks() if task.get_name().startswith("tool-result-heartbeat:")
    ]
