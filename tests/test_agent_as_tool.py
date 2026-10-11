import json
from datetime import date

import pytest

from app.agents.agent_tools import AgentToolRegistry
from app.api.v1 import chat
from app.schemas.research import SubagentResponse
from app.schemas.tools import AgentToolArguments, AgentToolCall, AgentToolResult, MainAgentDecision
from app.services.main_agent import MainAgentService


class RecordingRegistry:
    def __init__(self):
        self.calls = []

    async def run(self, task, requirement, *, event_callback=None):
        self.calls.append((task, requirement, event_callback))
        return SubagentResponse(
            task_id=task.id,
            worker=task.task_type,
            status="completed",
            summary="Panda Base is a suitable attraction.",
        )


@pytest.mark.asyncio
async def test_agent_as_tool_dispatches_to_the_selected_domain_subagent():
    registry = RecordingRegistry()
    tools = AgentToolRegistry(registry=registry)
    call = AgentToolCall(
        name="research_attractions",
        arguments=AgentToolArguments(
            destination="Chengdu",
            query="Research Chengdu attractions",
            departure_date=date(2026, 8, 1),
        ),
    )

    result = await tools.invoke(call)

    assert result.tool_name == "research_attractions"
    assert result.worker == "attractions"
    assert result.status == "completed"
    assert registry.calls[0][0].task_type == "attractions"
    assert registry.calls[0][0].query == "Research Chengdu attractions"
    assert registry.calls[0][1].destination == "Chengdu"
    assert registry.calls[0][1].departure_date == date(2026, 8, 1)


@pytest.mark.asyncio
async def test_agent_as_tool_reuses_the_existing_subagent_registry_without_a_second_worker_set():
    from app.agents import agent_tools
    from app.agents.subagents.registry import SubagentRegistry, create_default_subagent_registry

    assert agent_tools.AgentToolRegistry().registry.__class__ is SubagentRegistry
    supervisor_workers = set(create_default_subagent_registry().workers)
    assert set(agent_tools.TOOL_WORKERS.values()) == supervisor_workers


@pytest.mark.asyncio
async def test_run_agent_tool_uses_the_configured_planning_registry(monkeypatch):
    from app.agents import agent_tools

    registry = RecordingRegistry()
    monkeypatch.setattr(agent_tools, "create_planning_registry", lambda: (registry, None))
    call = AgentToolCall(
        name="research_attractions",
        arguments=AgentToolArguments(destination="Chengdu", query="Research attractions"),
    )

    result = await agent_tools.run_agent_tool(call)

    assert result.status == "completed"
    assert len(registry.calls) == 1


@pytest.mark.asyncio
async def test_unknown_tool_name_is_rejected_with_a_stable_code_without_running_a_worker():
    registry = RecordingRegistry()
    tools = AgentToolRegistry(registry=registry)
    call = AgentToolCall.model_construct(
        name="research_unknown",
        arguments=AgentToolArguments(destination="Chengdu", query="anything"),
    )

    result = await tools.invoke(call)

    assert result.status == "failed"
    assert result.worker is None
    assert result.warnings == ["agent_tool_error:unknown_tool"]
    assert result.evidence_count == 0
    assert registry.calls == []


@pytest.mark.asyncio
async def test_subagent_failure_is_sanitized_into_a_stable_failed_result():
    class ExplodingRegistry:
        async def run(self, _task, _requirement, **_kwargs):
            raise RuntimeError("provider payload sk-secret-token exploded")

    tools = AgentToolRegistry(registry=ExplodingRegistry())
    call = AgentToolCall(
        name="research_hotel",
        arguments=AgentToolArguments(destination="成都", query="深入研究成都的住宿"),
    )

    result = await tools.invoke(call)

    assert result.tool_name == "research_hotel"
    assert result.status == "failed"
    assert result.warnings == ["agent_tool_error:execution_failed"]
    assert "sk-secret-token" not in result.answer
    assert "RuntimeError" not in result.answer


@pytest.mark.asyncio
async def test_missing_evidence_returns_an_unavailable_answer_instead_of_invented_facts():
    class EmptyRegistry:
        async def run(self, task, _requirement, **_kwargs):
            return SubagentResponse(
                task_id=task.id,
                worker=task.task_type,
                status="unavailable",
                summary="",
                warnings=["No provider returned evidence for weather."],
            )

    tools = AgentToolRegistry(registry=EmptyRegistry())
    call = AgentToolCall(
        name="research_weather",
        arguments=AgentToolArguments(destination="成都", query="帮我查一下成都最近的天气"),
    )

    result = await tools.invoke(call)

    assert result.status == "unavailable"
    assert result.evidence_count == 0
    assert result.answer.strip()
    assert result.warnings == ["No provider returned evidence for weather."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_tool"),
    [
        ("深入研究成都适合亲子游的景点", "research_attractions"),
        ("帮我查一下成都最近的天气", "research_weather"),
        ("帮我查一下成都最新的交通", "research_transport"),
        ("仔细研究一下三亚的住宿", "research_hotel"),
        ("深入研究一下杭州的美食", "research_food"),
    ],
)
async def test_main_agent_routes_explicit_deep_research_to_the_matching_agent_tool(message, expected_tool):
    decision = await MainAgentService(use_llm=False).decide(message, [])

    assert decision.action == "invoke_agent_tool"
    assert decision.tool_call is not None
    assert decision.tool_call.name == expected_tool
    assert decision.tool_call.arguments.query == message


@pytest.mark.asyncio
async def test_main_agent_keeps_the_destination_from_the_current_message():
    decision = await MainAgentService(use_llm=False).decide("请深入研究成都有哪些好玩的景点", [])

    assert decision.tool_call.arguments.destination == "成都"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    ["成都有什么好玩的", "最近成都有什么好玩的", "成都有哪些适合亲子游的地方"],
)
async def test_ordinary_questions_still_use_local_rag(message):
    decision = await MainAgentService(use_llm=False).decide(message, [])

    assert decision.action == "answer_open_question"
    assert decision.tool_call is None


@pytest.mark.asyncio
async def test_explicit_planning_still_wins_over_research_markers():
    decision = await MainAgentService(use_llm=False).decide("深入研究之后帮我规划成都三天行程", [])

    assert decision.action == "collect_trip_requirements"


@pytest.mark.asyncio
async def test_research_request_without_a_known_destination_does_not_guess_a_worker():
    decision = await MainAgentService(use_llm=False).decide("深入研究一下当地的天气", [])

    assert decision.action != "invoke_agent_tool"
    assert decision.tool_call is None


@pytest.mark.asyncio
async def test_routing_model_cannot_emit_an_agent_tool_action_without_a_tool_call(monkeypatch):
    class StructuredOutput:
        async def ainvoke(self, _messages):
            return {"action": "invoke_agent_tool", "reason": "模型没有给出工具参数"}

    class Llm:
        def with_structured_output(self, _schema):
            return StructuredOutput()

    monkeypatch.setattr("app.services.main_agent.settings.llm_api_key", "configured")
    monkeypatch.setattr("app.agents.llm.get_llm", lambda: Llm())

    decision = await MainAgentService(use_llm=True).decide("我想想看", [])

    assert decision.action == "direct_response"
    assert decision.tool_call is None


def test_agent_tool_call_is_optional_for_legacy_main_agent_decisions():
    decision = MainAgentDecision(action="direct_response", reason="small talk")

    assert decision.tool_call is None


class StreamResult:
    def scalars(self):
        return self

    def all(self):
        return []


class StreamSession:
    def __init__(self):
        self.messages = []

    def add(self, message):
        self.messages.append(message)

    async def commit(self):
        return None

    async def refresh(self, message):
        return None

    async def execute(self, _statement):
        return StreamResult()


class DecisionAgent:
    async def decide(self, _message, _context, _slots=None):
        return MainAgentDecision(
            action="invoke_agent_tool",
            reason="deep research",
            tool_call=AgentToolCall(
                name="research_attractions",
                arguments=AgentToolArguments(
                    destination="Chengdu",
                    query="Research Chengdu attractions",
                ),
            ),
        )


def configure_agent_tool_stream(monkeypatch, result, calls):
    session = StreamSession()

    class SessionFactory:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *_args):
            return False

    async def invoke(call):
        calls.append(call)
        return result

    monkeypatch.setattr(chat, "async_session_maker", lambda: SessionFactory())
    monkeypatch.setattr(chat, "MainAgentService", lambda: DecisionAgent())
    monkeypatch.setattr(chat, "run_agent_tool", invoke)
    return session


async def collect_stream_events():
    return [
        json.loads(frame.removeprefix("data: ").strip())
        async for frame in chat.generate_sse_stream("conversation", "message", "user")
    ]


@pytest.mark.asyncio
async def test_chat_executes_agent_tool_and_streams_tool_result(monkeypatch):
    result = AgentToolResult(
        tool_name="research_attractions",
        worker="attractions",
        status="completed",
        answer="Panda Base is a suitable attraction.",
        evidence_count=1,
    )
    calls = []
    session = configure_agent_tool_stream(monkeypatch, result, calls)

    events = await collect_stream_events()

    assert [event["type"] for event in events] == ["tool_call", "tool_result", "token", "done"]
    assert events[0]["tool"] == "research_attractions"
    assert events[0]["arguments"]["destination"] == "Chengdu"
    assert events[1]["status"] == "completed"
    assert events[1]["result"]["evidence_count"] == 1
    assert events[1]["call_id"] == events[0]["call_id"]
    assert events[2]["content"] == result.answer
    assert calls[0].name == "research_attractions"
    assert [message.role for message in session.messages] == ["user", "assistant"]
    assert session.messages[-1].extra_info["action"] == "invoke_agent_tool"
    assert session.messages[-1].extra_info["tool_result"]["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_tool_turn_never_persists_a_form_tool_invocation(monkeypatch):
    result = AgentToolResult(
        tool_name="research_attractions",
        worker="attractions",
        status="completed",
        answer="Panda Base is a suitable attraction.",
        evidence_count=1,
    )
    configure_agent_tool_stream(monkeypatch, result, [])
    monkeypatch.setattr(
        chat,
        "PostgresToolInvocationRepository",
        lambda: (_ for _ in ()).throw(AssertionError("agent tools must not use the form repository")),
    )

    async def unexpected_rag(_question):
        raise AssertionError("agent tools must not fall back to local RAG")

    monkeypatch.setattr(chat, "answer_open_question", unexpected_rag)

    events = await collect_stream_events()

    assert [event["type"] for event in events] == ["tool_call", "tool_result", "token", "done"]


@pytest.mark.asyncio
async def test_failed_agent_tool_still_streams_a_sanitized_turn(monkeypatch):
    result = AgentToolResult(
        tool_name="research_weather",
        worker=None,
        status="failed",
        answer="这次研究没有完成，请稍后再试。",
        warnings=["agent_tool_error:execution_failed"],
    )
    configure_agent_tool_stream(monkeypatch, result, [])

    events = await collect_stream_events()

    assert [event["type"] for event in events] == ["tool_call", "tool_result", "token", "done"]
    assert events[1]["status"] == "failed"
    assert events[1]["result"]["warnings"] == ["agent_tool_error:execution_failed"]
    assert events[2]["content"] == result.answer


@pytest.mark.asyncio
async def test_agent_tool_action_without_a_tool_call_falls_back_to_direct_response(monkeypatch):
    session = StreamSession()

    class SessionFactory:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *_args):
            return False

    class BrokenDecisionAgent:
        async def decide(self, _message, _context, _slots=None):
            return MainAgentDecision(action="invoke_agent_tool", reason="missing tool call")

    async def unexpected_tool(_call):
        raise AssertionError("chat must not invoke an agent tool without a tool call")

    monkeypatch.setattr(chat, "async_session_maker", lambda: SessionFactory())
    monkeypatch.setattr(chat, "MainAgentService", lambda: BrokenDecisionAgent())
    monkeypatch.setattr(chat, "run_agent_tool", unexpected_tool)

    events = await collect_stream_events()

    assert [event["type"] for event in events] == ["token", "done"]
