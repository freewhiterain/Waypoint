"""线路 1 的 ReAct 循环（模型和工具都是假的，不访问网络）。"""

import json
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.api.v1 import chat
from app.schemas.slots import SlotUpdate, TripSlots
from app.schemas.tools import MainAgentDecision
from app.services import react_agent
from app.services.react_agent import MAX_TOOL_ROUNDS, ToolOutcome, run_react_agent
from app.services.slots import InMemorySlotRepository

from tests.test_chat_main_agent_flow import RecordingSession, SessionAwareRecordingRepository, session_factory


class ScriptedLLM:
    """按顺序返回预设回复；记录每次收到的消息和是否绑定了工具。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = [tool.name for tool in tools]
        return _Bound(self, with_tools=True)

    async def ainvoke(self, messages):
        return await self._reply(messages, with_tools=False)

    async def _reply(self, messages, *, with_tools):
        self.calls.append({"messages": list(messages), "with_tools": with_tools})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _Bound:
    def __init__(self, llm, *, with_tools):
        self.llm, self.with_tools = llm, with_tools

    async def ainvoke(self, messages):
        return await self.llm._reply(messages, with_tools=self.with_tools)


def call(name, call_id="c1", **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class _QueryArgs(BaseModel):
    query: str


def tool(name, handler):
    return StructuredTool.from_function(coroutine=handler, name=name, description=name, args_schema=_QueryArgs)


async def found(query: str) -> ToolOutcome:
    return ToolOutcome(text=f"查到：{query} 的资料", evidence_count=2)


async def nothing(query: str) -> ToolOutcome:
    return ToolOutcome(text="本地资料里没有查到相关内容。", status="empty")


async def broken(query: str) -> ToolOutcome:
    raise TimeoutError("tool down")


@pytest.mark.asyncio
async def test_answers_directly_when_no_tool_is_needed():
    llm = ScriptedLLM([AIMessage(content="成都最适合春秋去。")])

    result = await run_react_agent("成都什么时候去好", [], TripSlots(), llm=llm, tools=[tool("search", found)])

    assert result.answer == "成都最适合春秋去。"
    assert result.steps == []
    assert llm.bound_tools == ["search"]


@pytest.mark.asyncio
async def test_calls_a_tool_then_answers_from_its_result():
    llm = ScriptedLLM([call("search", query="成都 景点"), AIMessage(content="成都有大熊猫基地。")])
    seen_calls, seen_results = [], []

    async def on_call(payload):
        seen_calls.append(payload)

    async def on_result(payload):
        seen_results.append(payload)

    result = await run_react_agent(
        "成都有什么好玩的",
        [{"role": "user", "content": "想出去玩"}, {"role": "assistant", "content": "想去哪？"}],
        TripSlots(destination="成都", adults=2),
        llm=llm,
        tools=[tool("search", found)],
        on_tool_call=on_call,
        on_tool_result=on_result,
    )

    assert result.answer == "成都有大熊猫基地。"
    assert [(step.tool, step.arguments, step.outcome.status) for step in result.steps] == [
        ("search", {"query": "成都 景点"}, "completed")
    ]
    assert seen_calls == [{"call_id": "c1", "tool": "search", "arguments": {"query": "成都 景点"}}]
    assert seen_results[0]["result"]["evidence_count"] == 2
    # 第二次调用模型时，工具结果已经作为 ToolMessage 放进了上下文。
    second = llm.calls[1]["messages"]
    assert isinstance(second[-1], ToolMessage) and "成都 景点 的资料" in second[-1].content
    # 系统提示里带着当前槽位，历史对话也带上了。
    first = llm.calls[0]["messages"]
    assert '"destination":"成都"' in first[0].content and '"adults":2' in first[0].content
    assert [message.content for message in first[1:]] == ["想出去玩", "想去哪？", "成都有什么好玩的"]


@pytest.mark.asyncio
async def test_can_chain_several_tool_rounds():
    llm = ScriptedLLM(
        [
            call("search", "c1", query="成都"),
            call("lookup", "c2", query="西安"),
            AIMessage(content="两个城市都不错。"),
        ]
    )

    result = await run_react_agent(
        "成都和西安哪个好", [], TripSlots(), llm=llm, tools=[tool("search", found), tool("lookup", found)]
    )

    assert [step.tool for step in result.steps] == ["search", "lookup"]
    assert result.answer == "两个城市都不错。"


@pytest.mark.asyncio
async def test_empty_and_failing_tools_are_reported_to_the_model_not_raised():
    llm = ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[
                {"name": "empty", "args": {"query": "漠河"}, "id": "c1"},
                {"name": "broken", "args": {"query": "漠河"}, "id": "c2"},
                {"name": "missing", "args": {"query": "漠河"}, "id": "c3"},
            ]),
            AIMessage(content="没有查到漠河的资料。"),
        ]
    )

    result = await run_react_agent(
        "漠河怎么样", [], TripSlots(), llm=llm, tools=[tool("empty", nothing), tool("broken", broken)]
    )

    assert [step.outcome.status for step in result.steps] == ["empty", "failed", "failed"]
    assert result.steps[2].outcome.warnings == ["react_tool_error:unknown_tool"]
    assert result.answer == "没有查到漠河的资料。"


@pytest.mark.asyncio
async def test_stops_after_the_round_limit_and_forces_a_final_answer():
    replies = [call("search", f"c{index}", query="成都") for index in range(MAX_TOOL_ROUNDS)]
    llm = ScriptedLLM([*replies, AIMessage(content="先说这些。")])

    result = await run_react_agent("成都", [], TripSlots(), llm=llm, tools=[tool("search", found)])

    assert len(result.steps) == MAX_TOOL_ROUNDS
    assert result.answer == "先说这些。"
    assert llm.calls[-1]["with_tools"] is False  # 最后一次不再给工具


@pytest.mark.asyncio
async def test_real_tools_are_registered_with_argument_schemas():
    tools = {item.name: item for item in react_agent.build_react_tools()}

    assert set(tools) == {"search_travel_knowledge", "research_topic", "estimate_travel_time"}
    assert set(tools["research_topic"].args_schema.model_fields) == {"destination", "topic", "question", "deep"}


@pytest.mark.asyncio
async def test_travel_time_tool_does_not_invent_minutes_without_route_data(monkeypatch):
    from app.agents import travel_time

    monkeypatch.setattr(
        travel_time,
        "default_travel_time_service",
        lambda: travel_time.TravelTimeService([travel_time.DistrictEstimateProvider()]),
    )

    outcome = await react_agent._estimate_travel_time("成都", ["熊猫基地", "宽窄巷子", "熊猫基地"])

    assert outcome.status == "empty" and outcome.evidence_count == 0
    assert "需实时确认" in outcome.text and "分钟" not in outcome.text
    assert outcome.warnings == ["travel_time_estimated:no_route_data"]


# ---------------------------------------------------------------- 对话层


def _configure_chat(monkeypatch, decision, llm):
    class Agent:
        async def decide(self, message, context, slots=None):
            return decision

    session = RecordingSession([])
    monkeypatch.setattr(chat, "async_session_maker", session_factory(session))
    monkeypatch.setattr(chat, "MainAgentService", lambda: Agent())
    monkeypatch.setattr(chat, "PostgresToolInvocationRepository", lambda: SessionAwareRecordingRepository())
    monkeypatch.setattr(chat, "_react_llm", lambda: llm)
    monkeypatch.setattr(chat, "build_react_tools", lambda registry=None: [tool("search_travel_knowledge", found)])
    return session


async def _stream(message, repository=None):
    return [
        json.loads(frame.removeprefix("data: ").strip())
        async for frame in chat.generate_sse_stream(
            str(uuid.uuid4()), message, str(uuid.uuid4()), slot_repository=repository or InMemorySlotRepository()
        )
    ]


@pytest.mark.asyncio
async def test_route_one_streams_each_tool_step_then_the_answer(monkeypatch):
    llm = ScriptedLLM([call("search_travel_knowledge", query="成都 美食"), AIMessage(content="成都的火锅很有名。")])
    session = _configure_chat(
        monkeypatch, MainAgentDecision(action="answer_open_question", reason="q"), llm
    )

    events = await _stream("成都有什么好吃的")

    assert [event["type"] for event in events] == ["tool_call", "tool_result", "token", "done"]
    assert events[0]["tool"] == "search_travel_knowledge"
    assert events[1]["status"] == "completed"
    assert events[2]["content"] == "成都的火锅很有名。"
    saved = session.saved_messages[-1]
    assert saved.content == "成都的火锅很有名。"
    assert saved.extra_info["react_steps"] == [
        {"tool": "search_travel_knowledge", "arguments": {"query": "成都 美食"}, "status": "completed"}
    ]


@pytest.mark.asyncio
async def test_recommendation_answer_still_records_candidate_cities(monkeypatch):
    llm = ScriptedLLM([AIMessage(content="推荐你去成都。")])
    _configure_chat(
        monkeypatch,
        MainAgentDecision(action="recommend_destination", reason="r", slot_update=SlotUpdate()),
        llm,
    )
    repository = InMemorySlotRepository()
    conversation_id = str(uuid.uuid4())

    frames = [
        frame
        async for frame in chat.generate_sse_stream(
            conversation_id, "推荐个地方", str(uuid.uuid4()), slot_repository=repository
        )
    ]

    assert len(frames) == 2  # token + done
    assert (await repository.get(conversation_id)).suggested_destinations == ["成都"]


@pytest.mark.asyncio
async def test_react_failure_falls_back_to_the_fixed_answer_path(monkeypatch):
    llm = ScriptedLLM([TimeoutError("model down")])
    _configure_chat(monkeypatch, MainAgentDecision(action="answer_open_question", reason="q"), llm)

    async def fixed_answer(_message):
        return "为你找到以下相关资料。"

    monkeypatch.setattr(chat, "answer_open_question", fixed_answer)

    events = await _stream("成都有什么好玩的")

    assert [event["type"] for event in events] == ["token", "done"]
    assert events[0]["content"] == "为你找到以下相关资料。"


@pytest.mark.asyncio
async def test_planning_intent_never_enters_the_react_loop(monkeypatch):
    llm = ScriptedLLM([])  # 被调用就会因为没有预设回复而报错
    _configure_chat(
        monkeypatch,
        MainAgentDecision(
            action="collect_trip_requirements", reason="p", slot_update=SlotUpdate(destination="成都")
        ),
        llm,
    )

    events = await _stream("帮我规划成都旅行")

    assert [event["type"] for event in events] == ["tool_call", "done"]
    assert events[0]["tool"] == "collect_trip_requirements"
    assert llm.calls == []
