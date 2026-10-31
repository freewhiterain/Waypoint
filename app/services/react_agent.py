"""线路 1：ReAct 问答。

开放问题、目的地推荐、单项研究不再各走一条固定流程，而是交给同一个循环：
模型看问题 → 决定调哪个工具 → 看工具结果 → 继续查或者回答。最多 MAX_TOOL_ROUNDS 轮，
用完还没给出回答，就要求它基于已有结果直接作答。

工具只有读操作：
- search_travel_knowledge：本地旅行资料（文档 RAG）
- research_topic：某个城市某一方面的专项研究（子 Agent，会用到图谱、天气/搜索 MCP、Deep Search）
- estimate_travel_time：城市内两地之间的交通耗时（交通图谱 → 高德 → 按区粗估）

回答只能依据工具结果和用户自己说过的话，工具没查到就明说没查到。
"""

from __future__ import annotations

import json
from datetime import date
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.schemas.slots import TripSlots
from app.schemas.tools import AgentToolArguments, AgentToolCall
from app.utils.logger import app_logger

MAX_TOOL_ROUNDS = 4
_HISTORY_LIMIT = 6
_MAX_EVIDENCE_ITEMS = 6
_NO_ANSWER = "这个问题我暂时没有查到可靠的资料，换个问法或者告诉我具体城市，我再帮你查。"

ToolStatus = Literal["completed", "empty", "failed"]
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
# 流式输出回调：("delta", 文字) 追加一段回答；("reset", "") 撤回已经推出去的文字
# （模型先说了半句，随后又决定调工具）。
TokenCallback = Callable[[str, str], Awaitable[None]]


@dataclass
class ToolOutcome:
    """一次工具调用的结果：text 给模型看，其余用于前端展示和记录。"""

    text: str
    status: ToolStatus = "completed"
    evidence_count: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReactStep:
    tool: str
    arguments: dict[str, Any]
    outcome: ToolOutcome


@dataclass
class ReactResult:
    answer: str
    steps: list[ReactStep] = field(default_factory=list)


# ---------------------------------------------------------------- 工具


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(description="要查的内容，用简短的中文关键词")
    destination: str | None = Field(default=None, description="城市名；知道是哪个城市时填，查得更准")
    category: Literal["attractions", "food", "hotel", "transport", "weather"] | None = Field(
        default=None, description="资料类别；和 destination 一起填时只查该城市这一类资料"
    )


class ResearchTopicArgs(BaseModel):
    destination: str = Field(description="城市名")
    topic: Literal["attractions", "weather", "transport", "hotel", "food"] = Field(description="研究哪个方面")
    question: str = Field(description="具体想了解什么")
    deep: bool = Field(default=False, description="用户要求深入、仔细研究时为 true")


class TripCostArgs(BaseModel):
    destination: str = Field(description="城市名")
    days: int = Field(description="游玩天数", ge=1, le=30)
    people: int = Field(default=1, description="出行人数", ge=1, le=30)


class RecommendCitiesArgs(BaseModel):
    days: int | None = Field(default=None, description="打算玩几天，用户没说就不填", ge=1, le=30)
    departure_date: str | None = Field(default=None, description="出发日期 YYYY-MM-DD 或月份，用户没说就不填")
    interests: list[str] = Field(default_factory=list, description="兴趣，如 美食、亲子、海边、历史")
    children: int | None = Field(default=None, description="同行儿童人数")


class TravelTimeArgs(BaseModel):
    destination: str = Field(description="城市名")
    places: list[str] = Field(description="要比较的地点名称，至少两个", min_length=2, max_length=6)


def _mentions_city(item: Any, city: str) -> bool:
    metadata = getattr(item, "metadata", None) or {}
    tagged = {str(metadata.get(key) or "") for key in ("destination", "city")}
    return city in tagged or city in (item.content or "") or city in (item.source or "")


async def _search_travel_knowledge(query: str, destination: str | None = None, category: str | None = None) -> ToolOutcome:
    from app.rag.local_knowledge import get_local_knowledge_service

    knowledge = get_local_knowledge_service()
    if destination and category:
        evidence = knowledge.search_destination(destination, category, query)
    else:
        evidence = knowledge.search(" ".join(part for part in (destination, query) if part))
    if destination:
        # 本地资料只有少数城市：按语义找最像的，会把成都的住宿资料当成杭州的返回。只留提到这个城市的。
        evidence = [item for item in evidence if _mentions_city(item, destination)]
    if not evidence:
        return ToolOutcome(text="本地资料里没有查到相关内容。", status="empty")
    lines = [f"[{index + 1}] {item.content}（来源：{item.source}）" for index, item in enumerate(evidence[:_MAX_EVIDENCE_ITEMS])]
    return ToolOutcome(text="\n".join(lines), evidence_count=len(evidence))


def _make_research_topic(registry: Any | None):
    async def research_topic(destination: str, topic: str, question: str, deep: bool = False) -> ToolOutcome:
        from app.agents.agent_tools import run_agent_tool

        result = await run_agent_tool(
            AgentToolCall(
                name=f"research_{topic}",
                arguments=AgentToolArguments(
                    destination=destination,
                    query=question[:500],
                    research_mode="deep" if deep else "normal",
                ),
            ),
            registry=registry,
        )
        status: ToolStatus = "completed" if result.evidence_count else "empty"
        if result.status == "failed":
            status = "failed"
        return ToolOutcome(
            text=result.answer,
            status=status,
            evidence_count=result.evidence_count,
            warnings=list(result.warnings),
        )

    return research_topic


async def _estimate_travel_time(destination: str, places: list[str]) -> ToolOutcome:
    from app.agents.travel_time import Place, default_travel_time_service, describe_leg

    unique = list(dict.fromkeys(name.strip() for name in places if name.strip()))
    if len(unique) < 2:
        return ToolOutcome(text="至少需要两个不同的地点才能比较交通耗时。", status="empty")
    place_list = [Place(id=name, name=name, category="attractions") for name in unique]
    by_id = {place.id: place for place in place_list}
    legs = await default_travel_time_service().compute(destination, place_list)
    timed = [leg for leg in legs if leg.minutes is not None]
    lines = [describe_leg(leg, by_id[leg.from_id], by_id[leg.to_id]) for leg in legs]
    return ToolOutcome(
        text="\n".join(lines),
        status="completed" if timed else "empty",
        evidence_count=len(timed),
        warnings=[] if len(timed) == len(legs) else ["travel_time_estimated:no_route_data"],
    )


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]


async def _estimate_trip_cost(destination: str, days: int, people: int = 1) -> ToolOutcome:
    """按地点库的门票、人均和房价做区间估算，不经过模型、不编数字。

    门票：最热门的景点里每天两个；餐饮：日常价位餐厅每天两顿；住宿：热门酒店按晚、两人一间。
    区间取 25~75 分位。
    """
    from app.places.catalog import get_place_catalog, _price_cap, _NOT_A_MEAL

    catalog = get_place_catalog()
    attractions = [r.price for r in catalog.places(destination, "attractions")[:20] if r.price is not None]
    restaurants = catalog.places(destination, "food")
    cap = _price_cap(restaurants)
    meals = [
        r.price for r in restaurants if r.price and r.type not in _NOT_A_MEAL and (cap is None or r.price <= cap)
    ]
    hotels = [r.price for r in catalog.places(destination, "hotel")[:20] if r.price]
    if not (attractions and meals and hotels):
        return ToolOutcome(text=f"地点库没有覆盖{destination}的价格数据，无法估算费用。", status="empty")

    nights, rooms = max(days - 1, 0), (people + 1) // 2

    def band(values: list[float], times: float) -> tuple[int, int]:
        return round(_percentile(values, 0.25) * times), round(_percentile(values, 0.75) * times)

    ticket = band(attractions, 2 * days * people)
    meal = band(meals, 2 * days * people)
    hotel = band(hotels, nights * rooms)
    total = (ticket[0] + meal[0] + hotel[0], ticket[1] + meal[1] + hotel[1])
    lines = [
        f"{destination}{days}天、{people}人的费用估算（地点库价格快照，区间为 25~75 分位）：",
        f"- 门票：每天 2 个热门景点，约 {ticket[0]}~{ticket[1]} 元",
        f"- 餐饮：每天 2 顿日常价位正餐，约 {meal[0]}~{meal[1]} 元",
        f"- 住宿：{nights} 晚、{rooms} 间热门酒店，约 {hotel[0]}~{hotel[1]} 元",
        f"- 合计约 {total[0]}~{total[1]} 元，不含城际交通、市内打车和购物。",
    ]
    return ToolOutcome(text="\n".join(lines), evidence_count=len(attractions) + len(meals) + len(hotels))


async def _recommend_cities(
    days: int | None = None,
    departure_date: str | None = None,
    interests: list[str] | None = None,
    children: int | None = None,
) -> ToolOutcome:
    """在地点库覆盖的城市里按条件挑 2~3 个；和规划表单里的"帮我推荐"是同一套逻辑。"""
    from app.services.destination_recommender import recommend_destinations

    partial = {key: value for key, value in {"days": days, "departure_date": departure_date}.items() if value}
    context = {key: value for key, value in {"styles": interests or [], "children": children}.items() if value}
    result = await recommend_destinations(partial, context)
    return ToolOutcome(
        text=result.answer,
        status="completed" if result.cities else "empty",
        evidence_count=len(result.cities),
    )


def build_react_tools(registry: Any | None = None) -> list[StructuredTool]:
    """线路 1 可用的只读工具。registry 只在测试里替换子 Agent 注册表时传。"""
    return [
        StructuredTool.from_function(
            coroutine=_search_travel_knowledge,
            name="search_travel_knowledge",
            description="查本地旅行资料：城市概况、景点、美食、住宿片区、市内交通建议。先用它了解一个城市或比较几个城市。",
            args_schema=SearchKnowledgeArgs,
        ),
        StructuredTool.from_function(
            coroutine=_make_research_topic(registry),
            name="research_topic",
            description="对某个城市的某一方面做专项研究（景点/天气/交通/住宿/美食）。需要实时信息（如天气）或更详细的资料时用它。",
            args_schema=ResearchTopicArgs,
        ),
        StructuredTool.from_function(
            coroutine=_recommend_cities,
            name="recommend_cities",
            description="用户还没定去哪、想要推荐几个城市时用它：在有完整地点数据的城市里按天数、季节、同行人和兴趣挑 2~3 个，"
            "并附上每个城市的热门景点。",
            args_schema=RecommendCitiesArgs,
        ),
        StructuredTool.from_function(
            coroutine=_estimate_trip_cost,
            name="estimate_trip_cost",
            description="按地点库里的门票、餐厅人均和酒店价格估算一趟旅行大概花多少钱。用户问预算、要花多少钱时用它。",
            args_schema=TripCostArgs,
        ),
        StructuredTool.from_function(
            coroutine=_estimate_travel_time,
            name="estimate_travel_time",
            description="查同一个城市里几个地点两两之间的交通耗时和距离。用户问从 A 到 B 多久、哪些地方顺路时用它。",
            args_schema=TravelTimeArgs,
        ),
    ]


# ---------------------------------------------------------------- 循环


def _system_prompt(slots: TripSlots) -> str:
    known = slots.model_dump_json(exclude_defaults=True, exclude={"suggested_destinations"})
    return "\n".join(
        [
            "你是旅行助手，负责回答旅行相关的问题、推荐目的地、帮用户了解某个地方。",
            f"今天是 {date.today().isoformat()}（周{'一二三四五六日'[date.today().weekday()]}），涉及日期时以此推算。",
            "你可以调用工具查资料。先想清楚需要什么信息，再决定调用哪个工具；一次可以调用多个。",
            "规则：",
            "1. 具体数字（价格、营业时间、班次、天气、距离）只能来自工具返回的内容和用户说过的话，"
            "工具没有查到就明说没查到，不要编造。",
            "2. 广为人知的一般情况可以直接回答，用“一般”“通常”说明，并告诉用户去哪确认细节："
            "比如两个城市之间通常坐高铁还是飞机、热门景点要不要提前实名预约、某地有哪些代表菜。"
            "不要因为工具没查到就整个拒答。",
            "3. 给出数字时顺带说明来源：天气预报、地点库数据、按价格估算，或“大约”。",
            "4. 资料已经够回答时就直接回答，不要重复调用同一个工具查同样的内容。",
            "5. 推荐目的地时说明理由，理由要来自查到的资料。",
            "6. 你不负责生成完整的逐日行程。用户想要行程时，告诉他直接说“帮我规划”即可。",
            "7. 用中文回答，先给结论再补充，简洁、条理清楚，适合直接展示给用户。",
            f"用户目前已经确定的信息：{known}",
        ]
    )


def _history_messages(context: list[dict[str, Any]]) -> list[Any]:
    messages: list[Any] = []
    for item in context[-_HISTORY_LIMIT:]:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if item.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif item.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "").strip()


async def run_react_agent(
    question: str,
    context: list[dict[str, Any]],
    slots: TripSlots,
    *,
    llm: Any,
    tools: list[StructuredTool],
    on_tool_call: EventCallback | None = None,
    on_tool_result: EventCallback | None = None,
    on_token: TokenCallback | None = None,
) -> ReactResult:
    """跑一轮 ReAct，返回最终回答和中间每一步工具调用。

    传了 on_token 且模型支持 astream 时，每一轮都流式调用：模型直接回答的那一轮，
    文字边生成边推给前端。
    """
    tools_by_name = {tool.name: tool for tool in tools}
    model = llm.bind_tools(tools)
    messages: list[Any] = [SystemMessage(content=_system_prompt(slots)), *_history_messages(context), HumanMessage(content=question)]
    steps: list[ReactStep] = []

    for _round in range(MAX_TOOL_ROUNDS):
        reply = await _call(model, messages, on_token)
        tool_calls = list(getattr(reply, "tool_calls", None) or [])
        if not tool_calls:
            return ReactResult(answer=_text(reply) or _NO_ANSWER, steps=steps)

        messages.append(reply)
        for call in tool_calls:
            name, arguments, call_id = call.get("name", ""), dict(call.get("args") or {}), call.get("id") or ""
            if on_tool_call is not None:
                await on_tool_call({"call_id": call_id, "tool": name, "arguments": arguments})
            outcome = await _run_tool(tools_by_name.get(name), name, arguments)
            steps.append(ReactStep(tool=name, arguments=arguments, outcome=outcome))
            if on_tool_result is not None:
                await on_tool_result(
                    {
                        "call_id": call_id,
                        "tool": name,
                        "status": outcome.status,
                        "result": {
                            "tool_name": name,
                            "worker": arguments.get("topic") or name,
                            "status": outcome.status,
                            "answer": outcome.text[:400],
                            "evidence_count": outcome.evidence_count,
                            "warnings": outcome.warnings,
                        },
                    }
                )
            messages.append(ToolMessage(content=outcome.text, tool_call_id=call_id))

    # 工具轮数用完：不再给工具，要求直接基于已有结果回答。
    messages.append(HumanMessage(content="请基于上面已经查到的内容直接回答，不要再调用工具；没查到的部分如实说明。"))
    final = await _call(llm, messages, on_token)
    return ReactResult(answer=_text(final) or _NO_ANSWER, steps=steps)


async def _call(model: Any, messages: list[Any], on_token: TokenCallback | None) -> Any:
    """调用一次模型。能流式就流式，把正文增量交给 on_token；出现工具调用时撤回已推出的文字。"""
    if on_token is None or not hasattr(model, "astream"):
        return await model.ainvoke(messages)
    merged = None
    streamed = False
    calling_tools = False
    async for chunk in model.astream(messages):
        merged = chunk if merged is None else merged + chunk
        if getattr(chunk, "tool_call_chunks", None):
            if streamed and not calling_tools:
                await on_token("reset", "")
            calling_tools = True
            continue
        delta = _text_delta(chunk)
        if delta and not calling_tools:
            streamed = True
            await on_token("delta", delta)
    return merged if merged is not None else await model.ainvoke(messages)


def _text_delta(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


async def _run_tool(tool: StructuredTool | None, name: str, arguments: dict[str, Any]) -> ToolOutcome:
    if tool is None:
        return ToolOutcome(text=f"没有名为 {name} 的工具。", status="failed", warnings=["react_tool_error:unknown_tool"])
    try:
        outcome = await tool.ainvoke(arguments)
    except Exception as exc:
        app_logger.warning(f"ReAct 工具 {name} 执行失败: {type(exc).__name__}: {exc}")
        return ToolOutcome(
            text="这个工具这次没有成功返回结果。",
            status="failed",
            warnings=["react_tool_error:execution_failed"],
        )
    if isinstance(outcome, ToolOutcome):
        return outcome
    return ToolOutcome(text=outcome if isinstance(outcome, str) else json.dumps(outcome, ensure_ascii=False))


__all__ = [
    "MAX_TOOL_ROUNDS",
    "ReactResult",
    "ReactStep",
    "ToolOutcome",
    "build_react_tools",
    "run_react_agent",
]
