"""Per-turn routing for the conversational travel entry point."""

from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.schemas.tools import AgentToolArguments, AgentToolCall, MainAgentDecision
from app.services.planning import KNOWN_DESTINATIONS, RequirementExtractor


_AFFIRMATIONS = {"好的", "好啊", "好呀", "可以", "开始吧", "行", "行啊", "好的开始吧"}
_PLANNING_MARKERS = ("规划", "安排行程", "制定行程", "做个行程", "做攻略", "旅行计划")
_OPEN_QUESTION_MARKERS = (
    "最近",
    "什么好玩",
    "好玩吗",
    "哪里好玩",
    "什么时候去",
    "天气",
    "气温",
    "热门景点",
    "介绍一下",
    "了解一下",
)
_AGENT_TOOL_MARKERS = (
    "\u6df1\u5165\u7814\u7a76",
    "\u4ed4\u7ec6\u7814\u7a76",
    "\u8be6\u7ec6\u7814\u7a76",
    "\u67e5\u4e00\u4e0b",
    "\u67e5\u67e5",
    "\u8054\u7f51\u67e5",
    "\u6700\u65b0",
)
_DEEP_RESEARCH_MARKERS = ("\u6df1\u5165\u7814\u7a76", "\u4ed4\u7ec6\u7814\u7a76", "\u8be6\u7ec6\u7814\u7a76")
_AGENT_TOOL_WORKER_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("weather", ("\u5929\u6c14", "\u6c14\u6e29", "\u964d\u96e8", "\u4e0b\u96e8", "\u6c14\u5019")),
    ("transport", ("\u4ea4\u901a", "\u600e\u4e48\u53bb", "\u9ad8\u94c1", "\u98de\u673a", "\u673a\u7968", "\u5730\u94c1")),
    ("hotel", ("\u4f4f\u5bbf", "\u9152\u5e97", "\u6c11\u5bbf", "\u4f4f\u54ea", "\u5ba2\u6808")),
    ("food", ("\u7f8e\u98df", "\u5403", "\u9910\u5385", "\u5c0f\u5403", "\u83dc")),
)
_PROACTIVE_OFFER = "需要我帮你规划一下旅行吗"
# 与 app/api/v1/conversations.py 写入的 extra_info["kind"] 对应，
# 由 chat._load_recent_context 带进 context。
_OFFER_KIND = "conversation_offer"


class MainAgentService:
    def __init__(self, use_llm: bool | None = None) -> None:
        self.use_llm = use_llm

    async def decide(self, message: str, context: list[dict[str, Any]]) -> MainAgentDecision:
        text = message.strip()
        normalized = self._normalize(text)

        if self._is_explicit_planning_request(normalized):
            initial_values, requirement_context = await self._prefill(text)
            return MainAgentDecision(
                action="collect_trip_requirements",
                reason="用户明确请求规划",
                initial_values=initial_values,
                requirement_context=requirement_context,
            )
        agent_tool_call = self._build_agent_tool_call(text, normalized)
        if agent_tool_call is not None:
            return MainAgentDecision(
                action="invoke_agent_tool",
                reason="\u7528\u6237\u8bf7\u6c42\u5bf9\u7279\u5b9a\u9886\u57df\u8fdb\u884c\u6df1\u5165\u7814\u7a76",
                tool_call=agent_tool_call,
            )
        if self._is_destination_recommendation(normalized):
            return MainAgentDecision(action="recommend_destination", reason="用户请求目的地推荐")
        if self._is_open_question(normalized):
            return MainAgentDecision(action="answer_open_question", reason="用户提出开放式旅行问题")
        if normalized in _AFFIRMATIONS and self._has_recent_proactive_offer(context):
            return MainAgentDecision(action="collect_trip_requirements", reason="用户确认主动邀请")
        if self._llm_enabled():
            return await self._decide_with_llm(text, context)
        return MainAgentDecision(action="direct_response", reason="未识别到明确的规划意图")

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(text.split()).rstrip("。！？!?")

    @staticmethod
    def _is_destination_recommendation(text: str) -> bool:
        if MainAgentService._mentions_known_destination(text):
            return False
        if any(
            marker in text
            for marker in ("没想好", "不知道去哪", "去哪玩", "去哪", "去哪里", "目的地", "城市")
        ):
            return True
        return "推荐" in text and any(
            marker in text for marker in ("地方", "目的地", "城市")
        )

    @staticmethod
    def _is_open_question(text: str) -> bool:
        if any(marker in text for marker in _OPEN_QUESTION_MARKERS):
            return True
        return MainAgentService._mentions_known_destination(text) and any(
            marker in text for marker in ("推荐", "亲子", "景点", "地方", "好玩", "玩")
        )

    @staticmethod
    def _build_agent_tool_call(message: str, normalized: str) -> AgentToolCall | None:
        if not any(marker in normalized for marker in _AGENT_TOOL_MARKERS):
            return None
        destination = next(
            (candidate for candidate in KNOWN_DESTINATIONS if candidate in normalized),
            None,
        )
        if destination is None:
            return None

        worker = next(
            (
                candidate
                for candidate, markers in _AGENT_TOOL_WORKER_MARKERS
                if any(marker in normalized for marker in markers)
            ),
            "attractions",
        )
        return AgentToolCall(
            name=f"research_{worker}",
            arguments=AgentToolArguments(
                destination=destination,
                query=message,
                research_mode=(
                    "deep" if any(marker in normalized for marker in _DEEP_RESEARCH_MARKERS) else "normal"
                ),
            ),
        )

    @staticmethod
    def _is_explicit_planning_request(text: str) -> bool:
        return any(marker in text for marker in _PLANNING_MARKERS)

    @staticmethod
    def _has_recent_proactive_offer(context: list[dict[str, Any]]) -> bool:
        """判断上一条非系统消息是否就是那句主动邀请。

        优先认结构化标记 extra_info["kind"] == "conversation_offer"（生产者
        app/api/v1/conversations.py 建会话时就打上了）。原先只做文案全等比较，
        问候语改一个字就会让"好的"不再进入规划流程，且无任何报错。

        文案兜底必须保留：这次改动之前入库的会话行没有 kind 标记，去掉兜底会让
        所有老会话的确认路径突然失效。兜底保持全等而不是子串——"需要我帮你规划
        一下旅行吗？现在开始吧。"是另一句话，不该被当成待确认的邀请。
        """
        for item in reversed(context):
            if item.get("role") == "system":
                continue
            if item.get("role") != "assistant":
                return False
            if item.get("kind") == _OFFER_KIND:
                return True
            return MainAgentService._normalize(str(item.get("content", ""))) == _PROACTIVE_OFFER
        return False

    def _llm_enabled(self) -> bool:
        return self.use_llm is not False and bool(settings.llm_api_key)

    async def _prefill(self, message: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """返回 (表单预填值, 需求上下文)。

        表单只认目的地/日期/天数，前端按 initial_values 原样回填，所以这里不往里加字段；
        人数、预算、偏好另放进需求上下文，表单提交时由后端合并。只写用户明确说过的值，
        默认值（1 个大人、0 个孩子、空列表）不写，避免把"没说"伪装成"说了"。
        """
        draft = await RequirementExtractor().extract(message, use_llm=False)
        initial_values = draft.model_dump(
            mode="json",
            include={"destination", "departure_date", "days"},
            exclude_none=True,
        )
        context = draft.model_dump(
            mode="json",
            include={"origin", "budget", "styles", "special_needs", "food_preferences"},
            exclude_none=True,
        )
        context = {key: value for key, value in context.items() if value != []}
        if draft.adults != 1 or draft.children != 0:
            context["adults"] = draft.adults
            context["children"] = draft.children
        return initial_values, context

    @staticmethod
    def _mentions_known_destination(text: str) -> bool:
        return any(destination in text for destination in KNOWN_DESTINATIONS)

    async def _decide_with_llm(
        self,
        message: str,
        context: list[dict[str, Any]],
    ) -> MainAgentDecision:
        try:
            from app.agents.llm import get_llm

            structured = get_llm().with_structured_output(MainAgentDecision)
            decision = MainAgentDecision.model_validate(
                await structured.ainvoke(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是旅行对话入口路由器。只根据本轮用户消息判断 action。"
                                "历史中的目的地、日期、天数和工具结果不得用来推断用户想规划。"
                                "只有用户当前明确要规划时才返回 collect_trip_requirements。"
                                "开放式提问返回 answer_open_question，目的地推荐返回 recommend_destination，"
                                "其他情况返回 direct_response。"
                            ),
                        },
                        {"role": "assistant", "content": f"最近上下文：{json.dumps(context, ensure_ascii=False)}"},
                        {"role": "user", "content": message},
                    ]
                )
            )
        except Exception:
            return MainAgentDecision(action="direct_response", reason="路由模型不可用，保守地直接回复")

        if decision.action == "invoke_agent_tool" and decision.tool_call is None:
            return MainAgentDecision(
                action="direct_response",
                reason="路由模型未提供有效的工具调用参数",
            )

        if decision.action != "collect_trip_requirements":
            return decision.model_copy(update={"initial_values": {}, "requirement_context": {}})
        initial_values, requirement_context = await self._prefill(message)
        return decision.model_copy(
            update={"initial_values": initial_values, "requirement_context": requirement_context}
        )
