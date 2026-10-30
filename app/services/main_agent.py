"""Per-turn understanding for the conversational travel entry point.

每一轮由同一次 LLM 调用给出"意图 + 槽位变化"（TurnUnderstanding）。走哪条线路
不在这里决定：chat 层把槽位变化并进会话槽位后，按固定规则路由。

LLM 没配置或这一轮没按要求返回时，退回关键词规则和正则提取，保证对话不中断。
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from app.config import settings
from app.schemas.slots import SlotUpdate, TripSlots, TurnUnderstanding
from app.schemas.tools import AgentToolArguments, AgentToolCall, MainAgentDecision
from app.utils.logger import app_logger
from app.services.requirements import KNOWN_DESTINATIONS, RequirementExtractor


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
_UNDERSTAND_ATTEMPTS = 2
# 与 app/api/v1/conversations.py 写入的 extra_info["kind"] 对应，
# 由 chat._load_recent_context 带进 context。
_OFFER_KIND = "conversation_offer"


class MainAgentService:
    def __init__(self, use_llm: bool | None = None) -> None:
        self.use_llm = use_llm

    async def decide(
        self,
        message: str,
        context: list[dict[str, Any]],
        slots: TripSlots | None = None,
    ) -> MainAgentDecision:
        text = message.strip()
        slots = slots or TripSlots()
        if self._llm_enabled():
            understanding = await self._understand(text, context, slots)
            if understanding is not None:
                return self._decision_from_understanding(text, understanding, slots)
        return await self._decide_by_rules(text, context)

    async def _decide_by_rules(self, text: str, context: list[dict[str, Any]]) -> MainAgentDecision:
        """LLM 不可用时的兜底：关键词判断意图，正则提取槽位（只看当前这一条消息）。"""
        normalized = self._normalize(text)
        slot_update = await self._slot_update_by_rules(text)

        if self._is_explicit_planning_request(normalized):
            return MainAgentDecision(
                action="collect_trip_requirements",
                reason="用户明确请求规划",
                slot_update=slot_update,
            )
        agent_tool_call = self._build_agent_tool_call(text, normalized)
        if agent_tool_call is not None:
            return MainAgentDecision(
                action="invoke_agent_tool",
                reason="用户请求对特定领域进行深入研究",
                tool_call=agent_tool_call,
                slot_update=slot_update,
            )
        if self._is_destination_recommendation(normalized):
            return MainAgentDecision(
                action="recommend_destination", reason="用户请求目的地推荐", slot_update=slot_update
            )
        if self._is_open_question(normalized):
            return MainAgentDecision(
                action="answer_open_question", reason="用户提出开放式旅行问题", slot_update=slot_update
            )
        if normalized in _AFFIRMATIONS and self._has_recent_proactive_offer(context):
            return MainAgentDecision(action="collect_trip_requirements", reason="用户确认主动邀请")
        return MainAgentDecision(
            action="direct_response", reason="未识别到明确的规划意图", slot_update=slot_update
        )

    def _decision_from_understanding(
        self,
        text: str,
        understanding: TurnUnderstanding,
        slots: TripSlots,
    ) -> MainAgentDecision:
        update = understanding.slots
        if understanding.intent == "plan":
            return MainAgentDecision(
                action="collect_trip_requirements", reason="模型判断用户要规划行程", slot_update=update
            )
        if understanding.intent == "research":
            destination = (update.destination or slots.destination or "").strip()
            if destination:
                worker = understanding.research_topic or "attractions"
                return MainAgentDecision(
                    action="invoke_agent_tool",
                    reason="模型判断用户要研究某个方面",
                    tool_call=AgentToolCall(
                        name=f"research_{worker}",
                        arguments=AgentToolArguments(
                            destination=destination,
                            query=text[:500],
                            research_mode="deep" if understanding.deep_research else "normal",
                        ),
                    ),
                    slot_update=update,
                )
            # 没有目的地就没法做定向研究，按开放问题回答。
            return MainAgentDecision(
                action="answer_open_question", reason="研究请求缺少目的地", slot_update=update
            )
        if understanding.intent == "recommend_destination":
            return MainAgentDecision(
                action="recommend_destination", reason="模型判断用户想要目的地推荐", slot_update=update
            )
        if understanding.intent == "open_question":
            return MainAgentDecision(
                action="answer_open_question", reason="模型判断用户在询问旅行信息", slot_update=update
            )
        return MainAgentDecision(
            action="direct_response",
            reason="模型判断为闲聊或与旅行无关",
            response=understanding.reply,
            slot_update=update,
        )

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

    async def _slot_update_by_rules(self, message: str) -> SlotUpdate:
        """正则提取当前消息里的槽位。默认值（1 个大人、0 个孩子）不算"用户说过"。"""
        draft = await RequirementExtractor().extract(message, use_llm=False)
        party_stated = draft.adults != 1 or draft.children != 0
        return SlotUpdate(
            destination=draft.destination,
            departure_date=draft.departure_date.isoformat() if draft.departure_date else None,
            days=draft.days,
            origin=draft.origin,
            adults=draft.adults if party_stated else None,
            children=draft.children if party_stated else None,
            styles=draft.styles,
            food_preferences=draft.food_preferences,
        )

    @staticmethod
    def _mentions_known_destination(text: str) -> bool:
        return any(destination in text for destination in KNOWN_DESTINATIONS)

    async def _understand(
        self,
        message: str,
        context: list[dict[str, Any]],
        slots: TripSlots,
    ) -> TurnUnderstanding | None:
        """一次 LLM 调用同时给出意图和槽位变化；失败或返回空时返回 None，由规则兜底。"""
        today = date.today()
        weekday = "一二三四五六日"[today.weekday()]
        system_prompt = "\n".join(
            [
                f"你是旅行助手的对话理解模块，以 JSON 格式输出。今天是 {today.isoformat()}（周{weekday}）。",
                "任务：判断用户这一轮消息的意图（intent），并找出这一轮里新出现或被修改的行程信息（slots）。",
                "填写 slots 的规则：",
                "1. 只填用户这一轮明确表达的信息；没提到的留空，不要重复已有槽位，不要猜测。",
                "2. destination：用户确定要去、或只围绕这一个城市提问时填；在多个城市之间比较、还没决定时留空。",
                "3. 当前槽位里的 suggested_destinations 是助手之前推荐的候选城市。用户表示接受"
                "（比如说就去那里、那帮我规划一下，或明确指了其中一个）时，把对应城市填入 destination；"
                "有多个候选而用户没有指明是哪个时留空。",
                "4. departure_date 换算成 YYYY-MM-DD：明天、下周六这类说法按今天换算；"
                "这周末/本周末指这周六，下周末指下周六；"
                "国庆、春节等节假日，或只说了月份的，不换算，留空。",
                "5. 用户说不喜欢某样东西、不要某样东西时，不要把它记进 styles。",
                "6. 两个人、我和女朋友是 adults=2；一家三口是 adults=2、children=1。",
                "7. 用户说放 N 天假、有 N 天假期或有 N 天时间，就是打算玩 N 天，填 days=N。",
                "8. 看过行程后要换掉某个景点或餐厅、说不想去某处，把名字照抄进 avoid_places；"
                "说第几天要轻松些、少安排点，把那几天填进 relaxed_days，整趟都要轻松就列出每一天。",
                "9. 用户点名一定要去、想去的具体景点或餐厅（如洪崖洞、宽窄巷子），照抄进 must_visit；"
                "泛泛的兴趣（美食、博物馆、夜景）只记进 styles。",
                "10. 用户说了预算，换算成整趟所有人加起来的总额（元）填进 budget："
                "说人均或每人时乘以人数，两千填 2000、3k 填 3000。"
                "用户问“要花多少钱”“预算够不够”是在提问，不是给预算，budget 留空。",
                "intent 只看用户这一轮想做什么：",
                "- plan：用户这一轮明确要求规划、安排行程或做攻略。"
                "问大概花多少钱、天气怎么样、某个景点值不值得去，即使带了城市和天数，也是 open_question。",
                "- 用户只是补充或修改信息（去哪、哪天、几个人、喜好）而没有要求规划时，不是 plan：用 chat，"
                "在 reply 里简短确认已记下，并问一句是否现在开始规划。slots 照常填写。",
                "- 例外：最近对话里助手上一条是“正在收集旅行需求。”，说明规划表单还没填完，"
                "用户这一轮补充信息视为继续规划，用 plan。",
                "- 已经给出过行程（最近对话里有“行程草稿”）后，用户要求调整行程（换掉地点、某天轻松些、"
                "改天数或日期）也用 plan，并按上面的规则填 slots。",
                "- 用户问的内容与旅行无关时用 chat，并在 reply 里礼貌说明只能帮忙处理旅行相关的事。",
            ]
        )
        user_prompt = "\n".join(
            [
                f"当前槽位：{slots.model_dump_json(exclude_defaults=True)}",
                f"最近对话：{json.dumps(context, ensure_ascii=False)}",
                f"用户这一轮：{message}",
            ]
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # 强制工具调用下模型偶尔仍会返回空结果，重试一次；还不行再交给规则兜底。
        for attempt in range(_UNDERSTAND_ATTEMPTS):
            try:
                from app.core.llm import get_llm

                structured = get_llm().with_structured_output(TurnUnderstanding)
                return TurnUnderstanding.model_validate(await structured.ainvoke(messages))
            except Exception as exc:
                app_logger.warning(
                    f"对话理解模型未返回有效结果（第 {attempt + 1} 次）: {type(exc).__name__}: {exc}"
                )
        return None
