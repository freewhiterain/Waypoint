"""会话级行程槽位：跨轮次、跨线路累积的规划信息。"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# 进入线路 2（规划）必须填满的三个槽。
REQUIRED_SLOTS = ("destination", "departure_date", "days")

TurnIntent = Literal["plan", "open_question", "recommend_destination", "research", "chat"]
ResearchTopic = Literal["attractions", "weather", "transport", "hotel", "food"]


class TripSlots(BaseModel):
    """一个会话当前已知的规划信息，存在 conversation.extra_info["trip_slots"]。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    destination: str | None = Field(default=None, max_length=80)
    departure_date: date | None = None
    days: int | None = Field(default=None, ge=1, le=30)
    origin: str | None = Field(default=None, max_length=80)
    adults: int | None = Field(default=None, ge=1, le=30)
    children: int | None = Field(default=None, ge=0, le=20)
    styles: list[str] = Field(default_factory=list)
    food_preferences: list[str] = Field(default_factory=list)
    # 模型推荐过、用户还没明确接受的城市。只是候选，不等于目的地。
    suggested_destinations: list[str] = Field(default_factory=list)
    # 看过行程后提的修改：不想去的地点、希望轻松些的第几天。换目的地时清空。
    avoid_places: list[str] = Field(default_factory=list)
    relaxed_days: list[int] = Field(default_factory=list)

    def missing(self) -> list[str]:
        return [name for name in REQUIRED_SLOTS if getattr(self, name) in (None, "")]

    def is_complete(self) -> bool:
        return not self.missing()

    def form_values(self) -> dict[str, Any]:
        """已填的三个必填槽，作为表单预填值；表单只会询问其中缺的项。"""
        return self.model_dump(mode="json", include=set(REQUIRED_SLOTS), exclude_none=True)

    def requirement_context(self) -> dict[str, Any]:
        """表单不问、但要带进规划需求的信息（人数、偏好、出发地）。"""
        context = self.model_dump(
            mode="json",
            include={"origin", "adults", "children", "styles", "food_preferences", "avoid_places", "relaxed_days"},
            exclude_none=True,
        )
        return {key: value for key, value in context.items() if value != []}


class SlotUpdate(BaseModel):
    """模型在一轮对话里识别出的槽位变化；没提到的项留空。"""

    destination: str | None = Field(default=None, description="用户这一轮确定要去的城市；只是在询问或没确定时留空")
    departure_date: str | None = Field(default=None, description="出发日期，YYYY-MM-DD；节假日等说不准具体哪天的留空")
    days: int | None = Field(default=None, description="游玩天数")
    origin: str | None = Field(default=None, description="出发城市")
    adults: int | None = Field(default=None, description="成人人数")
    children: int | None = Field(default=None, description="儿童人数")
    styles: list[str] = Field(default_factory=list, description="游玩兴趣或风格，如 美食、熊猫、亲子")
    food_preferences: list[str] = Field(default_factory=list, description="口味或饮食要求，如 不吃辣、素食")
    avoid_places: list[str] = Field(
        default_factory=list, description="用户这一轮明确说不想去、要换掉的具体地点或餐厅，照抄用户说的名字"
    )
    relaxed_days: list[int] = Field(
        default_factory=list, description="用户希望安排得轻松些的第几天（从 1 开始）；整趟都要轻松时列出每一天"
    )

    def is_empty(self) -> bool:
        return not self.model_dump(exclude_defaults=True)


class TurnUnderstanding(BaseModel):
    """模型对一轮用户消息的理解：意图 + 槽位变化。走哪条线路由代码决定。"""

    intent: TurnIntent = Field(
        description=(
            "plan=用户要做行程规划；open_question=询问某地的旅行信息；"
            "recommend_destination=还没定去哪、想要推荐；research=要求查询或深入研究某个方面；"
            "chat=寒暄或与旅行无关"
        )
    )
    slots: SlotUpdate = Field(default_factory=SlotUpdate)
    research_topic: ResearchTopic | None = Field(default=None, description="intent=research 时研究哪个方面")
    deep_research: bool = Field(default=False, description="用户是否要求深入/仔细研究")
    reply: str | None = Field(default=None, description="intent=chat 时直接回复用户的话")
