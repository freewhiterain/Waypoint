"""多 Agent 旅行规划的公共数据契约。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskType = Literal["attractions", "transport", "hotel", "food", "weather"]
ResearchMode = Literal["normal", "deep"]
WorkerStatus = Literal["completed", "partial", "unavailable", "failed"]
PlanningStatus = Literal["draft", "degraded"]


class EvidenceSufficiency(BaseModel):
    """Typed outcome metadata for a normalized provider response."""

    status: Literal["sufficient", "partial", "empty", "failed"]
    evidence_count: int = Field(default=0, ge=0)
    reason_code: str = Field(min_length=1)


class TravelRequirement(BaseModel):
    """一次国内旅行规划所需的最小结构化需求。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    origin: str | None = Field(default=None, min_length=1, max_length=80)
    destination: str = Field(min_length=1, max_length=80)
    departure_date: date
    days: int = Field(ge=1, le=30)
    assumptions: list[str] = Field(default_factory=list)
    adults: int = Field(default=1, ge=1, le=30)
    children: int = Field(default=0, ge=0, le=20)
    styles: list[str] = Field(default_factory=list)
    special_needs: list[str] = Field(default_factory=list)
    transport_preferences: list[str] = Field(default_factory=list)
    accommodation_preferences: list[str] = Field(default_factory=list)
    food_preferences: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def destination_must_differ_from_origin(self):
        if self.origin is not None and self.origin == self.destination:
            raise ValueError("出发地和目的地不能相同")
        return self


class TravelRequirementDraft(BaseModel):
    """自然语言需求提取的中间结果；缺失字段不会被模型臆造。"""

    origin: str | None = None
    destination: str | None = None
    departure_date: date | None = None
    days: int | None = Field(default=None, ge=1, le=30)
    adults: int = Field(default=1, ge=1, le=30)
    children: int = Field(default=0, ge=0, le=20)
    styles: list[str] = Field(default_factory=list)
    special_needs: list[str] = Field(default_factory=list)
    food_preferences: list[str] = Field(default_factory=list)

    def missing_fields(self) -> list[str]:
        labels = {"origin": "出发地", "destination": "目的地", "departure_date": "出发日期", "days": "出行天数"}
        return [label for field, label in labels.items() if getattr(self, field) is None]

    def to_requirement(self) -> TravelRequirement:
        missing = self.missing_fields()
        if missing:
            raise ValueError(f"旅行需求缺少：{', '.join(missing)}")
        return TravelRequirement(**self.model_dump())

class TripDraftRecord(BaseModel):
    """会话级常驻行程草稿的持久化载体。"""

    user_id: str
    conversation_id: str
    version: int = 1
    requirement: dict[str, Any] = Field(default_factory=dict)
    content: dict[str, Any] = Field(default_factory=dict)


class ResearchTask(BaseModel):
    """Planner 生成的可调度研究任务。"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    task_type: TaskType
    query: str
    research_mode: ResearchMode = "normal"
    dependencies: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """事实性结论的最小证据格式。"""

    id: str | None = None
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_url: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateOption(BaseModel):
    """Worker 返回的结构化候选项。"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    category: str
    description: str = ""
    estimated_cost: float | None = Field(default=None, ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


class WorkerResult(BaseModel):
    """所有专业 Worker 的统一输出。"""

    task_id: str
    worker: TaskType
    status: WorkerStatus
    summary: str
    options: list[CandidateOption] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_mock: bool = False


class TimeSlot(BaseModel):
    period: Literal["morning", "afternoon", "evening"]
    title: str
    description: str = ""
    location: str | None = None
    travel_minutes: int | None = Field(default=None, ge=0)
    opening_window: str | None = None
    estimated_cost: float | None = Field(default=None, ge=0)
    evidence_indexes: list[int] = Field(default_factory=list)


class ItineraryDay(BaseModel):
    day: int = Field(ge=1)
    date: date
    slots: list[TimeSlot]
    notes: list[str] = Field(default_factory=list)


class TravelPlanDraft(BaseModel):
    requirement: TravelRequirement
    itinerary: list[ItineraryDay]
    worker_results: list[WorkerResult]
    evidence: list[Evidence]
    warnings: list[str] = Field(default_factory=list)
    status: PlanningStatus = "draft"
    degraded_reason: Literal["no_llm_or_provider", "worker_unavailable", "provider_degraded"] | None = None
