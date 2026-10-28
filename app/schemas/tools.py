from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.planning import TaskType, WorkerStatus
from app.schemas.slots import SlotUpdate


MainAgentAction = Literal[
    "collect_trip_requirements",
    "answer_open_question",
    "recommend_destination",
    "direct_response",
    "invoke_agent_tool",
]

AgentToolName = Literal[
    "research_attractions",
    "research_weather",
    "research_transport",
    "research_hotel",
    "research_food",
]


class AgentToolArguments(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    destination: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=500)
    research_mode: Literal["normal", "deep"] = "normal"
    departure_date: date | None = None
    days: int = Field(default=1, ge=1, le=30)


class AgentToolCall(BaseModel):
    name: AgentToolName
    arguments: AgentToolArguments


class AgentToolResult(BaseModel):
    """Public agent-tool outcome; raw provider payloads never cross this boundary."""

    tool_name: str
    worker: TaskType | None = None
    status: WorkerStatus
    answer: str
    evidence_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class MainAgentDecision(BaseModel):
    action: MainAgentAction
    reason: str
    response: str | None = None
    # 这一轮识别出的槽位变化。chat 层把它并进会话槽位，再由代码决定走哪条线路。
    slot_update: SlotUpdate = Field(default_factory=SlotUpdate)
    tool_call: AgentToolCall | None = None


class TripFormArguments(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    destination: str = Field(min_length=1, max_length=80)
    departure_date: date
    days: int = Field(ge=1, le=30)


class TripFormResult(TripFormArguments):
    # 表单里的"补充要求"：想去的地方、忌口、节奏等，提交时解析进需求。
    notes: str = Field(default="", max_length=300)


class ToolCallPayload(BaseModel):
    call_id: str
    tool: Literal["collect_trip_requirements"]
    arguments: dict[str, Any]


class ToolResultRequest(BaseModel):
    tool: Literal["collect_trip_requirements"]
    status: Literal["completed", "recommend_destination", "cancelled"]
    result: TripFormResult | None = None
    partial_values: dict[str, Any] = Field(default_factory=dict)
