from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.planning import TaskType, WorkerStatus


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
    initial_values: dict[str, Any] = Field(default_factory=dict)
    # 表单只问目的地/日期/天数；人数、偏好从用户原话里提取后存在这里，
    # 表单提交时由后端并进 TravelRequirement，不经过前端。
    requirement_context: dict[str, Any] = Field(default_factory=dict)
    tool_call: AgentToolCall | None = None


class TripFormArguments(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    destination: str = Field(min_length=1, max_length=80)
    departure_date: date
    days: int = Field(ge=1, le=30)


class TripFormResult(TripFormArguments):
    pass


class ToolCallPayload(BaseModel):
    call_id: str
    tool: Literal["collect_trip_requirements"]
    arguments: dict[str, Any]


class ToolResultRequest(BaseModel):
    tool: Literal["collect_trip_requirements"]
    status: Literal["completed", "recommend_destination", "cancelled"]
    result: TripFormResult | None = None
    partial_values: dict[str, Any] = Field(default_factory=dict)
