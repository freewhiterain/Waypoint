"""统一 SSE 事件契约。"""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.planning import EvidenceSufficiency


SSEType = Literal[
    "task",
    "plan",
    "worker",
    "evidence",
    "approval",
    "token",
    "token_reset",
    "result",
    "error",
    "done",
    "tool_call",
    "tool_result",
    "subagent_started",
    "subagent_completed",
    "evidence_collected",
    "subagent_tool_call",
    "follow_up_search",
    "research_conflict",
    "subagent_tool_completed",
]


class SSEEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    type: SSEType
    task_id: str | None = None
    conversation_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    def legacy_payload(self) -> dict[str, Any]:
        result = self.model_dump(mode="json", exclude_none=True)
        if self.type == "token":
            result["content"] = self.payload.get("content", "")
        if self.type == "error":
            result["message"] = self.payload.get("message", "未知错误")
        if self.type == "tool_call":
            payload = result.get("payload", {})
            for field in ("tool", "call_id", "arguments"):
                if field in payload:
                    result[field] = payload[field]
        if self.type == "tool_result":
            payload = result.get("payload", {})
            for field in ("call_id", "tool", "status", "result", "partial_values"):
                if field in payload:
                    result[field] = payload[field]
        return result
