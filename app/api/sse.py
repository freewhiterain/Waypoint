"""SSE 帧与事件序号：chat 和 tools 两条流共用。"""

from __future__ import annotations

import json

from app.schemas.events import SSEEvent


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class SSEEventBuilder:
    """为同一条流生成序号连续的事件载荷。"""

    def __init__(self, task_id: str, conversation_id: str | None):
        self.task_id = task_id
        self.conversation_id = conversation_id
        self.sequence = 0

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def __call__(self, event_type: str, payload: dict | None = None) -> dict:
        return SSEEvent(
            type=event_type,
            task_id=self.task_id,
            conversation_id=self.conversation_id,
            sequence=self.next_sequence(),
            payload=payload or {},
        ).legacy_payload()
