"""行程历史记录：用户确认保存正式行程后追加的 Layer 2b 记忆。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Protocol

from app.schemas.governance import TripHistoryRecord
from app.utils.logger import app_logger


class TripHistoryRepository(Protocol):
    async def append(self, record: TripHistoryRecord) -> TripHistoryRecord: ...
    async def list(self, user_id: str) -> list[TripHistoryRecord]: ...


class InMemoryTripHistoryRepository:
    def __init__(self):
        self.records: list[TripHistoryRecord] = []

    async def append(self, record: TripHistoryRecord) -> TripHistoryRecord:
        stored = record.model_copy(deep=True)
        self.records.append(stored)
        return stored.model_copy(deep=True)

    async def list(self, user_id: str) -> list[TripHistoryRecord]:
        return [record.model_copy(deep=True) for record in self.records if record.user_id == user_id]


_PLACEHOLDER_TITLES = ("自由活动", "自由安排")


def _extract_visited_attractions(content: dict[str, Any]) -> list[str]:
    """上午、下午时段里安排的景点。晚上是餐饮，不算；"自由活动"这类占位标题也不算。

    行程里带着景点候选（worker_results）时，以候选名称为准：润色后的标题可能加了修饰语，
    但会保留地点名称。
    """
    known = [
        str(option.get("name"))
        for result in content.get("worker_results", []) or []
        if result.get("worker") == "attractions"
        for option in result.get("options", []) or []
        if option.get("name")
    ]
    attractions: list[str] = []
    for day in content.get("itinerary", []) or []:
        for slot in day.get("slots", []) or []:
            title = str(slot.get("title") or "").strip()
            if slot.get("period") not in ("morning", "afternoon") or not title:
                continue
            if any(marker in title for marker in _PLACEHOLDER_TITLES):
                continue
            name = next((item for item in known if item in title), None) if known else title
            if name and name not in attractions:
                attractions.append(name)
    return attractions


def build_trip_history_record(
    user_id: str, source_itinerary_id: str, content: dict[str, Any]
) -> TripHistoryRecord | None:
    """从已保存的行程内容里提取行程历史；字段缺失或格式不符时返回 None，不编造数据。"""
    requirement = content.get("requirement")
    if not isinstance(requirement, dict):
        return None
    destination = requirement.get("destination")
    start_date_raw = requirement.get("departure_date")
    days = requirement.get("days")
    if not destination or not start_date_raw or not isinstance(days, int) or days < 1:
        return None
    try:
        start_date = date.fromisoformat(start_date_raw)
    except (TypeError, ValueError):
        return None
    end_date = start_date + timedelta(days=days - 1)
    return TripHistoryRecord(
        user_id=user_id,
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        visited_attractions=_extract_visited_attractions(content),
        source_itinerary_id=source_itinerary_id,
    )


async def record_trip_history_from_itinerary(
    user_id: str, source_itinerary_id: str, content: dict[str, Any], repository: TripHistoryRepository
) -> TripHistoryRecord | None:
    """构建并追加行程历史；任何异常都只记录 warning，不向上抛出——这是保存行程这个
    主动作的次要副作用，不能因为它失败而拖垮行程保存本身。"""
    try:
        record = build_trip_history_record(user_id, source_itinerary_id, content)
        if record is None:
            app_logger.warning(f"行程内容缺少必要字段，跳过行程历史记录: user={user_id}")
            return None
        return await repository.append(record)
    except Exception as exc:
        app_logger.warning(f"追加行程历史记录失败: user={user_id} error={exc}")
        return None
