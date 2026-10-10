"""长期偏好默认值解析：把已确认的偏好画像映射为规划请求的默认值。"""

from __future__ import annotations

from typing import Any

from app.memory.service import PreferenceRepository
from app.schemas.planning import TravelRequirement
from app.utils.logger import app_logger

LIST_PREFERENCE_KEYS = (
    "styles",
    "food_preferences",
    "accommodation_preferences",
    "transport_preferences",
    "special_needs",
)


async def resolve_preference_defaults(user_id: str, repository: PreferenceRepository) -> dict[str, Any]:
    """读取该用户已确认的偏好，按 key 取确认时间最新的一条，过滤词表外/类型不匹配的记录。"""
    records = await repository.list(user_id)
    valid_keys = set(LIST_PREFERENCE_KEYS)

    latest_value_by_key: dict[str, Any] = {}
    for record in sorted(records, key=lambda item: item.confirmed_at):
        if record.key in valid_keys:
            latest_value_by_key[record.key] = record.value

    defaults: dict[str, Any] = {}
    for key, value in latest_value_by_key.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            defaults[key] = value
        else:
            app_logger.warning(f"忽略类型不匹配的长期偏好: user={user_id} key={key} value={value!r}")
    return defaults


def apply_preference_defaults(requirement: TravelRequirement, defaults: dict[str, Any]) -> TravelRequirement:
    """只填充 requirement 里为空的字段，已有内容一律不动。返回一个新的 TravelRequirement。"""
    updates: dict[str, Any] = {}
    for key in LIST_PREFERENCE_KEYS:
        if key in defaults and not getattr(requirement, key):
            updates[key] = defaults[key]
    if not updates:
        return requirement
    return requirement.model_copy(update=updates)
