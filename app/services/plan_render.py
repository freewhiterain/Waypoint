"""把行程草稿渲染成展示给用户的 Markdown，并把内部告警码翻译成用户能看懂的说明。"""

from __future__ import annotations

import re

from app.schemas.planning import TravelPlanDraft


_WORKER_LABELS = {"attractions": "景点", "hotel": "住宿", "food": "美食", "weather": "天气", "transport": "交通"}
_WARNING_TEXT = {
    "travel_time_estimated:no_route_data": "部分地点之间没有查到真实交通耗时，顺序按所在区域粗略安排。",
    "planning_degraded:provider_degraded": "部分信息来源不可用，行程基于现有资料生成。",
    "itinerary_selection_fallback:llm_failed": "智能选点暂时不可用，本次按固定规则选点。",
}
_NO_EVIDENCE = re.compile(r"^No (?:evidence is available|provider returned evidence) for (\w+)\.$")
_MAX_USER_WARNINGS = 8


def user_facing_warnings(warnings: list[str]) -> list[str]:
    """把内部告警整理成给用户看的中文说明：翻译已知代码，丢掉只对开发者有意义的条目。"""
    shown: list[str] = []
    for warning in warnings:
        text = _WARNING_TEXT.get(warning)
        if text is None and warning.startswith("itinerary_constraint:"):
            text = warning.split(":", 2)[-1]
        if text is None and (match := _NO_EVIDENCE.match(warning)):
            text = f"没有查到{_WORKER_LABELS.get(match.group(1), match.group(1))}的资料。"
        if text is None:
            # 其余英文或代码形式的告警（去重、证据校验的过程记录）不展示；模型写的中文说明保留。
            chinese = len(re.findall(r"[一-鿿]", warning))
            text = warning if chinese >= max(4, len(warning) // 4) else None
        if text and text not in shown:
            shown.append(text)
    return shown[:_MAX_USER_WARNINGS]


def render_plan_markdown(draft: TravelPlanDraft) -> str:
    lines = [f"# {draft.requirement.destination}{draft.requirement.days}日行程草稿", ""]
    if draft.requirement.assumptions:
        lines.append("## 假设说明")
        lines.extend(f"- {assumption}" for assumption in draft.requirement.assumptions)
        lines.append("")
    for day in draft.itinerary:
        lines.append(f"## 第{day.day}天 · {day.date.isoformat()}")
        for label in ("天气", "住宿", "交通"):
            lines.extend(f"- **{label}**：{note.removeprefix(label + '：')}" for note in day.notes if note.startswith(label + "："))
        labels = {"morning": "上午", "lunch": "中午", "afternoon": "下午", "evening": "晚上"}
        for slot in day.slots:
            lines.append(f"- **{labels[slot.period]}**：{slot.title}。{slot.description}")
        lines.append("")
    notices = user_facing_warnings(draft.warnings)
    if notices:
        lines.append("## 数据说明")
        lines.extend(f"- {notice}" for notice in notices)
    lines.append("\n当前为可编辑规划草稿，不包含下单、预订或支付服务。")
    return "\n".join(lines)
