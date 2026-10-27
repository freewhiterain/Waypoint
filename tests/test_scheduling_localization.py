"""行程的用户可见文案必须是中文。

app/agents/itinerary_planner.py 的兜底文案（无候选时的时段标题/描述）
会被 app/services/plan_render.py:render_plan_markdown 直接插进中文正文里：

    - **上午**：Flexible local time。No second evidence-backed attraction was available.

模块内部的英文（docstring、字段名、warning code）不在此列——那些是给
开发者和程序看的。这里只钉住会出现在用户眼前的字符串。
"""

from datetime import date

from app.agents.itinerary_planner import schedule_by_rules
from app.schemas.planning import CandidateOption, Evidence, TravelRequirement, WorkerResult


def _requirement(*, days: int = 2) -> TravelRequirement:
    return TravelRequirement(
        destination="成都",
        departure_date=date(2026, 8, 1),
        days=days,
    )


def _has_latin_words(text: str) -> bool:
    """连续两个及以上的英文字母算作英文单词，避开 09:00-17:00 这类记号。"""
    import re

    return bool(re.search(r"[A-Za-z]{2,}", text))


def test_empty_plan_fallback_slots_are_chinese():
    itinerary, _warnings = schedule_by_rules(_requirement(), [])

    assert itinerary, "没有任何候选时也应产出逐日骨架"
    for day in itinerary:
        for note in day.notes:
            assert not _has_latin_words(note), f"逐日备注渗出英文: {note}"
        for slot in day.slots:
            assert not _has_latin_words(slot.title), f"时段标题渗出英文: {slot.title}"
            assert not _has_latin_words(slot.description), f"时段描述渗出英文: {slot.description}"



def test_candidate_titles_still_pass_through_untouched():
    """外文候选名来自证据，不该被这条规则波及——只有兜底文案受约束。"""
    option = CandidateOption(
        name="IFS Chengdu",
        category="attractions",
        estimated_cost=100.0,
        attributes={"location": "锦江"},
        evidence_ids=["attractions-evidence"],
    )
    results = [
        WorkerResult(
            task_id="attractions-task",
            worker="attractions",
            status="completed",
            summary="grounded",
            options=[option],
            evidence=[Evidence(id="attractions-evidence", content="有证据", source="official")],
        )
    ]

    itinerary, _warnings = schedule_by_rules(_requirement(days=1), results)

    assert itinerary[0].slots[0].title == "IFS Chengdu"


def test_internal_warnings_are_translated_or_hidden_from_the_user():
    from app.services.plan_render import user_facing_warnings

    shown = user_facing_warnings([
        "Deduplicated evidence a into b for task t.",
        "Dropped an unbound claim from subagent output.",
        "weather_mcp is unavailable for weather.",
        "travel_time_estimated:no_route_data",
        "travel_time_estimated:no_route_data",
        "No provider returned evidence for food.",
        "itinerary_constraint:closed:第1天上午武侯祠尚未开放",
        "证据未提供降水、风力、空气质量等信息。",
    ])

    assert shown == [
        "部分地点之间没有查到真实交通耗时，顺序按所在区域粗略安排。",
        "没有查到美食的资料。",
        "第1天上午武侯祠尚未开放",
        "证据未提供降水、风力、空气质量等信息。",
    ]


def test_model_notes_mentioning_internal_field_names_are_hidden():
    from app.services.plan_render import user_facing_warnings

    shown = user_facing_warnings(["证据中未提供开放时间，故未填写 opening_hours 或 estimated_cost。", "证据来源多为旅游介绍页面，信息时效性有限。"])

    assert shown == ["证据来源多为旅游介绍页面，信息时效性有限。"]


def test_known_notices_come_first_and_model_notes_are_capped():
    from app.services.plan_render import user_facing_warnings

    notes = [f"证据未提供第{index}项的营业时间信息。" for index in range(6)]
    shown = user_facing_warnings([*notes, "planning_degraded:provider_degraded"])

    assert shown[0] == "部分信息来源不可用，行程基于现有资料生成。"
    assert shown[1:] == notes[:3]
