"""行程的用户可见文案必须是中文。

app/agents/scheduling.py 的兜底文案（无候选时的时段标题/描述）
会被 app/services/planning.py:render_plan_markdown 直接插进中文正文里：

    - **上午**：Flexible local time。No second evidence-backed attraction was available.

模块内部的英文（docstring、字段名、warning code）不在此列——那些是给
开发者和程序看的。这里只钉住会出现在用户眼前的字符串。
"""

from datetime import date

from app.agents.scheduling import schedule_itinerary
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
    itinerary, _warnings = schedule_itinerary(_requirement(), [])

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

    itinerary, _warnings = schedule_itinerary(_requirement(days=1), results)

    assert itinerary[0].slots[0].title == "IFS Chengdu"
