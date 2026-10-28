"""一个具体地点都没查到时，不渲染每天"自由活动"的空行程，而是说明原因和出路。"""

from datetime import date

from app.schemas.planning import ItineraryDay, TimeSlot, TravelPlanDraft, TravelRequirement
from app.services.plan_render import has_concrete_places, render_plan_markdown


def draft(*titles):
    requirement = TravelRequirement(destination="西安", departure_date=date(2026, 11, 3), days=1)
    slots = [TimeSlot(period=period, title=title) for period, title in zip(("morning", "afternoon", "evening"), titles)]
    day = ItineraryDay(day=1, date=date(2026, 11, 3), slots=slots, notes=["天气：零星小雨 8~13℃"])
    return TravelPlanDraft(requirement=requirement, itinerary=[day], worker_results=[], evidence=[])


def test_a_plan_of_only_free_time_explains_instead_of_pretending():
    empty = draft("西安分区自由活动", "自由活动", "自由安排晚餐")

    text = render_plan_markdown(empty)

    assert not has_concrete_places(empty)
    assert "没能排出来" in text and "自由活动" in text  # 只在解释里提到
    assert "## 第1天" not in text
    assert "杭州" in text and "零星小雨" in text


def test_a_plan_with_one_real_place_renders_normally():
    partial = draft("大雁塔", "自由活动", "自由安排晚餐")

    assert has_concrete_places(partial)
    assert "## 第1天" in render_plan_markdown(partial)
