"""模型改写的描述里，价格和营业时间以字段为准；括号里顺带提到的别处不改。"""

from app.agents.supervisor import align_stated_facts
from app.schemas.planning import TimeSlot


def test_misstated_price_and_hours_are_put_back_to_the_fields():
    slot = TimeSlot(period="evening", title="南京大牌档", estimated_cost=70, opening_window="11:00-21:30")

    text = align_stated_facts("本地菜人均约 74 元，营业时间 10:30-22:00，推荐盐水鸭。", slot)

    assert text == "本地菜人均约 70 元，营业时间 11:00-21:30，推荐盐水鸭。"


def test_numbers_about_a_nearby_place_in_parentheses_are_kept():
    slot = TimeSlot(period="morning", title="洪崖洞", estimated_cost=0, opening_window="09:00-23:00")
    text = "开放时间 09:00-23:00，免费；可顺路逛重逢1980街区（11:00-24:00，门票 35 元，需另购）。"

    assert align_stated_facts(text, slot) == text
