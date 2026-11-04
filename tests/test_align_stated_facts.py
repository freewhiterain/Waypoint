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


def test_travel_time_follows_the_leg_and_its_mode():
    walk = TimeSlot(period="evening", title="听香", travel_minutes=8, travel_mode="walk")
    drive = TimeSlot(period="afternoon", title="武侯祠", travel_minutes=21, travel_mode="drive")

    assert align_stated_facts("从草堂步行约 34 分钟车程即到，推荐冷锅鱼。", walk) == "从草堂步行约 8 分钟即到，推荐冷锅鱼。"
    assert align_stated_facts("打车约 15 分钟，先看三义庙。", drive) == "约 21 分钟车程，先看三义庙。"


def test_only_the_first_travel_mention_is_this_leg():
    slot = TimeSlot(period="evening", title="火锅", travel_minutes=12, travel_mode="drive")
    text = "约 20 分钟车程到店；饭后步行约 6 分钟回酒店。"

    assert align_stated_facts(text, slot) == "约 12 分钟车程到店；饭后步行约 6 分钟回酒店。"


def test_travel_left_alone_without_route_data():
    slot = TimeSlot(period="morning", title="宽窄巷子")
    assert align_stated_facts("约 20 分钟车程，游览约 90 分钟。", slot) == "约 20 分钟车程，游览约 90 分钟。"
