"""一次说了几个城市：一份行程只排第一个，并告诉用户后面的城市排完再接着排。"""

from app.services.chat_flow import later_cities


def test_cities_mentioned_after_a_next_cue_are_later_stops():
    assert later_cities("先去成都玩三天再去重庆玩两天，帮我规划", "成都") == ["重庆"]
    assert later_cities("成都玩两天，然后去重庆、最后到西安", "成都") == ["重庆", "西安"]


def test_a_single_city_or_a_comparison_is_not_multi_city():
    assert later_cities("帮我规划成都三天", "成都") == []
    assert later_cities("成都和重庆哪个好玩", "成都") == []
    assert later_cities("从上海出发去南京玩两天", "南京") == []


def test_leg_days_are_read_for_the_later_city():
    from app.services.chat_flow import _leg_days

    assert _leg_days("先去成都玩三天再去重庆玩两天", "重庆") == 2
    assert _leg_days("成都3天，然后去重庆", "重庆") is None


def test_planning_the_next_leg_starts_the_day_after_the_first_ends():
    from datetime import date

    from app.schemas.slots import NextLeg, SlotUpdate, TripSlots
    from app.services.slots import apply_slot_update

    first = TripSlots(destination="成都", departure_date=date(2026, 11, 7), days=3,
                      next_legs=[NextLeg(city="重庆", days=2)])

    slots, _ = apply_slot_update(first, SlotUpdate(destination="重庆"), user_text="再规划重庆", today=date(2026, 10, 31))

    assert (slots.destination, slots.departure_date, slots.days) == ("重庆", date(2026, 11, 10), 2)
    assert slots.next_legs == []
