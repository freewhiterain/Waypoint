"""一次说了几个城市：一份行程只排第一个，并告诉用户后面的城市排完再接着排。"""

from app.services.chat_flow import later_cities


def test_cities_mentioned_after_a_next_cue_are_later_stops():
    assert later_cities("先去成都玩三天再去重庆玩两天，帮我规划", "成都") == ["重庆"]
    assert later_cities("成都玩两天，然后去重庆、最后到西安", "成都") == ["重庆", "西安"]


def test_a_single_city_or_a_comparison_is_not_multi_city():
    assert later_cities("帮我规划成都三天", "成都") == []
    assert later_cities("成都和重庆哪个好玩", "成都") == []
    assert later_cities("从上海出发去南京玩两天", "南京") == []
