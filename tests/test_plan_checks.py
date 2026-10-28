"""结构化行程的可判定约束：环境层（对得上地点库）和逻辑层（天数、去重、营业时间等）。"""

from datetime import date

from app.evaluation.plan_checks import check_plan
from app.places.catalog import PlaceRecord
from app.schemas.planning import ItineraryDay, TimeSlot, TravelRequirement

START = date(2026, 11, 2)


class Catalog:
    def __init__(self, *records):
        self.records = {record.name: record for record in records}

    def cities(self):
        return ["成都"]

    def find(self, city, name):
        return self.records.get(name)


WUHOU = PlaceRecord(city="成都", category="attractions", rank=1, name="武侯祠", open_time="08:30", close_time="19:00", price=50)
HOTPOT = PlaceRecord(city="成都", category="food", rank=1, name="蜀大侠火锅", open_time="11:00", close_time="22:00", price=120)


def day(number, *slots):
    return ItineraryDay(day=number, date=date(2026, 11, number + 1), slots=list(slots))


def slot(period, title, window=None, cost=None):
    return TimeSlot(period=period, title=title, opening_window=window, estimated_cost=cost)


def test_a_clean_plan_has_no_issues():
    requirement = TravelRequirement(destination="成都", departure_date=START, days=1)
    plan = [day(1, slot("morning", "武侯祠", "08:30-19:00", 50), slot("evening", "蜀大侠火锅", "11:00-22:00", 120))]

    assert check_plan(requirement, plan, Catalog(WUHOU, HOTPOT)) == []


def test_logic_issues_cover_days_repeats_avoided_relaxed_and_hours():
    requirement = TravelRequirement(
        destination="成都", departure_date=START, days=3, avoid_places=["武侯祠"], relaxed_days=[2]
    )
    plan = [
        day(1, slot("morning", "武侯祠"), slot("evening", "蜀大侠火锅")),
        day(2, slot("morning", "锦里"), slot("afternoon", "武侯祠"), slot("night", "九眼桥", "09:00-18:00")),
    ]

    messages = [issue.message for issue in check_plan(requirement, plan, Catalog())]

    assert any("天数 2" in m for m in messages)
    assert any("第1天已经去过" in m for m in messages)
    assert sum("不去「武侯祠」" in m for m in messages) == 2
    assert any("要求轻松" in m for m in messages)
    assert any("第2天没有晚饭" in m for m in messages)
    assert any("九眼桥" in m and "时段对不上" in m for m in messages)


def test_environment_issues_flag_unknown_places_and_made_up_prices():
    requirement = TravelRequirement(destination="成都", departure_date=START, days=1)
    plan = [day(1, slot("morning", "武侯祠", "08:30-19:00", 80), slot("evening", "不存在的饭馆"),
                slot("afternoon", "自由活动"))]

    issues = check_plan(requirement, plan, Catalog(WUHOU))

    assert {issue.category for issue in issues} == {"environment"}
    assert any("80" in issue.message for issue in issues)
    assert any("不存在的饭馆" in issue.message for issue in issues)


def test_cities_outside_the_catalog_skip_environment_checks():
    requirement = TravelRequirement(destination="大理", departure_date=START, days=1)
    plan = [day(1, slot("morning", "洱海"), slot("evening", "某饭馆"))]

    assert check_plan(requirement, plan, Catalog()) == []
