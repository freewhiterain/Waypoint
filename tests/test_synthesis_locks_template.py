"""成文阶段只改写描述：地点、时段、耗时等以算法排好的模板为准。"""

from datetime import date

import pytest

from app.agents import supervisor
from app.schemas.planning import ItineraryDay, TimeSlot, TravelRequirement


def _template() -> list[ItineraryDay]:
    return [
        ItineraryDay(
            day=1,
            date=date(2026, 11, 14),
            notes=["天气：晴"],
            slots=[
                TimeSlot(period="morning", title="武侯祠", description="模板描述", travel_minutes=5),
                TimeSlot(period="afternoon", title="锦里", description="模板描述", travel_minutes=8),
                TimeSlot(period="evening", title="火锅", description="模板描述"),
            ],
        )
    ]


class _FakeStructured:
    def __init__(self, days):
        self.days = days

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        return supervisor._LLMItinerary(days=self.days)


@pytest.fixture
def requirement():
    return TravelRequirement(destination="成都", departure_date=date(2026, 11, 14), days=1)


@pytest.mark.asyncio
async def test_only_descriptions_are_taken_from_the_model(monkeypatch, requirement):
    written = supervisor._DayText(
        day=1,
        slots=[
            supervisor._SlotText(period="morning", description="改写后的上午"),
            supervisor._SlotText(period="afternoon", description="改写后的下午"),
            supervisor._SlotText(period="evening", description="改写后的晚上"),
        ],
    )
    monkeypatch.setattr(supervisor.settings, "llm_api_key", "test-key")
    monkeypatch.setattr("app.core.llm.get_llm", lambda temperature=None: _FakeStructured([written]))

    days = await supervisor.synthesize_itinerary_with_llm(requirement, [], _template())

    morning = days[0].slots[0]
    assert (morning.title, morning.travel_minutes, morning.description) == ("武侯祠", 5, "改写后的上午")
    assert days[0].notes == ["天气：晴"]


@pytest.mark.asyncio
async def test_day_with_reshuffled_periods_falls_back_to_template(monkeypatch, requirement):
    written = supervisor._DayText(day=1, slots=[supervisor._SlotText(period="evening", description="只剩晚上")])
    monkeypatch.setattr(supervisor.settings, "llm_api_key", "test-key")
    monkeypatch.setattr("app.core.llm.get_llm", lambda temperature=None: _FakeStructured([written]))

    days = await supervisor.synthesize_itinerary_with_llm(requirement, [], _template())

    assert [slot.description for slot in days[0].slots] == ["模板描述"] * 3


@pytest.mark.asyncio
async def test_invalid_structured_output_is_retried_once(monkeypatch, requirement):
    written = supervisor._DayText(
        day=1,
        slots=[supervisor._SlotText(period=slot.period, description="重试后的描述") for slot in _template()[0].slots],
    )

    class Flaky(_FakeStructured):
        calls = 0

        async def ainvoke(self, messages):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise ValueError("invalid tool call json")
            return await super().ainvoke(messages)

    monkeypatch.setattr(supervisor.settings, "llm_api_key", "test-key")
    monkeypatch.setattr("app.core.llm.get_llm", lambda temperature=None: Flaky([written]))

    days = await supervisor.synthesize_itinerary_with_llm(requirement, [], _template())

    assert Flaky.calls == 2
    assert [slot.description for slot in days[0].slots] == ["重试后的描述"] * 3


@pytest.mark.asyncio
async def test_each_day_is_written_separately_and_a_failed_day_keeps_its_template(monkeypatch):
    first = _template()[0]
    second = first.model_copy(update={"day": 2, "date": date(2026, 11, 15)})
    requirement = TravelRequirement(destination="成都", departure_date=date(2026, 11, 14), days=2)

    class PerDay:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            prompt = messages[-1]["content"]
            if "第 2 天" in prompt:
                raise ValueError("bad json")
            return supervisor._LLMItinerary(days=[supervisor._DayText(day=1, slots=[
                supervisor._SlotText(period=slot.period, description="第一天改写") for slot in first.slots])])

    monkeypatch.setattr(supervisor.settings, "llm_api_key", "test-key")
    monkeypatch.setattr("app.core.llm.get_llm", lambda temperature=None: PerDay())

    days = await supervisor.synthesize_itinerary_with_llm(requirement, [], [first, second])

    assert [slot.description for slot in days[0].slots] == ["第一天改写"] * 3
    assert [slot.description for slot in days[1].slots] == ["模板描述"] * 3
