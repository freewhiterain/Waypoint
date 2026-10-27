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
async def test_model_cannot_swap_places_or_change_travel_minutes(monkeypatch, requirement):
    written = _template()[0].model_copy(
        update={
            "notes": [],
            "slots": [
                TimeSlot(period="morning", title="宽窄巷子", description="改写后的上午", travel_minutes=40),
                TimeSlot(period="afternoon", title="锦里", description="改写后的下午", travel_minutes=8),
                TimeSlot(period="evening", title="火锅", description="改写后的晚上"),
            ],
        }
    )
    monkeypatch.setattr(supervisor.settings, "llm_api_key", "test-key")
    monkeypatch.setattr("app.agents.llm.get_llm", lambda temperature=None: _FakeStructured([written]))

    days = await supervisor.synthesize_itinerary_with_llm(requirement, [], _template())

    morning = days[0].slots[0]
    assert (morning.title, morning.travel_minutes, morning.description) == ("武侯祠", 5, "改写后的上午")
    assert days[0].notes == ["天气：晴"]


@pytest.mark.asyncio
async def test_day_with_reshuffled_periods_falls_back_to_template(monkeypatch, requirement):
    written = _template()[0].model_copy(
        update={"slots": [TimeSlot(period="evening", title="火锅", description="只剩晚上")]}
    )
    monkeypatch.setattr(supervisor.settings, "llm_api_key", "test-key")
    monkeypatch.setattr("app.agents.llm.get_llm", lambda temperature=None: _FakeStructured([written]))

    days = await supervisor.synthesize_itinerary_with_llm(requirement, [], _template())

    assert [slot.description for slot in days[0].slots] == ["模板描述"] * 3
