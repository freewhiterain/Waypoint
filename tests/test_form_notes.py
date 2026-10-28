"""表单"补充要求"：解析进槽位和需求；目的地等三项以表单为准，解析失败不影响规划。"""

from datetime import date

import pytest

from app.schemas.slots import SlotUpdate, TripSlots
from app.services.slots import InMemorySlotRepository, load_slots, store_slots
from app.services.trip_form import apply_form_notes, build_requirement


class FakeLLM:
    def __init__(self, update):
        self.update = update

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        if isinstance(self.update, Exception):
            raise self.update
        return self.update


FORM = {"destination": "重庆", "departure_date": date(2026, 11, 3), "days": 2}


async def _repo():
    repository = InMemorySlotRepository()
    await store_slots(repository, "c1", TripSlots(**FORM))
    return repository


@pytest.mark.asyncio
async def test_notes_become_requirement_constraints_and_are_remembered():
    repository = await _repo()
    llm = FakeLLM(SlotUpdate(destination="成都", must_visit=["洪崖洞"], food_preferences=["不吃辣"]))

    arguments = await apply_form_notes(repository, "c1", "洪崖洞一定要去，全家不吃辣", {}, llm=llm)
    requirement = build_requirement(FORM, arguments)

    assert requirement.destination == "重庆"  # 表单说了算
    assert requirement.must_visit == ["洪崖洞"] and requirement.food_preferences == ["不吃辣"]
    assert (await load_slots(repository, "c1")).must_visit == ["洪崖洞"]


@pytest.mark.asyncio
async def test_places_the_notes_never_mention_are_dropped():
    repository = await _repo()

    arguments = await apply_form_notes(
        repository, "c1", "想去洪崖洞", {}, llm=FakeLLM(SlotUpdate(must_visit=["洪崖洞", "磁器口"]))
    )

    assert arguments["requirement_context"]["must_visit"] == ["洪崖洞"]


@pytest.mark.asyncio
async def test_empty_or_unparseable_notes_leave_arguments_alone():
    repository = await _repo()
    original = {"requirement_context": {"styles": ["美食"]}}

    assert await apply_form_notes(repository, "c1", "  ", original, llm=FakeLLM(SlotUpdate())) == original
    assert await apply_form_notes(repository, "c1", "随便", original, llm=FakeLLM(RuntimeError("boom"))) == original
