from datetime import date

import pytest

from app.memory.defaults import apply_preference_defaults, resolve_preference_defaults
from app.memory.service import InMemoryPreferenceRepository
from app.schemas.governance import PreferenceRecord
from app.schemas.planning import TravelRequirement


def _requirement(**overrides):
    base = dict(destination="成都", departure_date=date(2026, 8, 1), days=3)
    base.update(overrides)
    return TravelRequirement(**base)


@pytest.mark.asyncio
async def test_resolve_preference_defaults_picks_latest_value_per_key():
    repo = InMemoryPreferenceRepository()
    await repo.upsert(PreferenceRecord(user_id="u1", key="food_preferences", value=["清淡"]))
    await repo.upsert(PreferenceRecord(user_id="u1", key="food_preferences", value=["清淡", "不吃辣"]))

    defaults = await resolve_preference_defaults("u1", repo)

    assert defaults["food_preferences"] == ["清淡", "不吃辣"]


@pytest.mark.asyncio
async def test_resolve_preference_defaults_ignores_keys_outside_vocabulary():
    repo = InMemoryPreferenceRepository()
    await repo.upsert(PreferenceRecord(user_id="u1", key="favorite_color", value="blue"))

    defaults = await resolve_preference_defaults("u1", repo)

    assert defaults == {}


@pytest.mark.asyncio
async def test_resolve_preference_defaults_ignores_type_mismatched_values():
    repo = InMemoryPreferenceRepository()
    await repo.upsert(PreferenceRecord(user_id="u1", key="food_preferences", value="清淡"))

    defaults = await resolve_preference_defaults("u1", repo)

    assert defaults == {}



def test_apply_preference_defaults_fills_only_empty_fields():
    requirement = _requirement(food_preferences=["微辣"])
    defaults = {
        "food_preferences": ["清淡", "不吃辣"],
        "accommodation_preferences": ["经济型"],
    }

    result = apply_preference_defaults(requirement, defaults)

    assert result.food_preferences == ["微辣"]
    assert result.accommodation_preferences == ["经济型"]



def test_apply_preference_defaults_returns_equivalent_requirement_when_no_defaults_apply():
    requirement = _requirement()

    result = apply_preference_defaults(requirement, {})

    assert result == requirement
