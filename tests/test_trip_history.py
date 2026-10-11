from datetime import date

import pytest

from app.memory.trip_history import (
    InMemoryTripHistoryRepository,
    build_trip_history_record,
    record_trip_history_from_itinerary,
)


def _content(**overrides):
    base = {
        "requirement": {"destination": "成都", "departure_date": "2026-08-01", "days": 3},
        "itinerary": [
            {
                "day": 1,
                "date": "2026-08-01",
                "slots": [
                    {"period": "morning", "title": "熊猫基地", "description": ""},
                    {"period": "evening", "title": "锦里", "description": ""},
                ],
            },
            {
                "day": 2,
                "date": "2026-08-02",
                "slots": [
                    {"period": "morning", "title": "宽窄巷子", "description": ""},
                ],
            },
        ],
    }
    base.update(overrides)
    return base


def test_build_trip_history_record_extracts_destination_dates_and_attractions():
    record = build_trip_history_record("u1", "itin-1", _content())

    assert record is not None
    assert record.destination == "成都"
    assert record.start_date == date(2026, 8, 1)
    assert record.end_date == date(2026, 8, 3)
    assert record.visited_attractions == ["熊猫基地", "宽窄巷子"]
    assert record.source_itinerary_id == "itin-1"


def test_build_trip_history_record_returns_none_when_requirement_missing():
    assert build_trip_history_record("u1", "itin-1", {"itinerary": []}) is None


def test_build_trip_history_record_returns_none_when_destination_missing():
    content = _content(requirement={"departure_date": "2026-08-01", "days": 3})
    assert build_trip_history_record("u1", "itin-1", content) is None


def test_build_trip_history_record_tolerates_missing_itinerary_section():
    content = {"requirement": {"destination": "西安", "departure_date": "2026-09-01", "days": 2}}

    record = build_trip_history_record("u1", "itin-2", content)

    assert record is not None
    assert record.visited_attractions == []


@pytest.mark.asyncio
async def test_record_trip_history_from_itinerary_appends_to_repository():
    repository = InMemoryTripHistoryRepository()

    result = await record_trip_history_from_itinerary("u1", "itin-1", _content(), repository)

    assert result is not None
    stored = await repository.list("u1")
    assert len(stored) == 1
    assert stored[0].destination == "成都"


@pytest.mark.asyncio
async def test_record_trip_history_from_itinerary_degrades_to_none_on_malformed_content():
    repository = InMemoryTripHistoryRepository()

    result = await record_trip_history_from_itinerary("u1", "itin-1", {"not": "valid"}, repository)

    assert result is None
    assert await repository.list("u1") == []


def test_afternoon_attractions_are_recorded_and_placeholders_are_not():
    content = _content(itinerary=[
        {"day": 1, "date": "2026-08-01", "slots": [
            {"period": "morning", "title": "熊猫基地"},
            {"period": "afternoon", "title": "宽窄巷子"},
            {"period": "evening", "title": "火锅店"},
        ]},
        {"day": 2, "date": "2026-08-02", "slots": [
            {"period": "morning", "title": "成都分区自由活动"},
            {"period": "afternoon", "title": "自由活动"},
            {"period": "evening", "title": "自由安排晚餐"},
        ]},
    ])

    assert build_trip_history_record("u1", "itin-1", content).visited_attractions == ["熊猫基地", "宽窄巷子"]


def test_attraction_candidates_decide_what_counts_when_titles_were_reworded():
    content = _content(
        itinerary=[{"day": 1, "date": "2026-08-01", "slots": [
            {"period": "morning", "title": "漫步武侯祠"},
            {"period": "afternoon", "title": "酒店休息"},
        ]}],
        worker_results=[{"worker": "attractions", "options": [{"name": "武侯祠"}, {"name": "锦里"}]}],
    )

    assert build_trip_history_record("u1", "itin-1", content).visited_attractions == ["武侯祠"]
