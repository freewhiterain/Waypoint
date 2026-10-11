from datetime import date

import pytest
from pydantic import ValidationError

from app.agents.planner import create_research_plan, parallel_groups
from app.schemas.planning import ResearchTask, TravelRequirement


def make_requirement(**overrides):
    data = {
        "origin": "Shanghai",
        "destination": "Chengdu",
        "departure_date": date(2026, 8, 1),
        "days": 5,
        "adults": 2,
        "styles": ["culture", "food"],
    }
    data.update(overrides)
    return TravelRequirement(**data)


def test_requirement_rejects_invalid_trip():
    with pytest.raises(ValidationError):
        make_requirement(destination="Shanghai")
    with pytest.raises(ValidationError):
        make_requirement(days=0)


def test_planner_runs_transport_after_the_places_it_connects():
    tasks = create_research_plan(make_requirement())

    assert {task.task_type for task in tasks} == {
        "attractions",
        "transport",
        "hotel",
        "food",
        "weather",
    }
    assert len({task.id for task in tasks}) == 5
    by_type = {task.task_type: task for task in tasks}
    # 交通查的是市内地点之间的耗时，依赖景点、住宿、美食的候选。
    assert set(by_type["transport"].dependencies) == {
        by_type["attractions"].id,
        by_type["hotel"].id,
        by_type["food"].id,
    }

    groups = parallel_groups(tasks)
    assert [{task.task_type for task in group} for group in groups] == [
        {"attractions", "hotel", "food", "weather"},
        {"transport"},
    ]


def test_parallel_groups_respects_dependencies():
    first = ResearchTask(task_type="attractions", query="attractions")
    second = ResearchTask(task_type="hotel", query="hotel", dependencies=[first.id])

    groups = parallel_groups([second, first])

    assert [[task.id for task in group] for group in groups] == [[first.id], [second.id]]


def test_parallel_groups_rejects_cycles():
    first = ResearchTask(id="a", task_type="attractions", query="a", dependencies=["b"])
    second = ResearchTask(id="b", task_type="hotel", query="b", dependencies=["a"])

    with pytest.raises(ValueError, match="循环"):
        parallel_groups([first, second])
