"""表单只问目的地/日期/天数，人数、预算、偏好必须从原话一路带进 TravelRequirement。

回归：用户说"两个人，预算5000元，喜欢美食和熊猫"，规划拿到的却是
adults=1、budget=None、没有偏好——表单提交时只用了表单的三个字段。
"""

from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.v1.tools import build_requirement
from app.schemas.planning import TravelRequirement
from app.services.main_agent import MainAgentService
from app.services.planning import RequirementExtractor

from tests.test_trip_form_tool_flow import (
    FakeDraft,
    InMemoryInvocationRepository,
    configure_endpoint,
    endpoint_client,
    invocation,
)


MESSAGE = "我想下个月去成都玩三天，两个人，预算5000元，喜欢美食和熊猫，帮我规划一下行程"


@pytest.mark.parametrize(
    ("text", "adults", "children"),
    [
        ("两个人去成都", 2, 0),
        ("我们3人出行", 3, 0),
        ("两个大人一个孩子", 2, 1),
        ("三个人，其中一个孩子", 2, 1),
        ("一家三口去玩", 2, 1),
        ("去成都玩三天", 1, 0),
    ],
)
def test_party_size_is_extracted_only_when_stated(text, adults, children):
    draft = RequirementExtractor._extract_rules(text, date(2026, 10, 7))

    assert (draft.adults, draft.children) == (adults, children)


def test_interests_and_food_preferences_are_extracted():
    draft = RequirementExtractor._extract_rules(MESSAGE + "，不吃辣", date(2026, 10, 7))

    assert draft.budget == 5000
    assert "美食" in draft.styles and "熊猫" in draft.styles
    assert draft.food_preferences == ["不吃辣"]


@pytest.mark.asyncio
async def test_planning_request_keeps_form_prefill_unchanged_and_adds_context():
    decision = await MainAgentService(use_llm=False).decide(MESSAGE, [])

    assert decision.action == "collect_trip_requirements"
    # 前端按 initial_values 回填表单，这里不能混进表单没有的字段。
    assert decision.initial_values == {"destination": "成都", "days": 3}
    assert decision.requirement_context["adults"] == 2
    assert decision.requirement_context["budget"] == 5000
    assert {"美食", "熊猫"} <= set(decision.requirement_context["styles"])


@pytest.mark.asyncio
async def test_request_without_extra_details_has_empty_context():
    decision = await MainAgentService(use_llm=False).decide("帮我规划一次成都旅行", [])

    assert decision.requirement_context == {}


def test_form_values_override_context_and_bad_context_falls_back_to_form():
    form = {"destination": "成都", "departure_date": date(2026, 11, 14), "days": 3}

    merged = build_requirement(form, {"requirement_context": {"adults": 2, "budget": 5000, "days": 9}})
    assert merged.adults == 2 and merged.budget == 5000 and merged.days == 3

    conflicting = build_requirement(form, {"requirement_context": {"origin": "成都", "adults": 2}})
    assert conflicting.origin is None and conflicting.adults == 1

    assert build_requirement(form, None).adults == 1


@pytest.mark.asyncio
async def test_form_submission_passes_context_into_planning(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    record = invocation(user_id=str(user.id))
    record.arguments = {
        "initial_values": {"destination": "成都", "days": 3},
        "requirement_context": {"adults": 2, "budget": 5000, "styles": ["美食", "熊猫"]},
    }
    repository = InMemoryInvocationRepository([record])
    configure_endpoint(monkeypatch, repository)
    from app.api.v1 import tools

    calls = []

    async def supervisor(requirement, **kwargs):
        calls.append(requirement)
        return FakeDraft(requirement.destination)

    monkeypatch.setattr(tools, "run_travel_planning", supervisor, raising=False)

    async with endpoint_client(user) as client:
        await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={
                "tool": "collect_trip_requirements",
                "status": "completed",
                "result": {"destination": "成都", "departure_date": "2026-11-14", "days": 3},
            },
        )

    assert len(calls) == 1
    requirement = calls[0]
    assert isinstance(requirement, TravelRequirement)
    assert (requirement.adults, requirement.budget) == (2, 5000)
    assert requirement.styles == ["美食", "熊猫"]
