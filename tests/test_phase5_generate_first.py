"""阶段一改造:目的地即规划、默认值假设、提取器增强。"""

from datetime import date, timedelta

import pytest

from app.schemas.planning import ResearchTask, TravelRequirement, TravelRequirementDraft
from app.services.requirements import RequirementExtractor


TODAY = date(2026, 7, 19)


async def _extract(text: str, monkeypatch) -> TravelRequirementDraft:
    monkeypatch.setattr("app.services.requirements.settings.llm_api_key", "")
    return await RequirementExtractor().extract(text, today=TODAY)


async def test_rule_extractor_handles_label_style_phrasing(monkeypatch):
    draft = await _extract("出发地上海，目的地哈尔滨，明天出发，旅游30天", monkeypatch)

    assert draft.origin == "上海"
    assert draft.destination == "哈尔滨"
    assert draft.departure_date == TODAY + timedelta(days=1)
    assert draft.days == 30


async def test_rule_extractor_resolves_relative_dates(monkeypatch):
    draft = await _extract("后天从北京出发去成都玩五天", monkeypatch)

    assert draft.departure_date == TODAY + timedelta(days=2)
    assert draft.origin == "北京"
    assert draft.destination == "成都"
    assert draft.days == 5


def test_draft_exposes_only_strict_requirement_conversion():
    assert not hasattr(TravelRequirementDraft, "to_requirement_with_defaults")
    with pytest.raises(ValueError, match="旅行需求缺少"):
        TravelRequirementDraft(destination="大理").to_requirement()


def test_merge_prefers_rule_hits_and_fills_gaps():
    rules = TravelRequirementDraft(destination="哈尔滨", days=30)
    llm = TravelRequirementDraft(origin="上海", destination="错误目的地", days=5, styles=["美食"])

    merged = RequirementExtractor._merge(rules, llm)

    assert merged.destination == "哈尔滨"
    assert merged.days == 30
    assert merged.origin == "上海"
    assert merged.styles == ["美食"]



async def test_supervisor_draft_uses_confirmed_requirements(monkeypatch):
    from app.agents.supervisor import run_travel_planning
    from app.services.plan_render import render_plan_markdown

    requirement = TravelRequirement(destination="大理", departure_date=TODAY, days=3)
    draft = await run_travel_planning(requirement)

    rendered = render_plan_markdown(draft)
    assert "假设说明" not in rendered
