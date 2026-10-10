from datetime import date
from pathlib import Path

import pytest

from app.main import app
from app.schemas.events import SSEEvent
from app.schemas.planning import ItineraryDay, TimeSlot, TravelPlanDraft, TravelRequirement
from app.services.planning import RequirementExtractor, render_plan_markdown
from app.utils.logger import redact_text


def test_stage4_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert {
        "/api/v1/tasks",
        "/api/v1/tasks/{task_id}",
        "/api/v1/tasks/{task_id}/events",
        "/api/v1/preferences/proposals",
        "/api/v1/itineraries/{conversation_id}/save",
        "/api/v1/itineraries/{conversation_id}",
        "/api/v1/approvals/{approval_id}/decision",
    }.issubset(paths)


def test_sse_event_preserves_legacy_token_and_error_fields():
    token = SSEEvent(type="token", task_id="t", sequence=1, payload={"content": "成都"}).legacy_payload()
    error = SSEEvent(type="error", task_id="t", sequence=2, payload={"message": "失败", "code": "x"}).legacy_payload()

    assert token["content"] == "成都"
    assert error["message"] == "失败"
    assert token["payload"] == {"content": "成都"}


@pytest.mark.asyncio
async def test_rule_extractor_handles_common_complete_request(monkeypatch):
    monkeypatch.setattr("app.services.planning.settings.llm_api_key", "")
    draft = await RequirementExtractor().extract(
        "从上海出发，2026年8月1日去成都五日游，喜欢文化和美食"
    )

    requirement = draft.to_requirement()
    assert requirement.origin == "上海"
    assert requirement.destination == "成都"
    assert requirement.departure_date == date(2026, 8, 1)
    assert requirement.days == 5
    assert requirement.styles == ["文化", "美食"]


def test_rendered_plan_is_legacy_readable_and_transaction_free():
    requirement = TravelRequirement(
        origin="上海", destination="成都", departure_date=date(2026, 8, 1), days=1
    )
    draft = TravelPlanDraft(
        requirement=requirement,
        itinerary=[
            ItineraryDay(
                day=1,
                date=requirement.departure_date,
                slots=[
                    TimeSlot(period="morning", title="文化活动"),
                    TimeSlot(period="afternoon", title="休闲活动"),
                    TimeSlot(period="evening", title="本地餐饮"),
                ],
            )
        ],
        worker_results=[],
        evidence=[],
    )

    rendered = render_plan_markdown(draft)
    assert "第1天" in rendered
    assert "当前为可编辑规划草稿" in rendered
    assert "支付链接" not in rendered


def test_log_redaction_masks_common_credentials():
    value = redact_text("api_key=abc123 password: hello token=jwt-value")
    assert "abc123" not in value
    assert "hello" not in value
    assert "jwt-value" not in value
    assert value.count("***") == 3


def test_frontend_buffers_partial_sse_frames():
    html_path = Path("1_zhixing.html")
    if not html_path.exists():
        pytest.skip("前端演示页面未包含在当前后端部署中")
    html = html_path.read_text(encoding="utf-8")
    assert "sseBuffer" in html
    assert "split(/\\r?\\n\\r?\\n/)" in html
    assert 'parsed.type === "token"' in html
