"""前端（web/，React）与后端接口的契约：只做源码层面的静态检查，不需要 Node。"""

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web" / "src"


def source(*parts: str) -> str:
    return (WEB.joinpath(*parts)).read_text(encoding="utf-8")


def test_trip_form_tool_call_and_submission_contract_is_present():
    store = source("store", "chat.ts")

    assert 'event.tool === "collect_trip_requirements"' in store
    assert "/api/v1/chat/tools/${trip.callId}/result" in store
    assert 'status: "completed"' in store
    assert 'status: "recommend_destination"' in store
    for field in ("destination", "departure_date", "days: Number(values.days)"):
        assert field in store


def test_trip_form_only_asks_missing_slots_and_validates_them():
    store = source("store", "chat.ts")
    card = source("components", "TripCard.tsx")

    assert "askFields: TRIP_FIELDS.filter((field) => tripFieldError(field, values))" in store
    assert "出发日期不能早于今天" in store
    assert "天数请填 1-30 的整数" in store
    assert "还没想好，帮我推荐" in card


def test_history_restores_tool_state_and_plan():
    store = source("store", "chat.ts")

    assert 'toolCall?.tool === "collect_trip_requirements"' in store
    assert 'result.status === "completed"' in store
    assert 'result.status === "awaiting_destination"' in store
    assert "extra.assistant_result?.itinerary" in store


def test_composer_ignores_enter_while_ime_is_composing():
    # 中文输入法选词时按回车只是确认候选，原先的页面会直接把半截拼音发出去。
    chat = source("components", "ChatView.tsx")

    assert "e.nativeEvent.isComposing" in chat
    assert 'e.key === "Enter" && !e.shiftKey' in chat


def test_sse_reader_buffers_partial_frames():
    client = source("api", "client.ts")

    assert 'buffer.indexOf("\\n\\n")' in client
    assert "decoder.decode(value, { stream: true })" in client


def test_frontend_renders_research_events_and_react_tools():
    store = source("store", "chat.ts")
    item = source("components", "MessageItem.tsx")

    assert '"subagent_started"' in store and '"subagent_completed"' in store
    assert "evidence_count" in store
    for tool in ("search_travel_knowledge", "research_topic", "estimate_travel_time"):
        assert tool in store
    assert "ResearchProgress" in item


def test_fastapi_exposes_the_frontend_entrypoint():
    from app.main import app

    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/ui" in paths
