from pathlib import Path


HTML = Path("1_zhixing.html")


def read_page() -> str:
    return HTML.read_text(encoding="utf-8")


def test_trip_form_tool_call_and_submission_contract_is_present():
    html = read_page()

    assert 'parsed.type === "tool_call"' in html
    assert "renderTripTool" in html
    assert "submitTripToolResult" in html
    assert "/api/v1/chat/tools/${callId}/result" in html
    assert 'status: "recommend_destination"' in html
    assert 'status: "completed"' in html
    assert "parseSseStream" in html
    submit_start = html.index("function submitTripToolResult")
    submit_end = html.index("function restorePendingTool", submit_start)
    assert "sendMessage();" not in html[submit_start:submit_end]


def test_trip_form_has_three_steps_and_required_controls():
    html = read_page()

    assert "currentStep" in html
    assert "${page + 1}/${total}" in html
    assert 'name="destination"' in html
    assert 'name="departure_date"' in html
    assert 'name="days"' in html
    assert 'type="date"' in html
    assert 'min="${today}"' in html
    assert "[2, 3, 5, 7]" in html
    assert "min=\"1\"" in html
    assert "max=\"30\"" in html
    assert "closeTripTool" in html
    assert "submitting" in html


def test_trip_form_is_a_numbered_question_card():
    html = read_page()

    # 每题一页：编号选项 + "其他"自由输入，点选项即作答，最后一题由发送按钮提交。
    assert "chooseTripOption" in html
    assert "advanceTripTool" in html
    assert "trip-tool-index" in html
    assert "trip-tool-other" in html
    assert 'data-trip-action="back"' in html
    assert 'data-trip-action="forward"' in html
    assert "tripDateOptions" in html
    assert '"__recommend__"' in html
    # 填槽：只为缺失的槽位出题，卡片停靠在输入框位置。
    assert "missingTripSteps" in html
    assert "askSteps" in html
    assert 'id = "tripDock"' in html
    assert ".chat-input-area.trip-active .chat-input-wrapper" in html
    # 三个槽位都齐时不自动提交：先显示"将为你规划：……"，用户确认后才开始规划。
    receive_start = html.index("function receiveTripToolCall")
    receive_end = html.index("function localIso", receive_start)
    assert "submitTripToolResult" not in html[receive_start:receive_end]
    assert "confirming: askSteps.length === 0" in html
    confirm_start = html.index("function renderTripConfirm")
    confirm_end = html.index("function editTripTool", confirm_start)
    assert "将为你规划：" in html[confirm_start:confirm_end]
    assert 'data-trip-action="confirm"' in html[confirm_start:confirm_end]
    assert 'data-trip-action="edit"' in html[confirm_start:confirm_end]
    assert 'submitTripToolResult(callId, "completed")' in html[confirm_start:confirm_end]
    recommend_start = html.index("function chooseTripOption")
    recommend_end = html.index("function advanceTripTool", recommend_start)
    assert 'submitTripToolResult(callId, "recommend_destination")' in html[recommend_start:recommend_end]


def test_history_renders_tool_state_and_old_ask_card_is_removed():
    html = read_page()

    assert "renderContent(msg.role, msg.content)" in html
    assert "extra_info.tool_call" in html
    assert "extra_info.tool_result" in html
    assert "restorePendingTool" in html
    assert "awaiting_destination" in html
    assert "renderAskCard" not in html


def test_frontend_renders_mock_rag_worker_status_and_sources():
    html = read_page()

    assert "worker-results" in html
    assert "本地模拟资料" in html
    assert "worker_completed" in html or "worker_results" in html
    assert "parsed.payload?.result" in html
    assert "assistant_result" in html
    assert "result.evidence" in html
    assert "result.warnings" in html


def test_frontend_renders_agent_tool_and_deep_search_events():
    html = read_page()

    assert "appendAgentToolResult" in html
    assert "appendResearchEvent" in html
    assert 'parsed.type === "tool_result"' in html
    assert '"subagent_tool_call"' in html
    assert '"research_conflict"' in html
    assert "evidence_count" in html


def test_fastapi_exposes_the_frontend_entrypoint():
    from app.main import app

    assert any(route.path == "/ui" for route in app.routes)
