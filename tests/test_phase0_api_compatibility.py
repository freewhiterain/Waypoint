"""现有 HTTP/SSE 表面兼容性测试，不连接数据库或外部服务。"""

import json

from app.api.sse import sse
from app.main import app


def test_required_legacy_routes_remain_registered():
    paths = {route.path for route in app.routes}

    assert "/api/v1/users/register" in paths
    assert "/api/v1/users/login" in paths
    assert "/api/v1/conversations" in paths
    assert "/api/v1/chat/stream/{conversation_id}" in paths
    assert "/api/v1/chat/history/{conversation_id}" in paths


def test_legacy_sse_token_and_done_frames_remain_json_data_frames():
    token_frame = sse({"type": "token", "content": "成都"})
    done_frame = sse({"type": "done"})

    assert token_frame.endswith("\n\n")
    assert done_frame.endswith("\n\n")
    assert json.loads(token_frame.removeprefix("data: ")) == {
        "type": "token",
        "content": "成都",
    }
    assert json.loads(done_frame.removeprefix("data: ")) == {"type": "done"}
