"""路径里的会话 id 和消息内容在进入业务逻辑前就被校验（不访问数据库）。"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.main import app
from app.models.base import get_db


class _UnusedSession:
    async def execute(self, *_args, **_kwargs):
        raise AssertionError("校验失败的请求不应该走到数据库")


@pytest.fixture
def client():
    async def override_db():
        yield _UnusedSession()

    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid.uuid4())
    app.dependency_overrides[get_db] = override_db
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "123", "0.0", "' OR 1=1 --"])
@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/api/v1/chat/stream/{id}", {"content": "你好"}),
        ("get", "/api/v1/chat/history/{id}", None),
        ("get", "/api/v1/conversations/{id}", None),
        ("patch", "/api/v1/conversations/{id}", {"title": "x"}),
        ("delete", "/api/v1/conversations/{id}", None),
        ("get", "/api/v1/drafts/{id}", None),
    ],
)
def test_malformed_conversation_id_is_rejected_with_422(client, method, path, body, bad_id):
    response = client.request(method, path.format(id=bad_id), json=body)

    assert response.status_code == 422


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_blank_message_is_rejected_with_422(client, content):
    response = client.post(f"/api/v1/chat/stream/{uuid.uuid4()}", json={"content": content})

    assert response.status_code == 422
