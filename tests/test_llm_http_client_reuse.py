"""get_llm 不得每次调用都新建 httpx 连接池。

原实现每次调用都 new 一对 httpx.Client / httpx.AsyncClient，两个都带
独立连接池且永不 close。get_llm 在每一次对话轮次都会被调用
（main_agent 路由、planning 抽取、open_qa、supervisor 行程合成），
于是套接字和连接池随请求量单调增长，直到耗尽文件描述符。

复用同一对客户端即可：httpx 客户端本身是线程安全、可并发复用的，
连接池正是为跨请求复用而设计的。
"""

import httpx
import pytest

from app.core import llm as llm_module


@pytest.fixture(autouse=True)
def _reset_shared_clients(monkeypatch):
    """每个用例前后都清空缓存，避免相互污染。构造客户端需要 Key，给一个假的即可。"""
    monkeypatch.setattr("app.config.settings.llm_api_key", "test-key")
    llm_module.reset_http_clients()
    yield
    llm_module.reset_http_clients()


def test_repeated_get_llm_calls_share_one_pair_of_http_clients(monkeypatch):
    """统计底层客户端的创建次数。

    不能用函数替换 httpx.Client：openai SDK 内部会对客户端做 isinstance
    检查，传入函数会让它抛 TypeError。这里改成子类，既保持类型又能计数。
    """
    created: list[object] = []

    class CountingClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    class CountingAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(llm_module.httpx, "Client", CountingClient)
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", CountingAsyncClient)

    first = llm_module.get_llm()
    second = llm_module.get_llm()
    third = llm_module.get_llm(temperature=0.3)

    assert len(created) == 2, f"应只创建一对 httpx 客户端，实际创建了 {len(created)} 个"
    assert first.http_client is second.http_client is third.http_client
    assert first.async_client is not None


def test_temperature_override_still_takes_effect():
    """复用客户端不能把 temperature 也一起缓存掉。"""
    from app.config import settings

    default = llm_module.get_llm()
    overridden = llm_module.get_llm(temperature=0.3)

    assert overridden.temperature == 0.3
    assert default.temperature == settings.llm_temperature


def test_reset_closes_the_shared_sync_client(monkeypatch):
    llm_module.get_llm()
    client = llm_module._http_client

    assert client is not None
    llm_module.reset_http_clients()

    assert client.is_closed
    assert llm_module._http_client is None
