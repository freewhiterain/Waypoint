"""
pytest 配置
"""
import sys
import asyncio
import pytest

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def _hermetic_defaults(monkeypatch):
    """测试默认不访问真实模型和真实数据库里的会话槽位。

    本机 .env 里有真实的 LLM Key；不清掉的话，凡是走到主 Agent、子 Agent 分析、
    行程合成的测试都会真的调用模型，又慢又不稳定。需要"已配置 Key"的测试自己再设。
    槽位默认存内存：生产实现读写 conversation.extra_info，会连本机 PostgreSQL。
    """
    from app.services import trip_form as tools
    from app.services import chat_flow as chat
    from app.config import settings
    from app.services.slots import InMemorySlotRepository

    repository = InMemorySlotRepository()
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(chat, "PostgresSlotRepository", lambda: repository)
    monkeypatch.setattr(tools, "PostgresSlotRepository", lambda: repository)
