import pytest

from app.rag.graph_knowledge import GraphKnowledgeService


class _RaisingSessionFactory:
    def __call__(self):
        raise RuntimeError("database unavailable")


@pytest.mark.asyncio
async def test_search_related_entities_returns_empty_list_on_session_error():
    service = GraphKnowledgeService(session_factory=_RaisingSessionFactory())

    result = await service.search_related_entities("成都", "attractions", "宽窄巷子")

    assert result == []
