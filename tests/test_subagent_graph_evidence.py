"""知识图谱证据必须进入 subagents 的 local_rag 检索，且不能拖垮文档检索。

真实请求走的是 app/agents/subagents/**。图谱检索曾经只接在已删除的旧 Worker
路径上，GraphKnowledgeService 和两张图谱表在真实规划里一条也查不到。

接入方式的取舍：不新增一个 "graph" provider 塞进 provider_order。
provider_order 是**降级链**——base.run 里只要有一个 provider 报 sufficient 就
break，local_rag 一命中，排在它后面的 graph 永远轮不到；排在它前面又会让图谱
抢占文档检索。图谱和文档是互补关系，不是备选关系，所以在 local_rag 这个
provider 内部按 `[*document_evidence, *graph_evidence]` 合并。
"""

import pytest

from app.agents.subagents.tools import build_subagent_tools
from app.schemas.planning import Evidence


class _FakeKnowledge:
    def __init__(self, evidence):
        self._evidence = evidence
        self.calls = []

    def search_destination(self, destination, category, query):
        self.calls.append((destination, category, query))
        return self._evidence


class _FakeGraph:
    def __init__(self, evidence=None, error=None):
        self._evidence = evidence or []
        self._error = error
        self.calls = []

    async def search_related_entities(self, destination, category, query):
        self.calls.append((destination, category, query))
        if self._error is not None:
            raise self._error
        return self._evidence


def _document(content="### 宽窄巷子\n位于青羊区。"):
    return Evidence(
        content=content,
        source="attractions/chengdu.md",
        metadata={"source_type": "mock_markdown"},
    )


def _relation(content="宽窄巷子 位于 青羊区"):
    return Evidence(
        content=content,
        source="attractions/chengdu.md",
        metadata={"source_type": "graph_relation"},
    )


async def _local_rag_tool(knowledge, graph, worker="attractions"):
    tools = await build_subagent_tools(worker, knowledge=knowledge, graph=graph)
    tool = next((item for item in tools if item.name == "local_rag"), None)
    assert tool is not None, "local_rag 工具必须存在，否则测的不是现役路径"
    return tool


def _payload(destination="成都", category="attractions", query="宽窄巷子"):
    return {"destination": destination, "category": category, "query": query}


@pytest.mark.asyncio
async def test_local_rag_merges_graph_evidence():
    knowledge = _FakeKnowledge([_document()])
    graph = _FakeGraph([_relation()])

    result = await (await _local_rag_tool(knowledge, graph)).ainvoke(_payload())

    source_types = [item.metadata.get("source_type") for item in result]
    assert "mock_markdown" in source_types
    assert "graph_relation" in source_types


@pytest.mark.asyncio
async def test_graph_receives_the_same_destination_category_and_query():
    knowledge = _FakeKnowledge([_document()])
    graph = _FakeGraph([_relation()])

    await (await _local_rag_tool(knowledge, graph)).ainvoke(_payload(query="宽窄巷子怎么玩"))

    assert graph.calls == [("成都", "attractions", "宽窄巷子怎么玩")]
    assert knowledge.calls == [("成都", "attractions", "宽窄巷子怎么玩")]


@pytest.mark.asyncio
async def test_graph_failure_does_not_lose_document_evidence():
    """图谱是补充信号，它挂了不该让整个 local_rag provider 报失败。"""
    knowledge = _FakeKnowledge([_document()])
    graph = _FakeGraph(error=RuntimeError("graph down"))

    result = await (await _local_rag_tool(knowledge, graph)).ainvoke(_payload())

    assert [item.metadata.get("source_type") for item in result] == ["mock_markdown"]
    assert result.sufficiency.status == "sufficient"


@pytest.mark.asyncio
async def test_graph_alone_is_enough_when_documents_are_empty():
    """文档库空但图谱有内容时，这一轮就该算命中，而不是退化成 empty。"""
    knowledge = _FakeKnowledge([])
    graph = _FakeGraph([_relation()])

    result = await (await _local_rag_tool(knowledge, graph)).ainvoke(_payload())

    assert [item.metadata.get("source_type") for item in result] == ["graph_relation"]
    assert result.sufficiency.status == "sufficient"


@pytest.mark.asyncio
async def test_both_empty_still_reports_empty():
    knowledge = _FakeKnowledge([])
    graph = _FakeGraph([])

    result = await (await _local_rag_tool(knowledge, graph)).ainvoke(_payload())

    assert result.sufficiency.status == "empty"


@pytest.mark.asyncio
async def test_duplicate_content_from_both_sources_is_not_double_counted():
    """同一句话既在文档里又被抽成关系时，不该在 prompt 里出现两遍。"""
    shared = "宽窄巷子 位于 青羊区"
    knowledge = _FakeKnowledge([_document(shared)])
    graph = _FakeGraph([_relation(shared)])

    result = await (await _local_rag_tool(knowledge, graph)).ainvoke(_payload())

    assert len(result) == 1


@pytest.mark.asyncio
async def test_food_worker_also_gets_the_graph_wired():
    """food 的 policy 里同样有 local_rag，不该只给 attractions/hotel 接。"""
    knowledge = _FakeKnowledge([_document()])
    graph = _FakeGraph([_relation()])

    await (await _local_rag_tool(knowledge, graph, worker="food")).ainvoke(
        _payload(category="food")
    )

    assert graph.calls == [("成都", "food", "宽窄巷子")]
