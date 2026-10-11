"""图谱检索必须真的用上 query，并且不能无上界地灌证据。

原先 search_related_entities 收下 query 参数却一次也没引用：返回的是
(城市, 类别) 下的**全部**关系。两个后果：

1. 相关性为零。问"宽窄巷子附近有什么"和问"都江堰怎么走"拿到完全一样的结果，
   query 形同装饰。
2. 数量无上界。每条关系都会在 rag_analysis 里变成一个 CandidateOption 并写进
   LLM prompt（见 _prompt 的 digest 拼接），而文档侧是 HybridRetriever(k=4)
   有上限的。城市图谱一旦长起来（这正是后续要导数据的方向），图谱证据会把
   文档证据在 prompt 里挤到边缘，且 token 成本随库增长线性上涨。

设计取舍：这里做的是**排序 + 截断**，不是硬过滤。图谱关系文本只有两个实体名
加一个关系词，词面命中极其稀疏——"成都 attractions" 这类 task.query 和
"宽窄巷子 位于 青羊区" 一个词都不重叠。硬过滤会把稀疏图谱直接清空，
比不过滤更糟。所以命中的排前面，没命中的按 confidence 兜底，统一截断。
"""

import uuid

import pytest

from app.rag.graph_knowledge import _MAX_GRAPH_EVIDENCE, GraphKnowledgeService


class _Entity:
    def __init__(self, city, category, name):
        self.id = uuid.uuid4()
        self.city = city
        self.category = category
        self.name = name


class _Relation:
    def __init__(self, from_entity, to_entity, relation_type, source_document, confidence=1.0):
        self.id = uuid.uuid4()
        self.from_entity_id = from_entity.id
        self.to_entity_id = to_entity.id
        self.relation_type = relation_type
        self.source_document = source_document
        self.confidence = confidence


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Session:
    """按 execute 的调用顺序回放：源实体 → 关系 → 目标实体。"""

    def __init__(self, sources, relations, targets):
        self._queue = [sources, relations, targets]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, _statement):
        return _Result(self._queue.pop(0) if self._queue else [])


def _service(sources, relations, targets) -> GraphKnowledgeService:
    return GraphKnowledgeService(session_factory=lambda: _Session(sources, relations, targets))


def _chengdu_graph(count: int):
    """一个源实体连出 count 条关系，目标依次是 区域0..区域N。"""
    source = _Entity("成都", "attractions", "宽窄巷子")
    targets = [_Entity("成都", "area", f"区域{index}") for index in range(count)]
    relations = [
        _Relation(source, target, "located_in", "attractions/chengdu.md") for target in targets
    ]
    return [source], relations, targets


@pytest.mark.asyncio
async def test_query_matching_relations_rank_above_non_matching_ones():
    source_a = _Entity("成都", "attractions", "宽窄巷子")
    source_b = _Entity("成都", "attractions", "都江堰")
    area = _Entity("成都", "area", "青羊区")
    relations = [
        # 故意把不相关的放在前面：不排序的话它会先被返回。
        _Relation(source_b, area, "located_in", "b.md"),
        _Relation(source_a, area, "located_in", "a.md"),
    ]
    service = _service([source_a, source_b], relations, [area])

    evidence = await service.search_related_entities("成都", "attractions", "宽窄巷子怎么玩")

    assert evidence[0].content.startswith("宽窄巷子")


@pytest.mark.asyncio
async def test_result_count_is_capped():
    sources, relations, targets = _chengdu_graph(_MAX_GRAPH_EVIDENCE + 6)
    service = _service(sources, relations, targets)

    evidence = await service.search_related_entities("成都", "attractions", "宽窄巷子")

    assert len(evidence) == _MAX_GRAPH_EVIDENCE


@pytest.mark.asyncio
async def test_unmatched_query_still_returns_evidence_ordered_by_confidence():
    """稀疏图谱不能因为词面不命中就被清空——硬过滤在这里比不过滤更糟。"""
    source = _Entity("成都", "attractions", "宽窄巷子")
    low = _Entity("成都", "area", "青羊区")
    high = _Entity("成都", "area", "锦江区")
    relations = [
        _Relation(source, low, "located_in", "a.md", confidence=0.2),
        _Relation(source, high, "near", "a.md", confidence=0.9),
    ]
    service = _service([source], relations, [low, high])

    # "成都 attractions" 是现役 worker 真实传进来的 task.query，与关系文本零重叠。
    evidence = await service.search_related_entities("成都", "attractions", "成都 attractions")

    assert len(evidence) == 2
    assert evidence[0].confidence == 0.9


@pytest.mark.asyncio
async def test_relevance_score_is_exposed_in_metadata():
    """与文档侧 rerank_score 一致：排序依据必须可追溯，否则没法排查召回问题。"""
    sources, relations, targets = _chengdu_graph(1)
    service = _service(sources, relations, targets)

    evidence = await service.search_related_entities("成都", "attractions", "宽窄巷子")

    assert evidence[0].metadata["graph_relevance"] >= 0.0
    # 既有字段不能因为这次改动丢掉。
    assert evidence[0].metadata["source_type"] == "graph_relation"
    assert evidence[0].metadata["from_entity"] == "宽窄巷子"


@pytest.mark.asyncio
async def test_empty_query_does_not_crash():
    """query 可能是空串（上游拼接失败时），不能因此炸掉整个 worker。"""
    sources, relations, targets = _chengdu_graph(2)
    service = _service(sources, relations, targets)

    evidence = await service.search_related_entities("成都", "attractions", "   ")

    assert len(evidence) == 2
