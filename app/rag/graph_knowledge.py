"""Local knowledge graph persistence and query, used alongside document RAG."""

from __future__ import annotations

from uuid import UUID

import jieba
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.models.base import async_session_maker
from app.models.knowledge_graph import KnowledgeEntity, KnowledgeRelation
from app.rag.graph_extraction import ExtractedEntity, ResolvedRelation
from app.schemas.planning import Evidence, TaskType
from app.utils.logger import app_logger


_RELATION_LABELS = {
    "located_in": "位于",
    "near": "临近",
    "connects_to": "连接到",
}

# 与文档侧 HybridRetriever(k=4) 对齐的量级。图谱证据在 rag_analysis 里会逐条变成
# CandidateOption 并拼进 LLM prompt，没有上界的话城市图谱一长起来就会把文档证据
# 挤到 prompt 边缘，token 成本还随库线性上涨。
_MAX_GRAPH_EVIDENCE = 6


def _query_terms(query: str) -> set[str]:
    """按 app/rag/reranker.py 的同一套中文分词口径切 query。"""
    return {term.strip().casefold() for term in jieba.cut(query or "") if term.strip()}


def _relevance(query_terms: set[str], from_name: str, to_name: str) -> float:
    """图谱关系与 query 的实体名命中度，0.0 表示完全没命中。

    只对两端实体名打分，不含关系词（"位于""临近"这类词在任何 query 里都可能
    出现，计进去只会让所有关系一起加分，等于没排序）。

    命中度用"实体名整体是否出现在 query 里"和"分词后是否有交集"两路取大：前者
    覆盖 jieba 把"宽窄巷子"切成"宽窄"+"巷子"而 query 里是完整词的情况。
    """
    if not query_terms:
        return 0.0
    query_text = "".join(query_terms)
    best = 0.0
    for name in (from_name, to_name):
        normalized = name.strip().casefold()
        if not normalized:
            continue
        if normalized in query_text:
            best = max(best, 1.0)
            continue
        name_terms = {term.strip().casefold() for term in jieba.cut(normalized) if term.strip()}
        if name_terms:
            best = max(best, len(name_terms & query_terms) / len(name_terms))
    return best


class GraphKnowledgeService:
    def __init__(self, session_factory=async_session_maker):
        self.session_factory = session_factory

    async def write_entities_and_relations(
        self,
        entities: list[ExtractedEntity],
        relations: list[ResolvedRelation],
    ) -> None:
        async with self.session_factory() as session, session.begin():
            entity_ids: dict[tuple[str, str, str], UUID] = {}
            for item in entities:
                statement = (
                    insert(KnowledgeEntity)
                    .values(
                        city=item.city, category=item.category, name=item.name,
                        source_document=item.source_document, attributes={},
                    )
                    .on_conflict_do_update(
                        constraint="uq_knowledge_entity_identity",
                        set_={"source_document": item.source_document},
                    )
                    .returning(KnowledgeEntity.id)
                )
                entity_id = (await session.execute(statement)).scalar_one()
                entity_ids[(item.city, item.category, item.name)] = entity_id

            for relation in relations:
                from_id = entity_ids.get((relation.from_city, relation.from_category, relation.from_name))
                to_id = entity_ids.get((relation.to_city, relation.to_category, relation.to_name))
                if from_id is None or to_id is None:
                    continue
                statement = (
                    insert(KnowledgeRelation)
                    .values(
                        from_entity_id=from_id, to_entity_id=to_id,
                        relation_type=relation.relation_type,
                        source_document=relation.source_document,
                        confidence=relation.confidence,
                    )
                    .on_conflict_do_nothing(constraint="uq_knowledge_relation_identity")
                )
                await session.execute(statement)

    async def search_related_entities(
        self,
        destination: str,
        category: TaskType,
        query: str,
    ) -> list[Evidence]:
        city = destination.strip()
        normalized_category = category.strip().casefold()
        try:
            async with self.session_factory() as session:
                entities = (
                    await session.execute(
                        select(KnowledgeEntity).where(
                            KnowledgeEntity.city == city,
                            KnowledgeEntity.category == normalized_category,
                        )
                    )
                ).scalars().all()
                if not entities:
                    return []
                entity_ids = [entity.id for entity in entities]
                relations = (
                    await session.execute(
                        select(KnowledgeRelation).where(KnowledgeRelation.from_entity_id.in_(entity_ids))
                    )
                ).scalars().all()
                if not relations:
                    return []
                target_ids = {relation.to_entity_id for relation in relations}
                targets = (
                    await session.execute(select(KnowledgeEntity).where(KnowledgeEntity.id.in_(target_ids)))
                ).scalars().all()

                entities_by_id = {entity.id: entity for entity in entities}
                targets_by_id = {entity.id: entity for entity in targets}
                query_terms = _query_terms(query)
                scored: list[tuple[float, float, Evidence]] = []
                for relation in relations:
                    source_entity = entities_by_id.get(relation.from_entity_id)
                    target_entity = targets_by_id.get(relation.to_entity_id)
                    if source_entity is None or target_entity is None:
                        continue
                    label = _RELATION_LABELS.get(relation.relation_type, relation.relation_type)
                    content = f"{source_entity.name} {label} {target_entity.name}"
                    relevance = _relevance(query_terms, source_entity.name, target_entity.name)
                    scored.append(
                        (
                            relevance,
                            relation.confidence,
                            Evidence(
                                content=content,
                                source=relation.source_document,
                                confidence=relation.confidence,
                                metadata={
                                    "source_type": "graph_relation",
                                    "category": normalized_category,
                                    "relation_type": relation.relation_type,
                                    "from_entity": source_entity.name,
                                    "to_entity": target_entity.name,
                                    # 和文档侧 rerank_score 一样把排序依据留在证据上，
                                    # 否则召回不对时无从判断是打分问题还是数据问题。
                                    "graph_relevance": relevance,
                                },
                            ),
                        )
                    )
                # 命中 query 的排前面，同分再按关系置信度；最后统一截断。
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                evidence = [item[2] for item in scored[:_MAX_GRAPH_EVIDENCE]]
        except Exception as exc:
            app_logger.warning(f"知识图谱查询失败，返回空结果：{type(exc).__name__}: {exc}")
            return []

        return evidence


_graph_knowledge_service: GraphKnowledgeService | None = None


def get_graph_knowledge_service() -> GraphKnowledgeService:
    global _graph_knowledge_service
    if _graph_knowledge_service is None:
        _graph_knowledge_service = GraphKnowledgeService()
    return _graph_knowledge_service
