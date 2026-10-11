"""离线构建本地知识图谱：从模拟 Markdown 资料中抽取实体和关系，写入 Postgres。

不在 FastAPI 请求路径上运行，可重复执行（按唯一约束幂等）。
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from app.rag.graph_knowledge import GraphKnowledgeService
from app.config import settings
from app.models.base import init_db
from app.rag.document_loader import DocumentManager
from app.rag.graph_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    extract_from_documents,
    extract_relations_with_llm,
    resolve_relations,
)
from app.utils.logger import app_logger


async def build_graph(
    *,
    document_manager: DocumentManager | None = None,
    service_factory: Callable[[], GraphKnowledgeService] = GraphKnowledgeService,
    llm_factory: Callable[[], object] | None = None,
    ensure_schema: Callable[[], Awaitable[None]] = init_db,
) -> None:
    await ensure_schema()
    document_manager = document_manager or DocumentManager()
    documents = [
        document
        for document in document_manager.load_all_documents()
        if document.metadata.get("source_type") == "mock_markdown"
    ]
    result = extract_from_documents(documents)
    relations: list[ExtractedRelation] = list(result.relations)

    if settings.llm_api_key:
        if llm_factory is None:
            from app.agents.llm import get_llm as llm_factory  # type: ignore[assignment]
        try:
            llm = llm_factory()
        except Exception as exc:
            app_logger.warning(f"初始化 LLM 失败，跳过 LLM 补充抽取：{type(exc).__name__}: {exc}")
            llm = None
        if llm is not None:
            for document in documents:
                relations.extend(await extract_relations_with_llm(document, llm))
    else:
        app_logger.info("未配置 LLM_API_KEY，跳过 LLM 补充抽取，仅写入规则抽取结果。")

    entities_by_city: dict[str, list[ExtractedEntity]] = {}
    for entity in result.entities:
        entities_by_city.setdefault(entity.city, []).append(entity)

    service = service_factory()
    for city, entities in entities_by_city.items():
        extra_entities, resolved = resolve_relations(city, entities, relations)
        await service.write_entities_and_relations(entities + extra_entities, resolved)
        app_logger.info(
            f"{city}: 写入 {len(entities) + len(extra_entities)} 个实体，{len(resolved)} 条关系。"
        )


def main() -> None:
    asyncio.run(build_graph())


if __name__ == "__main__":
    main()
