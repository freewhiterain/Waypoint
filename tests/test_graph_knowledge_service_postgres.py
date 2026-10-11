import os

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

import app.models  # noqa: F401
from app.rag.graph_knowledge import GraphKnowledgeService
from app.models.base import async_session_maker, init_db
from app.models.knowledge_graph import KnowledgeEntity, KnowledgeRelation
from app.rag.graph_extraction import ExtractedEntity, ResolvedRelation


pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        os.getenv("RUN_POSTGRES_TESTS") != "1",
        reason="requires RUN_POSTGRES_TESTS=1 and a reachable PostgreSQL database",
    ),
]


@pytest.mark.asyncio
async def test_knowledge_entity_identity_is_unique():
    await init_db()
    async with async_session_maker() as session, session.begin():
        session.add(
            KnowledgeEntity(
                city="__test_city__", category="attractions", name="__test_entity__",
                source_document="tests/fixture.md",
            )
        )

    with pytest.raises(IntegrityError):
        async with async_session_maker() as session, session.begin():
            session.add(
                KnowledgeEntity(
                    city="__test_city__", category="attractions", name="__test_entity__",
                    source_document="tests/fixture.md",
                )
            )

    # No KnowledgeRelation rows were created in this test, so only entities need cleanup.
    async with async_session_maker() as session, session.begin():
        await session.execute(
            KnowledgeEntity.__table__.delete().where(KnowledgeEntity.city == "__test_city__")
        )


@pytest.mark.asyncio
async def test_write_and_search_round_trip_is_idempotent_and_category_scoped():
    await init_db()
    service = GraphKnowledgeService()
    entities = [
        ExtractedEntity(city="__test_city__", category="attractions", name="__attraction__", source_document="a.md"),
        ExtractedEntity(city="__test_city__", category="area", name="__area__", source_document="a.md"),
        ExtractedEntity(city="__test_city__", category="hotel", name="__hotel__", source_document="h.md"),
    ]
    relations = [
        ResolvedRelation(
            from_city="__test_city__", from_category="attractions", from_name="__attraction__",
            to_city="__test_city__", to_category="area", to_name="__area__",
            relation_type="located_in", source_document="a.md",
        ),
        ResolvedRelation(
            from_city="__test_city__", from_category="hotel", from_name="__hotel__",
            to_city="__test_city__", to_category="attractions", to_name="__attraction__",
            relation_type="near", source_document="h.md",
        ),
    ]

    try:
        await service.write_entities_and_relations(entities, relations)
        await service.write_entities_and_relations(entities, relations)  # must be idempotent

        attractions_evidence = await service.search_related_entities("__test_city__", "attractions", "q")
        hotel_evidence = await service.search_related_entities("__test_city__", "hotel", "q")
        weather_evidence = await service.search_related_entities("__test_city__", "weather", "q")

        assert [item.content for item in attractions_evidence] == ["__attraction__ 位于 __area__"]
        assert attractions_evidence[0].metadata["source_type"] == "graph_relation"
        assert [item.content for item in hotel_evidence] == ["__hotel__ 临近 __attraction__"]
        assert weather_evidence == []
    finally:
        async with async_session_maker() as session, session.begin():
            ids = (
                await session.execute(
                    select(KnowledgeEntity.id).where(KnowledgeEntity.city == "__test_city__")
                )
            ).scalars().all()
            if ids:
                await session.execute(
                    KnowledgeRelation.__table__.delete().where(
                        KnowledgeRelation.from_entity_id.in_(ids)
                    )
                )
                await session.execute(
                    KnowledgeEntity.__table__.delete().where(KnowledgeEntity.id.in_(ids))
                )
