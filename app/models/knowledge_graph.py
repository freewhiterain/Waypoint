"""本地知识图谱：实体与关系。"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class KnowledgeEntity(Base):
    __table_args__ = (
        UniqueConstraint("city", "category", "name", name="uq_knowledge_entity_identity"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    city: Mapped[str] = mapped_column(String(80), index=True)
    category: Mapped[str] = mapped_column(String(40), index=True)
    name: Mapped[str] = mapped_column(String(200))
    source_document: Mapped[str] = mapped_column(String(255))
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())


class KnowledgeRelation(Base):
    __table_args__ = (
        UniqueConstraint(
            "from_entity_id", "to_entity_id", "relation_type", name="uq_knowledge_relation_identity"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledgeentity.id", ondelete="CASCADE"), index=True
    )
    to_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledgeentity.id", ondelete="CASCADE"), index=True
    )
    relation_type: Mapped[str] = mapped_column(String(40), index=True)
    source_document: Mapped[str] = mapped_column(String(255))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())


class TravelRoute(Base):
    """交通图谱的边：两个地点实体之间的交通耗时和距离。

    按无向边存储（from/to 按实体 id 排序后只存一份），同一对地点每种交通方式一条。
    数据来自高德等路线服务，retrieved_at 用来判断缓存是否过期。
    """

    __table_args__ = (
        UniqueConstraint("from_entity_id", "to_entity_id", "mode", name="uq_travel_route_identity"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledgeentity.id", ondelete="CASCADE"), index=True
    )
    to_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("knowledgeentity.id", ondelete="CASCADE"), index=True
    )
    mode: Mapped[str] = mapped_column(String(20))
    minutes: Mapped[int] = mapped_column(Integer)
    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(40))
    source_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
