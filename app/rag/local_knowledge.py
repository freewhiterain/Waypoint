"""本地静态知识的 Hybrid RAG 查询入口。"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.rag.document_loader import DocumentManager
from app.rag.evidence import evidence_from_document
from app.rag.local_embeddings import LOCAL_MOCK_COLLECTION, get_ollama_embeddings
from app.rag.reranker import CrossEncoderReranker, RelevanceReranker
from app.rag.retriever import HybridRetriever
from app.rag.text_splitter import ParentDocumentSplitter
from app.rag.vectorstore import VectorStoreManager
from app.schemas.planning import Evidence, TaskType
from app.utils.logger import app_logger


class LocalKnowledgeService:
    def __init__(
        self,
        documents: list[Document] | None = None,
        vectorstore: Chroma | None = None,
    ):
        self.documents = documents if documents is not None else DocumentManager().load_all_documents()
        self.vectorstore = vectorstore if vectorstore is not None else self._load_vectorstore()
        self.retriever = self._build_retriever(self.documents, self.vectorstore)
        # 按 (城市, 类别) 缓存检索器：构建包含文档切分和 BM25 索引重建，
        # 放在请求路径上每次重做是 CPU 热点。
        self._scoped_retrievers: dict[tuple[str, str], HybridRetriever | None] = {}

    @staticmethod
    def _load_vectorstore() -> Chroma | None:
        try:
            manager = VectorStoreManager(
                collection_name=LOCAL_MOCK_COLLECTION,
                embeddings=get_ollama_embeddings(),
            )
            return manager.load_vectorstore()
        except Exception as exc:
            app_logger.warning(f"本地向量库不可用，Dense 检索退化为跳过：{type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _select_reranker() -> RelevanceReranker:
        if settings.enable_cross_encoder_rerank:
            return CrossEncoderReranker(settings.cross_encoder_model)
        return RelevanceReranker()

    @staticmethod
    def _build_retriever(documents: list[Document], vectorstore: Chroma | None) -> HybridRetriever | None:
        parents, children = ParentDocumentSplitter().split_documents(documents)
        if not children:
            return None
        return HybridRetriever(
            vectorstore=vectorstore,
            documents=children,
            parent_documents=parents,
            reranker=LocalKnowledgeService._select_reranker(),
            k=4,
        )

    def search(self, query: str) -> list[Evidence]:
        if self.retriever is None:
            return []
        return [evidence_from_document(document) for document in self.retriever.retrieve(query)]

    def search_destination(
        self,
        destination: str,
        category: TaskType,
        query: str,
    ) -> list[Evidence]:
        normalized_destination = destination.strip().casefold()
        normalized_category = category.strip().casefold()
        retriever = self._scoped_retriever(normalized_destination, normalized_category)
        if retriever is None:
            return []

        retrieval_query = f"{destination} {category} {query}"
        metadata_filter = (
            {"$and": [{"city": destination.strip()}, {"category": category.strip()}]}
            if self.vectorstore is not None
            else None
        )
        return [
            evidence_from_document(document)
            for document in retriever.retrieve(retrieval_query, metadata_filter=metadata_filter)
        ]

    def _scoped_retriever(
        self,
        normalized_destination: str,
        normalized_category: str,
    ) -> HybridRetriever | None:
        cache_key = (normalized_destination, normalized_category)
        if cache_key in self._scoped_retrievers:
            return self._scoped_retrievers[cache_key]

        documents = [
            document
            for document in self.documents
            if str(document.metadata.get("city", "")).strip().casefold() == normalized_destination
            and str(document.metadata.get("category", "")).strip().casefold() == normalized_category
        ]
        retriever = self._build_retriever(documents, self.vectorstore) if documents else None
        self._scoped_retrievers[cache_key] = retriever
        return retriever


@lru_cache(maxsize=1)
def get_local_knowledge_service() -> LocalKnowledgeService:
    return LocalKnowledgeService()


def load_destination_evidence(destination: str, topic: str) -> list[Evidence]:
    items = get_local_knowledge_service().search(f"{destination} {topic}")
    return [
        item
        for item in items
        if destination in item.content or destination in item.source
    ]
