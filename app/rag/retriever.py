"""
混合检索器：BM25（标题加权 + 同义词扩展 + 相邻词组加分）+ Dense + RRF 融合
"""
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
import jieba
from app.utils.logger import app_logger
from app.rag.identifiers import chunk_id
from app.rag.reranker import RelevanceReranker
from app.rag.synonyms import expand_synonyms


TITLE_WEIGHT_REPEAT = 3
BIGRAM_BONUS = 0.5
# 融合阶段保留的候选倍数：RRF 先留出比 k 更宽的池子，重排才有挑选空间。
CANDIDATE_POOL_MULTIPLIER = 4
# 同义词扩展只作为补充召回，权重低于原始查询词，避免顶掉精确命中。
SYNONYM_WEIGHT = 0.3


class HybridRetriever:
    """
    混合检索器

    结合：
    - BM25（关键词匹配，标题字段加权 + 同义词扩展召回 + 相邻词组加分）
    - Dense（语义相似度，检索失败时自动降级为跳过）
    - RRF（倒数排名融合）
    """

    def __init__(
            self,
            vectorstore: Chroma | None,
            documents: List[Document],
            k: int = 5,
            parent_documents: List[Document] | None = None,
            reranker: RelevanceReranker | None = None,
    ):
        self.vectorstore = vectorstore
        self.documents = documents
        self.k = k
        self.parent_documents = {
            str(doc.metadata.get("parent_id")): doc
            for doc in (parent_documents or [])
            if doc.metadata.get("parent_id")
        }
        self.reranker = reranker
        self._documents_by_id = {chunk_id(doc): doc for doc in documents}
        self._init_bm25()

    def _init_bm25(self):
        """初始化 BM25 索引：section_title 命中的词会重复计入，获得更高权重。"""
        app_logger.info("初始化 BM25 索引...")
        tokenized_docs = [self._tokenize_document(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        app_logger.info("✅ BM25 索引初始化完成")

    @staticmethod
    def _tokenize_document(document: Document) -> List[str]:
        body_tokens = list(jieba.cut(document.page_content))
        section_title = str(document.metadata.get("section_title", "")).strip()
        if not section_title:
            return body_tokens
        title_tokens = list(jieba.cut(section_title)) * TITLE_WEIGHT_REPEAT
        return title_tokens + body_tokens

    def _bigram_scores(self, query_tokens: List[str]) -> List[float]:
        bigrams = ["".join(pair) for pair in zip(query_tokens, query_tokens[1:])]
        if not bigrams:
            return [0.0] * len(self.documents)
        return [
            BIGRAM_BONUS * sum(1 for bigram in bigrams if bigram in document.page_content)
            for document in self.documents
        ]

    def retrieve(
            self,
            query: str,
            *,
            metadata_filter: dict | None = None,
    ) -> List[Document]:
        """
        混合检索

        流程：
        1. BM25 检索候选池（原始查询词 + 降权的同义词扩展 + 相邻词组加分）
        2. Dense 检索候选池（失败时自动降级，不抛出异常）
        3. RRF 融合，保留比 k 更宽的候选池
        4. 重排并截断到 top-k
        5. 回溯父文档
        """
        pool_size = max(self.k * CANDIDATE_POOL_MULTIPLIER, self.k)

        # BM25 检索
        query_tokens = list(jieba.cut(query))
        bm25_scores = self._bm25_scores(query_tokens)
        bigram_bonus = self._bigram_scores(query_tokens)
        bm25_scores = [score + bigram_bonus[i] for i, score in enumerate(bm25_scores)]
        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:pool_size]
        bm25_docs = [(self.documents[i], bm25_scores[i]) for i in bm25_top_indices]
        app_logger.debug(f"BM25 检索到 {len(bm25_docs)} 个候选")

        # Dense 检索（失败时降级为空候选，不影响 BM25 结果）
        dense_docs: List[Tuple[Document, float]] = []
        if self.vectorstore is not None:
            try:
                dense_docs = self.vectorstore.similarity_search_with_score(
                    query, k=pool_size, filter=metadata_filter
                )
            except Exception as exc:
                app_logger.warning(f"Dense 检索失败，本次查询退化为纯 BM25：{type(exc).__name__}: {exc}")
                dense_docs = []
            # 持久化的向量库可能比当前语料旧，也可能装着别的城市/类别：只认当前
            # 检索器自己的切片，并换成内存里的那份，保证父文档回溯和元数据一致。
            dense_docs = [
                (self._documents_by_id[chunk_id(doc)], score)
                for doc, score in dense_docs
                if chunk_id(doc) in self._documents_by_id
            ]
        app_logger.debug(f"Dense 检索到 {len(dense_docs)} 个候选")

        # RRF 融合：保留宽候选池，让重排器有挑选空间而不只是调序
        fused_docs = self._rrf_fusion(bm25_docs, dense_docs, k=60, limit=pool_size)
        if self.reranker:
            # 重排整个候选池而非仅 top-k：父文档折叠会合并候选，
            # 需要多余的候选来把结果补齐到 k。
            fused_docs = self.reranker.rerank(query, fused_docs, top_k=pool_size)
        resolved = self._resolve_parent_documents(fused_docs)
        app_logger.info(f"✅ 混合检索完成，返回 {len(resolved)} 个结果")
        return resolved

    def _bm25_scores(self, query_tokens: List[str]) -> List[float]:
        """原始查询词全权重，同义词扩展按 SYNONYM_WEIGHT 降权后叠加。"""
        scores = list(self.bm25.get_scores(query_tokens))
        synonym_tokens = expand_synonyms(query_tokens)
        if not synonym_tokens:
            return scores
        synonym_scores = self.bm25.get_scores(synonym_tokens)
        return [score + SYNONYM_WEIGHT * synonym_scores[i] for i, score in enumerate(scores)]

    def _rrf_fusion(
            self,
            bm25_docs: List[Tuple[Document, float]],
            dense_docs: List[Tuple[Document, float]],
            k: int = 60,
            limit: int | None = None,
    ) -> List[Document]:
        """倒数排名融合（RRF）"""
        scores = {}

        for rank, (doc, _) in enumerate(bm25_docs, 1):
            doc_id = chunk_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        for rank, (doc, _) in enumerate(dense_docs, 1):
            doc_id = chunk_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        all_docs = {chunk_id(doc): doc for doc, _ in bm25_docs + dense_docs}
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        cutoff = self.k if limit is None else limit
        return [all_docs[doc_id] for doc_id, _ in sorted_docs[:cutoff]]

    def _resolve_parent_documents(self, documents: List[Document]) -> List[Document]:
        """把命中的子块换成父文档；多个子块折叠到同一父文档时按顺序补齐到 k。"""
        if not self.parent_documents:
            return documents[:self.k]

        resolved = []
        seen = set()
        for document in documents:
            parent_id = str(document.metadata.get("parent_id", ""))
            parent = self.parent_documents.get(parent_id, document)
            key = chunk_id(parent)
            if key in seen:
                continue
            # 子块上的重排分数在换成父文档后会丢失，这里显式传递下去。
            rerank_score = document.metadata.get("rerank_score")
            if rerank_score is not None and parent.metadata.get("rerank_score") != rerank_score:
                parent = Document(
                    page_content=parent.page_content,
                    metadata={**parent.metadata, "rerank_score": rerank_score},
                )
            resolved.append(parent)
            seen.add(key)
            if len(resolved) >= self.k:
                break
        return resolved
