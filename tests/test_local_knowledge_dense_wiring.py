from langchain_core.documents import Document

from app.rag.local_knowledge import LocalKnowledgeService


class _RecordingVectorstore:
    def __init__(self):
        self.calls: list[dict] = []

    def similarity_search_with_score(self, query, k, filter=None):
        self.calls.append({"query": query, "filter": filter})
        return []


def test_search_destination_filters_dense_search_by_city_and_category():
    documents = [
        Document(
            page_content="位于成华区。是熊猫文化主题下的代表性自然教育地点。",
            metadata={"city": "成都", "category": "attractions", "chunk_id": "panda"},
        ),
    ]
    vectorstore = _RecordingVectorstore()
    service = LocalKnowledgeService(documents=documents, vectorstore=vectorstore)

    service.search_destination("成都", "attractions", "熊猫基地")

    assert vectorstore.calls
    assert vectorstore.calls[0]["filter"] == {"$and": [{"city": "成都"}, {"category": "attractions"}]}


def test_search_destination_strips_but_does_not_casefold_dense_metadata_filter():
    documents = [
        Document(
            page_content="位于成华区。是熊猫文化主题下的代表性自然教育地点。",
            metadata={"city": "成都", "category": "attractions", "chunk_id": "panda"},
        ),
    ]
    vectorstore = _RecordingVectorstore()
    service = LocalKnowledgeService(documents=documents, vectorstore=vectorstore)

    service.search_destination(" 成都 ", " attractions ", "熊猫基地")

    assert vectorstore.calls
    assert vectorstore.calls[0]["filter"] == {"$and": [{"city": "成都"}, {"category": "attractions"}]}


def test_service_reuses_the_same_vectorstore_instance_across_queries_without_rebuilding():
    documents = [
        Document(
            page_content="位于成华区。",
            metadata={"city": "成都", "category": "attractions", "chunk_id": "panda"},
        ),
        Document(
            page_content="位于青羊区。",
            metadata={"city": "成都", "category": "hotel", "chunk_id": "hotel"},
        ),
    ]
    vectorstore = _RecordingVectorstore()
    service = LocalKnowledgeService(documents=documents, vectorstore=vectorstore)

    service.search_destination("成都", "attractions", "熊猫")
    service.search_destination("成都", "hotel", "住宿")

    assert service.vectorstore is vectorstore
    assert len(vectorstore.calls) == 2


def test_repeated_scoped_queries_reuse_the_cached_retriever():
    # 改动前每次 search_destination 都重新切分文档并重建 BM25 索引，
    # 这是请求路径上的 CPU 热点。缓存生效时同一 (城市, 类别) 只应构建一次。
    documents = [
        Document(
            page_content="位于成华区。是熊猫文化主题下的代表性自然教育地点。",
            metadata={"city": "成都", "category": "attractions", "chunk_id": "panda"},
        ),
    ]
    service = LocalKnowledgeService(documents=documents, vectorstore=None)
    builds: list[int] = []
    original_build = LocalKnowledgeService._build_retriever

    def counting_build(docs, vectorstore):
        builds.append(len(docs))
        return original_build(docs, vectorstore)

    service._build_retriever = staticmethod(counting_build)

    service.search_destination("成都", "attractions", "熊猫基地")
    service.search_destination("成都", "attractions", "门票")

    assert len(builds) == 1


def test_scoped_retriever_cache_keys_do_not_collide_across_categories():
    # 缓存必须按 (城市, 类别) 分键：否则第二个类别会复用第一个类别的检索器，
    # 返回错误类别的资料。
    documents = [
        Document(
            page_content="熊猫基地位于成华区。",
            metadata={"city": "成都", "category": "attractions", "chunk_id": "panda"},
        ),
        Document(
            page_content="青羊区住宿片区靠近宽窄巷子。",
            metadata={"city": "成都", "category": "hotel", "chunk_id": "hotel"},
        ),
    ]
    service = LocalKnowledgeService(documents=documents, vectorstore=None)

    attractions = service.search_destination("成都", "attractions", "熊猫")
    hotels = service.search_destination("成都", "hotel", "住宿")

    assert all("熊猫基地" in item.content for item in attractions)
    assert all("住宿" in item.content for item in hotels)


def test_missing_scope_is_cached_as_empty_instead_of_rebuilding_each_time():
    documents = [
        Document(
            page_content="熊猫基地位于成华区。",
            metadata={"city": "成都", "category": "attractions", "chunk_id": "panda"},
        ),
    ]
    service = LocalKnowledgeService(documents=documents, vectorstore=None)

    assert service.search_destination("西安", "attractions", "兵马俑") == []
    assert service.search_destination("西安", "attractions", "兵马俑") == []
    assert service._scoped_retrievers[("西安", "attractions")] is None
