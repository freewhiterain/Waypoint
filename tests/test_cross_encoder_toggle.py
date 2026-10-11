from app.rag.local_knowledge import LocalKnowledgeService
from app.config import settings
from app.rag.reranker import CrossEncoderReranker, RelevanceReranker


def test_default_reranker_is_lexical_and_does_not_load_cross_encoder(monkeypatch):
    monkeypatch.setattr(settings, "enable_cross_encoder_rerank", False)

    reranker = LocalKnowledgeService._select_reranker()

    assert type(reranker) is RelevanceReranker


def test_toggle_enabled_selects_cross_encoder_reranker_without_downloading_model(monkeypatch):
    monkeypatch.setattr(settings, "enable_cross_encoder_rerank", True)
    monkeypatch.setattr(settings, "cross_encoder_model", "BAAI/bge-reranker-base")

    reranker = LocalKnowledgeService._select_reranker()

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "BAAI/bge-reranker-base"
    assert reranker._model is None
