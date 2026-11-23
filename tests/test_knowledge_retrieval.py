"""Exercise corpus-wide retrieval, fusion, token boundaries and artifact integrity."""
import json
import re

import numpy as np
import pytest

from fintech_support.knowledge import HandbookRetriever


class WordTokenizer:
    def encode(self, text, **kwargs):
        return re.findall(r"\S+", text)

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


class Encoder:
    tokenizer = WordTokenizer()

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            row = np.zeros(384)
            row[0] = 1 if any(word in text.lower() for word in ("transfer", "recipient", "sent", "wire")) else 0
            row[1] = 1 if "card" in text.lower() else 0
            row[2] = 0.01
            rows.append(row / np.linalg.norm(row))
        return np.asarray(rows)


class Reranker:
    def predict(self, pairs, **kwargs):
        return np.asarray([8 if "recipient" in passage.lower() else -1 for _, passage in pairs])


def handbook(tmp_path, *, long=False):
    path = tmp_path / "handbook.json"
    articles = [
        {"id": "KB-CARD-01", "title": "Card declined", "topic": "card_payments",
         "content": "A card can be declined because it is frozen. Check the card status.",
         "source": "Fictional handbook / cards"},
        {"id": "KB-TRANSFER-01", "title": "Transfer tracking", "topic": "transfers",
         "content": "A transfer is pending while checks are in progress. Check recipient details.",
         "source": "Fictional handbook / transfers"},
    ]
    if long:
        articles[0]["content"] = " ".join(f"word{i}" for i in range(450))
    for article in articles:
        article.update(version="1.0.0", effective_date="2026-11-19", related_ids=[])
    path.write_text(json.dumps({"version": "1.0.0", "brand": "Demo", "synthetic_only": True,
                                "articles": articles}))
    return path


def test_lexical_retrieves_across_topics_without_classifier_filter(tmp_path):
    retriever = HandbookRetriever.from_path(handbook(tmp_path), backend="lexical")
    hits = retriever.search("recipient transfer tracking")
    assert hits[0].article_id == "KB-TRANSFER-01"
    assert hits[0].source == "Fictional handbook / transfers"
    assert hits[0].id != hits[0].article_id
    assert retriever.degraded is True
    assert retriever.version == "1.0.0"
    assert retriever.search("blorple zyxwvu") == []


def test_semantic_retrieval_recovers_no_keyword_match_and_reranks(tmp_path):
    retriever = HandbookRetriever.from_path(handbook(tmp_path), encoder=Encoder(), reranker=Reranker())
    hits = retriever.search("wire")
    assert hits[0].article_id == "KB-TRANSFER-01"
    assert hits[0].score == 8
    assert retriever.degraded is False
    assert retriever.search("card", min_relevance_score=9) == []


def test_reranker_can_reorder_fused_candidates(tmp_path):
    retriever = HandbookRetriever.from_path(handbook(tmp_path), encoder=Encoder(), reranker=Reranker())
    # Lexical and dense ranking both prefer the card; the cross encoder changes the order.
    assert retriever.search("card")[0].article_id == "KB-TRANSFER-01"


def test_chunk_limit_includes_heading_and_preserves_overlap(tmp_path):
    retriever = HandbookRetriever.from_path(handbook(tmp_path, long=True), backend="lexical",
                                            tokenizer=WordTokenizer())
    chunks = [chunk for chunk in retriever.chunks if chunk.article_id == "KB-CARD-01"]
    assert len(chunks) == 3
    assert all(len(WordTokenizer().encode(chunk.text)) <= 180 for chunk in chunks)
    assert "word149" in chunks[0].text and "word149" in chunks[1].text
    assert len({chunk.id for chunk in chunks}) == 3


def test_index_roundtrip_reuses_embeddings_and_checks_identity(tmp_path):
    source = handbook(tmp_path)
    retriever = HandbookRetriever.from_path(source, encoder=Encoder(), reranker=Reranker())
    index = tmp_path / "index.json"
    retriever.save_index(index)
    restored = HandbookRetriever.from_path(source, index_path=index, encoder=Encoder(), reranker=Reranker())
    assert restored.index_identity == retriever.index_identity
    assert restored.search("wire") == retriever.search("wire")
    document = json.loads(source.read_text())
    document["articles"][0]["content"] += " Policy changed."
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="identity"):
        HandbookRetriever.from_path(source, index_path=index, encoder=Encoder(), reranker=Reranker())


def test_index_rejects_corruption_and_non_finite_scores(tmp_path):
    source = handbook(tmp_path)
    retriever = HandbookRetriever.from_path(source, encoder=Encoder(), reranker=Reranker())
    index = tmp_path / "index.json"
    retriever.save_index(index)
    document = json.loads(index.read_text())
    document["embeddings"][0][0] += .2
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checksum"):
        HandbookRetriever.from_path(source, index_path=index, encoder=Encoder(), reranker=Reranker())
    class BadReranker:
        def predict(self, pairs, **kwargs):
            return np.full(len(pairs), float("nan"))
    bad = HandbookRetriever.from_path(source, encoder=Encoder(), reranker=BadReranker())
    with pytest.raises(ValueError, match="finite"):
        bad.search("card")


def test_invalid_inputs_fail_without_model_download(tmp_path):
    source = handbook(tmp_path)
    with pytest.raises(ValueError, match="backend"):
        HandbookRetriever.from_path(source, backend="other")
    retriever = HandbookRetriever.from_path(source, backend="lexical")
    with pytest.raises(ValueError):
        retriever.search(" ")
    with pytest.raises(ValueError):
        retriever.search("card", top_k=0)
    with pytest.raises(FileNotFoundError):
        HandbookRetriever.from_path(source, backend="lexical", index_path=tmp_path / "absent.json")
