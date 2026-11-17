"""Small, local FAQ retriever for supported demo intents."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class FAQ:
    id: str
    title: str
    content: str
    intents: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class Hit:
    faq: FAQ
    score: float
    tfidf_score: float | None = None
    semantic_score: float | None = None


def load_faqs(path: Path) -> list[FAQ]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("FAQ knowledge base must contain a nonempty list")
    faqs = []
    ids = set()
    for item in raw:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ("id", "title", "content", "source")
        ):
            raise ValueError("FAQ entries require id, title, content, source")
        intents = item.get("intents")
        if not isinstance(intents, list) or not intents or not all(
            isinstance(intent, str) and intent.strip() for intent in intents
        ):
            raise ValueError("FAQ entries require a nonempty intent list")
        if item["id"] in ids:
            raise ValueError(f"Duplicate FAQ id: {item['id']}")
        ids.add(item["id"])
        faqs.append(FAQ(item["id"], item["title"], item["content"], tuple(intents), item["source"]))
    return faqs


class FAQRetriever:
    SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RRF_K = 60
    HYBRID_SEMANTIC_FLOOR = 0.35

    def __init__(self, faqs: list[FAQ], *, backend: str = "tfidf"):
        if not faqs:
            raise ValueError("At least one FAQ entry is required")
        if backend not in {"tfidf", "semantic", "hybrid"}:
            raise ValueError(f"Unknown retrieval backend: {backend}")
        self.faqs = faqs
        self.backend = backend
        texts = [f"{faq.title}. {faq.content}" for faq in faqs]
        self.tfidf_encoder = None
        self.tfidf_embeddings = None
        self.semantic_encoder = None
        self.semantic_embeddings = None
        if backend in {"tfidf", "hybrid"}:
            self.tfidf_encoder = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.tfidf_embeddings = self.tfidf_encoder.fit_transform(texts)
        if backend in {"semantic", "hybrid"}:
            from sentence_transformers import SentenceTransformer

            self.semantic_encoder = SentenceTransformer(self.SEMANTIC_MODEL)
            self.semantic_embeddings = self.semantic_encoder.encode(texts, normalize_embeddings=True)

    def search(self, question: str, *, intent: str, top_k: int = 3) -> list[Hit]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        allowed = [i for i, faq in enumerate(self.faqs) if intent in faq.intents]
        if not allowed:
            return []

        tfidf_scores = None
        semantic_scores = None
        if self.tfidf_encoder is not None:
            query = self.tfidf_encoder.transform([question])
            tfidf_scores = cosine_similarity(query, self.tfidf_embeddings[allowed]).ravel()
        if self.semantic_encoder is not None:
            query = self.semantic_encoder.encode([question], normalize_embeddings=True)
            semantic_scores = (query @ self.semantic_embeddings[allowed].T).ravel()

        if self.backend == "hybrid":
            ranked_tfidf = sorted(
                range(len(allowed)),
                key=lambda index: (-float(tfidf_scores[index]), self.faqs[allowed[index]].id),
            )
            ranked_semantic = sorted(
                range(len(allowed)),
                key=lambda index: (-float(semantic_scores[index]), self.faqs[allowed[index]].id),
            )
            tfidf_ranks = {index: rank for rank, index in enumerate(ranked_tfidf, start=1)}
            semantic_ranks = {index: rank for rank, index in enumerate(ranked_semantic, start=1)}
            candidates = [
                index for index in range(len(allowed))
                if (
                    np.isfinite(tfidf_scores[index]) and tfidf_scores[index] > 0
                ) or (
                    np.isfinite(semantic_scores[index])
                    and semantic_scores[index] >= self.HYBRID_SEMANTIC_FLOOR
                )
            ]
            fusion_scores = {}
            for index in candidates:
                lexical_score = float(tfidf_scores[index])
                semantic_score = float(semantic_scores[index])
                lexical_contribution = (
                    1 / (self.RRF_K + tfidf_ranks[index]) if lexical_score > 0 else 0
                )
                semantic_contribution = (
                    1 / (self.RRF_K + semantic_ranks[index])
                    if semantic_score >= self.HYBRID_SEMANTIC_FLOOR else 0
                )
                fusion_scores[index] = lexical_contribution + semantic_contribution
            ranked = sorted(
                candidates,
                key=lambda index: (-fusion_scores[index], self.faqs[allowed[index]].id),
            )
            return [
                Hit(
                    self.faqs[allowed[index]],
                    fusion_scores[index],
                    float(tfidf_scores[index]),
                    float(semantic_scores[index]),
                )
                for index in ranked[:top_k]
            ]

        scores = tfidf_scores if self.backend == "tfidf" else semantic_scores
        ranked = sorted(
            zip(allowed, scores),
            key=lambda pair: (-float(pair[1]), self.faqs[pair[0]].id),
        )
        return [
            Hit(
                self.faqs[index],
                float(score),
                float(score) if self.backend == "tfidf" else None,
                float(score) if self.backend == "semantic" else None,
            )
            for index, score in ranked[:top_k]
            if np.isfinite(score) and score > 0
        ]
