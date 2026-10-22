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
    def __init__(self, faqs: list[FAQ], *, backend: str = "tfidf"):
        if not faqs:
            raise ValueError("At least one FAQ entry is required")
        self.faqs = faqs
        self.backend = backend
        texts = [f"{faq.title}. {faq.content}" for faq in faqs]
        if backend == "tfidf":
            self.encoder = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.embeddings = self.encoder.fit_transform(texts)
        elif backend == "semantic":
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError(f"Unknown retrieval backend: {backend}")

    def search(self, question: str, *, intent: str, top_k: int = 3) -> list[Hit]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        allowed = [i for i, faq in enumerate(self.faqs) if intent in faq.intents]
        if not allowed:
            return []
        if self.backend == "tfidf":
            query = self.encoder.transform([question])
            scores = cosine_similarity(query, self.embeddings[allowed]).ravel()
        else:
            query = self.encoder.encode([question], normalize_embeddings=True)
            scores = (query @ self.embeddings[allowed].T).ravel()
        ranked = sorted(zip(allowed, scores), key=lambda pair: (-float(pair[1]), self.faqs[pair[0]].id))
        return [
            Hit(self.faqs[i], float(score))
            for i, score in ranked[:top_k]
            if np.isfinite(score) and score > 0
        ]
