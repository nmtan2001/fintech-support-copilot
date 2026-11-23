"""Versioned corpus-wide BM25, dense retrieval and CPU cross-encoder reranking.

Lexical mode is an explicit offline fallback. Hybrid scores are raw reranker
logits, not calibrated probabilities or proof that an answer is supported.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "retrieval-manifest.json"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _checksum(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class _WordTokenizer:
    """Offline fallback; its identity never claims embedding-token boundaries."""
    name_or_path = "offline-word-v1"

    def encode(self, text, **kwargs):
        return re.findall(r"\S+", text)

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


@dataclass(frozen=True)
class Evidence:
    id: str
    article_id: str
    title: str
    text: str
    source: str
    topic: str
    score: float = 0.0


class _BM25:
    def __init__(self, documents: list[str]):
        self.rows = [Counter(_words(document)) for document in documents]
        self.lengths = [sum(row.values()) for row in self.rows]
        self.average_length = sum(self.lengths) / max(1, len(self.rows))
        self.frequencies = Counter(word for row in self.rows for word in row)

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.rows))
        for word in set(_words(query)):
            count = self.frequencies[word]
            if not count:
                continue
            inverse_frequency = math.log(1 + (len(self.rows) - count + .5) / (count + .5))
            for index, row in enumerate(self.rows):
                frequency = row[word]
                if frequency:
                    norm = 1.2 * (.25 + .75 * self.lengths[index] / max(1, self.average_length))
                    result[index] += inverse_frequency * frequency * 2.2 / (frequency + norm)
        return result


def _verified_model(spec: dict) -> None:
    """Verify the pinned public safetensors artifact before model construction."""
    from huggingface_hub import hf_hub_download
    file = hf_hub_download(repo_id=spec["model_id"], revision=spec["revision"],
                           filename=spec["weights_file"], token=False)
    digest = hashlib.sha256()
    with open(file, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != spec["weights_sha256"]:
        raise ValueError("Retrieval model weight checksum mismatch")


def _load_encoder(spec):
    from sentence_transformers import SentenceTransformer
    _verified_model(spec)
    return SentenceTransformer(spec["model_id"], revision=spec["revision"], device="cpu",
                               trust_remote_code=False, model_kwargs={"use_safetensors": True})


def _load_reranker(spec):
    from sentence_transformers import CrossEncoder
    from torch.nn import Identity
    _verified_model(spec)
    return CrossEncoder(spec["model_id"], revision=spec["revision"], device="cpu",
                        trust_remote_code=False, max_length=512, activation_fn=Identity(),
                        model_kwargs={"use_safetensors": True})


def _chunks(articles: list[dict], tokenizer, settings: dict) -> tuple[Evidence, ...]:
    chunks = []
    maximum = settings["max_tokens"]
    overlap = settings["overlap_tokens"]
    for article in articles:
        heading = article["title"].strip() + "\n"
        budget = maximum - len(tokenizer.encode(heading, add_special_tokens=False))
        if budget <= overlap:
            raise ValueError("Handbook title leaves insufficient chunk token budget")
        body = article["content"]
        offsets = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        offsets = [(int(start), int(end)) for start, end in offsets if end > start]
        for number, start in enumerate(range(0, len(offsets), budget - overlap)):
            window = offsets[start:start + budget]
            text = heading + body[window[0][0]:window[-1][1]]
            # Joining heading and content can change subword segmentation slightly.
            while len(tokenizer.encode(text, add_special_tokens=False)) > maximum and len(window) > 1:
                window = window[:-1]
                text = heading + body[window[0][0]:window[-1][1]]
            if len(tokenizer.encode(text, add_special_tokens=False)) > maximum:
                raise ValueError("Chunk cannot fit the configured token budget")
            fingerprint = hashlib.sha256(text.encode()).hexdigest()[:12]
            chunks.append(Evidence(f"{article['id']}#chunk-{number:03d}-{fingerprint}",
                                   article["id"], article["title"], text,
                                   article["source"], article["topic"]))
            if start + len(window) >= len(offsets):
                break
    return tuple(chunks)


class HandbookRetriever:
    @classmethod
    def from_path(cls, path: Path, *, backend="hybrid", index_path=None,
                  model_manifest_path=None, encoder=None, reranker=None, tokenizer=None):
        if backend not in {"hybrid", "lexical"}:
            raise ValueError("Unknown handbook retrieval backend")
        if index_path is not None and not Path(index_path).is_file():
            raise FileNotFoundError(index_path)
        source = Path(path).read_bytes()
        document = json.loads(source)
        if document.get("synthetic_only") is not True or not document.get("version"):
            raise ValueError("Handbook must declare synthetic provenance and version")
        articles = document.get("articles")
        if not isinstance(articles, list) or not articles:
            raise ValueError("Handbook requires articles")
        seen = set()
        for article in articles:
            if not isinstance(article, dict) or not all(
                isinstance(article.get(key), str) and article[key].strip()
                for key in ("id", "title", "content", "source", "topic")
            ) or article["id"] in seen:
                raise ValueError("Invalid or duplicate handbook article")
            seen.add(article["id"])
        manifest = json.loads(Path(model_manifest_path or DEFAULT_MANIFEST).read_text())
        instance = cls()
        instance.backend = backend
        instance.degraded = backend == "lexical"
        instance.version = document["version"]
        instance.manifest = manifest
        instance.encoder = encoder if encoder is not None else (
            _load_encoder(manifest["embedding"]) if backend == "hybrid" else None)
        tokenizer = tokenizer or getattr(instance.encoder, "tokenizer", None) or _WordTokenizer()
        instance.chunks = _chunks(articles, tokenizer, manifest["chunker"])
        if not instance.chunks:
            raise ValueError("Handbook has no searchable chunks")
        instance.index_identity = {
            "schema_version": 1, "source_sha256": hashlib.sha256(source).hexdigest(),
            "knowledge_version": instance.version, "backend": backend,
            "tokenizer_type": type(tokenizer).__name__,
            "chunker": manifest["chunker"],
            "embedding": manifest["embedding"] if backend == "hybrid" else None,
            "reranker": manifest["reranker"] if backend == "hybrid" else None,
            "retrieval": manifest["retrieval"],
        }
        instance.index_id = _checksum(instance.index_identity)
        instance.lexical = _BM25([chunk.text for chunk in instance.chunks])
        instance.embeddings = None
        if index_path is not None:
            instance._load_index(Path(index_path))
        elif backend == "hybrid":
            instance.embeddings = instance._encode([chunk.text for chunk in instance.chunks])
        instance.reranker = reranker if reranker is not None else (
            _load_reranker(manifest["reranker"]) if backend == "hybrid" else None)
        return instance

    def _encode(self, texts):
        matrix = np.asarray(self.encoder.encode(texts, normalize_embeddings=True,
                                               convert_to_numpy=True, show_progress_bar=False), dtype=float)
        if matrix.shape != (len(texts), self.manifest["embedding"]["dimension"]) or not np.isfinite(matrix).all():
            raise ValueError("Embedding dimensions or finite values are invalid")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Embedding norms must be positive")
        return matrix / norms

    def save_index(self, path):
        payload = {"identity": self.index_identity, "chunks": [asdict(chunk) for chunk in self.chunks],
                   "embeddings": self.embeddings.tolist() if self.embeddings is not None else None}
        payload["checksum"] = _checksum(payload)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    def _load_index(self, path):
        payload = json.loads(path.read_text())
        checksum = payload.pop("checksum", None)
        try:
            actual = _checksum(payload)
        except ValueError as exc:
            raise ValueError("Index checksum contains invalid numeric values") from exc
        if checksum != actual:
            raise ValueError("Index checksum mismatch")
        if payload.get("identity") != self.index_identity:
            raise ValueError("Index identity does not match corpus/model/chunker")
        if payload.get("chunks") != [asdict(chunk) for chunk in self.chunks]:
            raise ValueError("Index chunks do not match source identity")
        if self.backend == "hybrid":
            self.embeddings = np.asarray(payload.get("embeddings"), dtype=float)
            expected = (len(self.chunks), self.manifest["embedding"]["dimension"])
            if self.embeddings.shape != expected or not np.isfinite(self.embeddings).all():
                raise ValueError("Index embedding dimensions or finite values are invalid")
            if not np.allclose(np.linalg.norm(self.embeddings, axis=1), 1, atol=1e-4):
                raise ValueError("Index embeddings must be normalized")
        elif payload.get("embeddings") is not None:
            raise ValueError("Lexical index cannot contain dense embeddings")

    def search(self, question: str, *, top_k=5, min_relevance_score=None) -> list[Evidence]:
        if not question.strip() or len(question) > 8000:
            raise ValueError("Search question must contain 1 to 8000 characters")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if min_relevance_score is not None and not math.isfinite(min_relevance_score):
            raise ValueError("Relevance threshold must be finite")
        settings = self.manifest["retrieval"]
        lexical_scores = self.lexical.scores(question)
        def order(scores):
            return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.chunks[i].id))
        lexical = [i for i in order(lexical_scores) if lexical_scores[i] > 0][:settings["lexical_candidates"]]
        if self.backend == "lexical":
            return [replace(self.chunks[i], score=float(lexical_scores[i])) for i in lexical[:top_k]]
        dense_scores = (self._encode([question]) @ self.embeddings.T).ravel()
        dense = order(dense_scores)[:settings["dense_candidates"]]
        fused = {}
        for ranking in (lexical, dense):
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0) + 1 / (settings["rrf_k"] + rank)
        candidates = sorted(fused, key=lambda i: (-fused[i], self.chunks[i].id))[:settings["rerank_candidates"]]
        scores = np.asarray(self.reranker.predict([(question, self.chunks[i].text) for i in candidates],
                                                 show_progress_bar=False), dtype=float).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError("Reranker returned invalid or non-finite scores")
        threshold = min_relevance_score if min_relevance_score is not None else self.manifest.get("relevance_threshold")
        ranked = sorted(zip(candidates, scores), key=lambda pair: (-float(pair[1]), self.chunks[pair[0]].id))
        return [replace(self.chunks[index], score=float(score)) for index, score in ranked
                if threshold is None or score >= threshold][:top_k]
