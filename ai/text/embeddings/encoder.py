"""Sentence-transformer embedding engine with lexical fallback."""

from __future__ import annotations

import time
from typing import Any

from ai.models.cache import timed_load
from ai.models.config import Phase4ModelConfig
from ai.text.semantic.exemplars import EXEMPLARS
from ai.text.semantic.similarity import cosine, jaccard, tokenize


class SentenceTransformerEngine:
    """EmbeddingEngine protocol — lazy SentenceTransformer or lexical vectors."""

    name = "sentence_transformer_engine"
    version = "0.4.0"

    def __init__(self, config: Phase4ModelConfig | None = None) -> None:
        self.config = config or Phase4ModelConfig.from_env()
        self._backend: str | None = None
        self._load_ms = 0.0
        self._exemplar_embeddings: dict[str, list[list[float]]] | None = None

    def _resolve_backend(self) -> str:
        if self._backend:
            return self._backend
        if self.config.stub_mode and self.config.backend == "auto":
            self._backend = "stub"
            return self._backend
        if self.config.backend == "lexical":
            self._backend = "lexical"
            return self._backend
        if self.config.backend in ("pretrained", "auto") and not self.config.stub_mode:
            try:
                import sentence_transformers  # noqa: F401

                self._backend = "pretrained"
                return self._backend
            except ImportError:
                if self.config.backend == "pretrained":
                    self._backend = "unavailable"
                    return self._backend
                self._backend = "lexical"
                return self._backend
        self._backend = "lexical" if not self.config.stub_mode else "stub"
        return self._backend

    def _get_st_model(self):
        cfg = self.config
        key = f"st:{cfg.text_model_name}:{cfg.device}"

        def factory():
            from sentence_transformers import SentenceTransformer

            cache = str(cfg.cache_dir) if cfg.cache_models else None
            if cache:
                cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            model = SentenceTransformer(
                cfg.text_model_name,
                cache_folder=cache,
                device=cfg.device,
            )
            return model

        model, load_ms, _ = timed_load(key, factory)
        self._load_ms = load_ms
        return model

    def embed(self, texts: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        backend = self._resolve_backend()
        if backend == "stub":
            return {
                "status": "not_loaded",
                "backend": "stub",
                "embeddings": [],
                "message": "AI_STUB_MODE enabled — embeddings not loaded.",
                "inference_ms": 0.0,
                "load_ms": 0.0,
            }
        if backend == "unavailable":
            return {
                "status": "error",
                "backend": "unavailable",
                "embeddings": [],
                "message": "sentence-transformers not installed.",
                "inference_ms": 0.0,
                "load_ms": 0.0,
            }
        if backend == "pretrained":
            model = self._get_st_model()
            vectors = model.encode(
                texts,
                batch_size=self.config.embedding_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            emb = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
            return {
                "status": "ready",
                "backend": "pretrained",
                "model": self.config.text_model_name,
                "embeddings": emb,
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "load_ms": round(self._load_ms, 2),
            }
        # lexical bag-of-char trigrams as sparse-ish dense hash
        emb = [_lexical_vector(t) for t in texts]
        return {
            "status": "ready",
            "backend": "lexical",
            "model": "lexical-jaccard-hash",
            "embeddings": emb,
            "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "load_ms": 0.0,
        }

    def match_exemplars(self, texts: list[str], *, threshold: float = 0.42) -> list[dict[str, Any]]:
        """Return pattern matches with similarity and matched example."""
        backend = self._resolve_backend()
        if backend in ("stub", "unavailable") or not texts:
            return []

        matches: list[dict[str, Any]] = []
        if backend == "lexical":
            for text in texts:
                tokens = tokenize(text)
                best_pattern = None
                best_sim = 0.0
                best_ex = ""
                for pattern, examples in EXEMPLARS.items():
                    for ex in examples:
                        sim = jaccard(tokens, tokenize(ex))
                        if sim > best_sim:
                            best_sim = sim
                            best_pattern = pattern
                            best_ex = ex
                if best_pattern and best_sim >= threshold:
                    matches.append(
                        {
                            "pattern": best_pattern,
                            "confidence": round(best_sim, 4),
                            "embedding_similarity": round(best_sim, 4),
                            "matched_example": best_ex,
                            "source_text": text[:240],
                        }
                    )
            return _dedupe_patterns(matches)

        # pretrained cosine against exemplar embeddings
        flat_examples: list[tuple[str, str]] = []
        for pattern, examples in EXEMPLARS.items():
            for ex in examples:
                flat_examples.append((pattern, ex))
        ex_texts = [e[1] for e in flat_examples]
        query = self.embed(texts)
        bank = self.embed(ex_texts)
        if query.get("status") != "ready" or bank.get("status") != "ready":
            return []
        q_emb = query["embeddings"]
        b_emb = bank["embeddings"]
        for i, text in enumerate(texts):
            best_pattern = None
            best_sim = 0.0
            best_ex = ""
            for j, (pattern, ex) in enumerate(flat_examples):
                sim = cosine(q_emb[i], b_emb[j])
                if sim > best_sim:
                    best_sim = sim
                    best_pattern = pattern
                    best_ex = ex
            if best_pattern and best_sim >= threshold:
                matches.append(
                    {
                        "pattern": best_pattern,
                        "confidence": round(best_sim, 4),
                        "embedding_similarity": round(best_sim, 4),
                        "matched_example": best_ex,
                        "source_text": text[:240],
                    }
                )
        return _dedupe_patterns(matches)

    def is_ready(self) -> bool:
        return self._resolve_backend() not in ("stub", "unavailable")


def _lexical_vector(text: str, dim: int = 64) -> list[float]:
    vec = [0.0] * dim
    toks = list(tokenize(text))
    if not toks:
        return vec
    for tok in toks:
        h = hash(tok) % dim
        vec[h] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm:
        vec = [v / norm for v in vec]
    return vec


def _dedupe_patterns(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for m in matches:
        p = m["pattern"]
        if p not in best or m["confidence"] > best[p]["confidence"]:
            best[p] = m
    return sorted(best.values(), key=lambda x: x["confidence"], reverse=True)
