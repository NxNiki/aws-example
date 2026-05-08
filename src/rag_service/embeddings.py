"""
Embedding model factory.

Two providers are supported and selected via env vars:

    RAG_EMBEDDING_PROVIDER = openai | gemini | auto   (default: auto)
    RAG_EMBEDDING_MODEL    = override the default model name for the chosen
                             provider (optional).
    RAG_EMBEDDING_DIM      = vector dimension. Defaults match the chosen
                             provider's default model. Must match the
                             OpenSearch ``knn_vector`` mapping.

Defaults per provider:
    openai → text-embedding-3-small (1536 dim)
    gemini → gemini-embedding-001   (768 dim via Matryoshka truncation)

Auto-selection: if no provider is set, prefer OpenAI (broader ecosystem
support) and fall back to Gemini when only ``GOOGLE_API_KEY`` /
``GEMINI_API_KEY`` is present. API keys are loaded via the chat agent's
secret loader (env > AWS Secrets Manager).

Important: changing providers (or models with different dimensions) requires
a clean reindex — run ``python jobs/build_rag_index.py --drop`` because the
OpenSearch ``knn_vector`` mapping is fixed at index creation time.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from rag_service.config import RagSettings, load_settings

logger = logging.getLogger(__name__)


_PROVIDER_DEFAULTS = {
    "openai": {"model": "text-embedding-3-small", "dim": 1536},
    "gemini": {"model": "gemini-embedding-001", "dim": 768},
}


def _resolve_provider() -> str:
    """Return 'openai' or 'gemini' based on env preference and available keys."""
    from dashboards.chat_agent import _get_secret

    raw = (os.environ.get("RAG_EMBEDDING_PROVIDER") or "auto").lower()
    if raw in ("openai", "gemini"):
        return raw
    if raw != "auto":
        logger.warning("Unknown RAG_EMBEDDING_PROVIDER=%r; falling back to auto", raw)

    if _get_secret("OPENAI_API_KEY"):
        return "openai"
    if _get_secret("GOOGLE_API_KEY") or _get_secret("GEMINI_API_KEY"):
        return "gemini"
    raise ValueError(
        "No embedding provider available — set OPENAI_API_KEY or GOOGLE_API_KEY/"
        "GEMINI_API_KEY (or pin RAG_EMBEDDING_PROVIDER explicitly)."
    )


def _provider_defaults(provider: str) -> dict:
    return _PROVIDER_DEFAULTS[provider]


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


class _OpenAIEmbedder:
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        from dashboards.chat_agent import _get_secret

        api_key = _get_secret("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not available for OpenAI embeddings.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out


class _GeminiEmbedder:
    """Gemini embeddings via the official ``google-generativeai`` SDK.

    The SDK only embeds one input per call — we batch in Python for
    consistency with the OpenAI path.
    """

    def __init__(self, model: str, dim: int) -> None:
        from dashboards.chat_agent import _get_secret

        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for Gemini embeddings. "
                "Install with: poetry add --group rag_service google-generativeai"
            ) from exc

        api_key = _get_secret("GOOGLE_API_KEY") or _get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY/GEMINI_API_KEY not available for Gemini embeddings.")
        genai.configure(api_key=api_key)
        # Gemini model names should be prefixed with "models/" if the caller forgot.
        self._model = model if model.startswith("models/") else f"models/{model}"
        # gemini-embedding-001 returns 3072 dims by default; use Matryoshka
        # truncation to match our pinned dimension.
        self._dim = dim
        self._genai = genai

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        # batch_size accepted for parity; SDK is one-at-a-time per call.
        del batch_size
        out: List[List[float]] = []
        for text in texts:
            resp = self._genai.embed_content(
                model=self._model,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=self._dim,
            )
            out.append(resp["embedding"])
        return out

    def embed_query_one(self, text: str) -> List[float]:
        resp = self._genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=self._dim,
        )
        return resp["embedding"]


# ---------------------------------------------------------------------------
# Public Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """Provider-agnostic embedder. Resolves provider/model/dim lazily."""

    def __init__(self, settings: Optional[RagSettings] = None) -> None:
        self.settings = settings or load_settings()
        self._provider: Optional[str] = None
        self._impl = None

    def _ensure_impl(self):
        if self._impl is not None:
            return self._impl

        provider = (self.settings.embedding_provider or "auto").lower()
        if provider == "auto":
            provider = _resolve_provider()
        if provider not in _PROVIDER_DEFAULTS:
            raise ValueError(f"Unsupported RAG_EMBEDDING_PROVIDER: {provider!r}")

        defaults = _provider_defaults(provider)
        model = self.settings.embedding_model or defaults["model"]

        # Soft check: warn (don't crash) if dim looks wrong for the chosen model.
        if self.settings.embedding_dim != defaults["dim"]:
            logger.warning(
                "RAG_EMBEDDING_DIM=%d does not match %s/%s default (%d). "
                "Make sure your OpenSearch mapping was built with the same dim.",
                self.settings.embedding_dim,
                provider,
                model,
                defaults["dim"],
            )

        if provider == "openai":
            self._impl = _OpenAIEmbedder(model)
        else:
            self._impl = _GeminiEmbedder(model, dim=self.settings.embedding_dim)

        self._provider = provider
        logger.info("Embedder ready: provider=%s model=%s dim=%d", provider, model, self.settings.embedding_dim)
        return self._impl

    @property
    def provider(self) -> str:
        self._ensure_impl()
        return self._provider or ""

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        impl = self._ensure_impl()
        return impl.embed(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> List[float]:
        impl = self._ensure_impl()
        # Gemini exposes a query-specific task type — use it when available.
        if hasattr(impl, "embed_query_one"):
            return impl.embed_query_one(text)
        return impl.embed([text])[0]


def resolved_embedding_dim(settings: Optional[RagSettings] = None) -> int:
    """Return the dimension that *will* be used, after auto-resolution.

    Used by ``opensearch_client.ensure_index`` so the mapping matches the
    embedder when the user left ``RAG_EMBEDDING_DIM`` at its default.
    """
    cfg = settings or load_settings()
    provider = (cfg.embedding_provider or "auto").lower()
    if provider == "auto":
        provider = _resolve_provider()
    if cfg.embedding_dim_explicit:
        return cfg.embedding_dim
    return _provider_defaults(provider)["dim"]
