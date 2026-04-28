"""
OpenSearch connection factory.

Same client code path for:
- local Docker OpenSearch (basic auth, self-signed certs)
- AWS OpenSearch Serverless (SigV4)
- AWS OpenSearch managed (SigV4 with service='es')

Switch via env vars only — see ``rag_service.config.RagSettings``.
"""

from __future__ import annotations

import logging
from typing import Optional

from opensearchpy import OpenSearch, RequestsHttpConnection

from rag_service.config import RagSettings, load_settings

logger = logging.getLogger(__name__)

_CLIENT: Optional[OpenSearch] = None


def get_client(settings: Optional[RagSettings] = None) -> OpenSearch:
    """Return a singleton OpenSearch client built from env-driven settings."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    cfg = settings or load_settings()

    if cfg.opensearch_auth_mode == "aws":
        import boto3
        from opensearchpy import AWSV4SignerAuth

        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise RuntimeError("AWS credentials not found for OpenSearch SigV4 auth.")
        auth = AWSV4SignerAuth(credentials, cfg.aws_region, cfg.aws_service)
        verify_certs = True
    else:
        if not cfg.opensearch_password:
            raise ValueError(
                "OPENSEARCH_PASSWORD is required for basic auth. " "Set it in .env or your task definition."
            )
        auth = (cfg.opensearch_user or "admin", cfg.opensearch_password)
        # Local Docker uses self-signed certs; skip verification in dev.
        verify_certs = False

    logger.info(
        "Connecting to OpenSearch %s:%d (auth=%s, ssl=%s)",
        cfg.opensearch_host,
        cfg.opensearch_port,
        cfg.opensearch_auth_mode,
        cfg.opensearch_use_ssl,
    )

    _CLIENT = OpenSearch(
        hosts=[{"host": cfg.opensearch_host, "port": cfg.opensearch_port}],
        http_auth=auth,
        use_ssl=cfg.opensearch_use_ssl,
        verify_certs=verify_certs,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )
    return _CLIENT


def index_body(embedding_dim: int) -> dict:
    """Index mapping with both BM25 text and a knn_vector field."""
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "page_id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "url": {"type": "keyword"},
                "space_key": {"type": "keyword"},
                "source_name": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "updated_at": {"type": "date"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                        "parameters": {"ef_construction": 256, "m": 16},
                    },
                },
            }
        },
    }


def ensure_index(client: OpenSearch, index_name: str, embedding_dim: int) -> None:
    """Create the index if it doesn't exist."""
    if not client.indices.exists(index_name):
        client.indices.create(index_name, body=index_body(embedding_dim))
        logger.info("Created OpenSearch index: %s", index_name)
    else:
        logger.info("OpenSearch index already exists: %s", index_name)
