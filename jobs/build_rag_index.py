"""
CLI entry point — build (or rebuild) the Confluence RAG OpenSearch index
from src/rag_service/config/rag_sources.yaml.

Local dev (Docker OpenSearch):
    docker compose -f infra/docker-compose.opensearch.yml up -d
    poetry run python jobs/build_rag_index.py

Production (ECS / scheduled task):
    OPENSEARCH_HOST=...aoss.amazonaws.com OPENSEARCH_AUTH=aws \\
    poetry run python jobs/build_rag_index.py

Flags:
    --drop          Delete the index before rebuilding (use when the schema
                    has changed; otherwise re-runs upsert in place).
    --sources PATH  Override RAG_SOURCES_PATH for one-off ad-hoc runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("build_rag_index")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop", action="store_true", help="Delete the index first")
    parser.add_argument("--sources", type=str, default=None, help="Path to rag_sources.yaml")
    args = parser.parse_args()

    if args.sources:
        os.environ["RAG_SOURCES_PATH"] = args.sources

    from rag_service.config import load_settings
    from rag_service.indexer import build_index, delete_index

    settings = load_settings()
    logger.info(
        "Indexing → host=%s index=%s sources=%d",
        settings.opensearch_host,
        settings.index_name,
        len(settings.sources),
    )

    if args.drop:
        delete_index(settings)

    counts = build_index(settings)
    logger.info("Done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
