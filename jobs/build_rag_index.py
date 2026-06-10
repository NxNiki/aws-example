"""
CLI entry point — build (or rebuild) the Confluence RAG index from
src/rag_service/config/rag_sources.yaml.

Default backend is Faiss; the artifact is written to
``bituslabs_ds.config.DEFAULT_RAG_INDEX_URI``. Set ``RAG_BACKEND=opensearch``
to write to an OpenSearch cluster instead.

    poetry run python jobs/build_rag_index.py

Flags:
    --drop          OpenSearch only: delete the index before rebuilding.
                    Faiss rebuilds always replace the artifact in place.
    --sources PATH  Override RAG_SOURCES_PATH for one-off ad-hoc runs.
    --refresh-ecs-service
                    After a successful Faiss build+upload, force a new
                    rag-service deployment so it reloads the fresh index.
                    Used by the daily scheduled reindex task; see
                    infra/rag_service/setup_reindex_schedule.sh.
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


def _force_service_redeploy() -> None:
    """Force a rolling restart of the rag-service so it serves the new index.

    The FastAPI service loads the Faiss index from S3 once at startup and keeps
    it cached in memory (``rag_service.retriever``); a freshly uploaded S3
    artifact is NOT served until the task restarts. ``forceNewDeployment`` does
    a rolling restart (with ``desiredCount=1`` + rolling deploy this is
    near-zero downtime) so the replacement task loads the updated index.

    Cluster/service come from ``ECS_CLUSTER`` / ``RAG_SERVICE_NAME`` env vars
    (defaults: ``ai-team-dashboard-cluster`` / ``rag-service``).
    """
    import boto3

    from bituslabs_ds.config import REGION

    cluster = os.environ.get("ECS_CLUSTER", "ai-team-dashboard-cluster")
    service = os.environ.get("RAG_SERVICE_NAME", "rag-service")
    logger.info("Forcing new deployment of ECS service %s/%s to reload the index...", cluster, service)
    boto3.client("ecs", region_name=REGION).update_service(cluster=cluster, service=service, forceNewDeployment=True)
    logger.info("force-new-deployment requested for %s/%s", cluster, service)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop", action="store_true", help="Delete the index first")
    parser.add_argument("--sources", type=str, default=None, help="Path to rag_sources.yaml")
    parser.add_argument(
        "--refresh-ecs-service",
        action="store_true",
        help="After a successful Faiss build, force-new-deployment the rag-service so it reloads the index.",
    )
    args = parser.parse_args()

    if args.sources:
        os.environ["RAG_SOURCES_PATH"] = args.sources

    from rag_service.config import load_settings
    from rag_service.indexer import build_index, delete_index

    settings = load_settings()
    if settings.backend == "faiss":
        logger.info(
            "Indexing → backend=faiss uri=%s sources=%d",
            settings.index_uri,
            len(settings.sources),
        )
    else:
        logger.info(
            "Indexing → backend=opensearch host=%s index=%s sources=%d",
            settings.opensearch_host,
            settings.index_name,
            len(settings.sources),
        )

    if args.drop:
        delete_index(settings)

    counts = build_index(settings)
    logger.info("Done: %s", counts)

    if args.refresh_ecs_service:
        if settings.backend != "faiss":
            logger.warning("--refresh-ecs-service ignored: backend is %s, not faiss", settings.backend)
        else:
            # Index is already on S3 at this point; the redeploy only makes the
            # running service pick it up. A redeploy failure is surfaced (non-zero
            # exit) but does not undo the successful S3 build above.
            _force_service_redeploy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
