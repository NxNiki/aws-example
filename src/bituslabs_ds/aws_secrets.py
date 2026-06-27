"""
Lightweight secret lookup shared by the chat agent, RAG service, and
Confluence client.

Resolution order: process env var → AWS Secrets Manager (cached for the
process lifetime). Lives in its own module — without dragging
``langchain_core`` (which ``ai_agent.chat_agent`` imports at module
top) — so the slim RAG-service Docker image can use it without
shipping the LangChain stack.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_SECRETS_MANAGER_NAME = "ai-dashboard_ai_agent"
_SECRETS_MANAGER_REGION = "us-west-2"

_secrets_cache: Optional[Dict[str, str]] = None


def get_secret(key: str) -> Optional[str]:
    """Return an API key/secret from env var first, then AWS Secrets Manager.

    Secrets Manager results are cached per-process; subsequent calls are
    free. A failed lookup is also cached (as an empty dict) so we don't
    hammer the API for an unconfigured environment.
    """
    value = os.environ.get(key)
    if value:
        return value

    global _secrets_cache
    if _secrets_cache is None:
        try:
            import json

            import boto3  # type: ignore[import-untyped]

            session = boto3.Session(region_name=_SECRETS_MANAGER_REGION)
            client = session.client(service_name="secretsmanager")
            resp = client.get_secret_value(SecretId=_SECRETS_MANAGER_NAME)
            _secrets_cache = json.loads(resp["SecretString"])
        except Exception as exc:
            logger.warning("Secrets Manager lookup failed: %s", exc)
            _secrets_cache = {}

    return (_secrets_cache or {}).get(key)
