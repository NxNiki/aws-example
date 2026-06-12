"""Confluence integration shared by dashboard_api, ai_agent, and rag_service.

Credentials resolve via env vars → AWS Secrets Manager (see
``bituslabs_ds.aws_secrets``); the heavyweight atlassian-python-api import is
lazy, so this package is safe to ship in images without the ``confluence``
poetry group as long as those code paths aren't exercised.
"""
