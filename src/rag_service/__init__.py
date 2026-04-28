"""
rag_service — isolated RAG microservice for Confluence documentation.

The service indexes Confluence pages declared in ``config/rag_sources.yaml``
into an OpenSearch cluster (local Docker for dev, OpenSearch Serverless on
AWS) and exposes a hybrid (BM25 + vector) retrieval HTTP endpoint that the
AI chat agent calls.

Public surface (used by the chat agent):
    from rag_service.client import retrieve_passages
"""
