"""dashboard_api — lightweight FastAPI service backing the React dashboard.

Serves the data + report API the SPA and the agent share, and (in prod)
serves the built SPA static assets. LLM-free by design: agent turns are
handled by the separate `ai_agent` service (see docs/frontend_redesign.md).
"""
