"""Health-check endpoint for load balancers and uptime checks."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict[str, str]:
    """API endpoint: GET /api/health — ALB/uptime health check for the dashboard_api service."""
    return {"status": "ok", "service": "dashboard_api"}
