"""Moved to ``bituslabs_ds.metrics.user_stats_aggregates``; import from there.

Shim kept so the legacy Dash app (and its frozen production image) stays
rebuildable until the package is deleted post-bake.
"""

from bituslabs_ds.metrics.user_stats_aggregates import *  # noqa: F401,F403
from bituslabs_ds.metrics.user_stats_aggregates import _bootstrap_ci  # noqa: F401
