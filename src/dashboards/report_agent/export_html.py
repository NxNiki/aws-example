"""Moved to ``bituslabs_ds.confluence.export_html``; import from there.

Shim kept so the legacy Dash app (and its frozen production image) stays
rebuildable until the package is deleted post-bake.
"""

from bituslabs_ds.confluence.export_html import *  # noqa: F401,F403
