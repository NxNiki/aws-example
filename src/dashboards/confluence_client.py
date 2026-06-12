"""Moved to ``bituslabs_ds.confluence.client``; import from there.

Shim kept so the legacy Dash app (and its frozen production image) stays
rebuildable until the package is deleted post-bake.
"""

from bituslabs_ds.confluence.client import *  # noqa: F401,F403
from bituslabs_ds.confluence.client import _get_client, _strip_html  # noqa: F401
