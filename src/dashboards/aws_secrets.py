"""Moved to ``bituslabs_ds.aws_secrets``; import from there.

Shim kept so the legacy Dash app (and its frozen production image) stays
rebuildable until the package is deleted post-bake.
"""

from bituslabs_ds.aws_secrets import *  # noqa: F401,F403
