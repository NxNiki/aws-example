"""Shared SQL-emission helpers used across CTE builders.

Conventions:
- Every helper returns a string (possibly empty), never a list of lines.
- Indentation is "spaces, no tabs"; the caller controls the leading indent.
- Helpers that emit multi-line fragments do *not* end with a trailing newline.
"""

from bituslabs_ds.features.config import AI_GROUP_PARTITION_IDS, GameFeatureConfig

# Always-present aggregation key columns. ``cfg.partition_cols`` extends this.
BASE_PARTITION_COLS: tuple[str, ...] = (
    "user_id",
    "ai_group",
    "session_start_date",
    "session_group",
    "agg_group",
)


def all_partition_cols(cfg: GameFeatureConfig) -> tuple[str, ...]:
    """Full aggregation key for stats_base / percentile CTEs / final join.

    Order: ``user_id``, ``ai_group``, ``cfg.partition_cols``, then
    ``session_start_date``, ``session_group``, ``agg_group``. ``session_start_date``
    is the stable per-session identifier (date of the session's first bet);
    ``session_group`` is a within-(user, session_start_date) ordinal so multiple
    sessions on the same calendar date stay distinct.
    """
    return (
        "user_id",
        "ai_group",
        *cfg.partition_cols,
        "session_start_date",
        "session_group",
        "agg_group",
    )


def select_lines(cols: tuple[str, ...], alias: str = "t", indent: str = "    ") -> str:
    """Render ``{alias}.{col}`` lines, comma-separated, each prefixed with ``indent``.

    No leading or trailing newline. Caller embeds with the right surrounding
    context.
    """
    return "\n".join(f"{indent}{alias}.{c}," for c in cols)


def equi_join_on(left: str, right: str, cols: tuple[str, ...], indent: str = "        ") -> str:
    """Render ``ON {left}.col = {right}.col`` joined by ``AND``."""
    parts = [f"{left}.{c} = {right}.{c}" for c in cols]
    glue = f"\n{indent}    AND "
    return f"{indent}ON " + glue.join(parts)


def ai_group_case(cfg: GameFeatureConfig, alias: str = "t", indent: str = "                ") -> str:
    """SQL ``CASE`` expression labeling the ``ai_group`` column.

    Emits one ``WHEN`` per non-Default label in ``cfg.ai_groups``, with an
    ``ELSE 'Default'`` catch-all. If ``ai_groups`` contains only ``"Default"``
    the CASE collapses to the constant ``'Default' AS ai_group``.
    """
    non_default = [lbl for lbl in cfg.ai_groups if lbl != "Default"]
    if not non_default:
        return "'Default' AS ai_group"
    branches = "\n".join(
        f"{indent}    WHEN {alias}.partition_ab[0] = '{AI_GROUP_PARTITION_IDS[lbl]}' THEN '{lbl}'"
        for lbl in non_default
    )
    return f"CASE\n" f"{branches}\n" f"{indent}    ELSE 'Default'\n" f"{indent}END AS ai_group"


def ai_group_filter(cfg: GameFeatureConfig, alias: str = "t") -> str:
    """SQL ``AND (...)`` fragment restricting to ``cfg.selected_groups``.

    Returns ``""`` when ``selected_groups`` already covers every label in
    ``ai_groups`` (no filter needed). The fragment is a single line with no
    leading indent.
    """
    if set(cfg.selected_groups) >= set(cfg.ai_groups):
        return ""

    non_default_ids = [AI_GROUP_PARTITION_IDS[lbl] for lbl in cfg.ai_groups if lbl != "Default"]
    conditions: list[str] = []
    for label in cfg.selected_groups:
        if label == "Default":
            if not non_default_ids:
                conditions.append("TRUE")
            else:
                in_list = ", ".join(f"'{gid}'" for gid in non_default_ids)
                conditions.append(f"({alias}.partition_ab[0] IS NULL OR {alias}.partition_ab[0] NOT IN ({in_list}))")
        else:
            conditions.append(f"{alias}.partition_ab[0] = '{AI_GROUP_PARTITION_IDS[label]}'")
    return "AND (" + " OR ".join(conditions) + ")"
