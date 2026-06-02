"""Per-game configuration for the unified feature engineering pipeline.

A ``GameFeatureConfig`` captures every knob that differs across SS01 /
SS02 / SS03 today. The SQL CTE builders consume the config; per-game
job scripts under ``jobs/ss0x/`` construct one and hand it to
``FeaturePipelineRunner``.

Adding a new game ideally requires only a new ``GameFeatureConfig``
instance and a thin wrapper script — no new SQL.
"""

import json
import math
from dataclasses import asdict, dataclass

from bituslabs_ds.config import AB_TEST_GROUP_A, AB_TEST_GROUP_B, AI_GROUP_ID

# Map from human-readable ai_group label -> partition_ab[0] id in fct_bet_orders.
# 'Default' has no id; it is the catch-all bucket for rows whose partition_ab[0]
# is NULL or unmapped.
AI_GROUP_PARTITION_IDS: dict[str, str] = {
    "AI": AI_GROUP_ID,
    "AB_TEST_A": AB_TEST_GROUP_A,
    "AB_TEST_B": AB_TEST_GROUP_B,
}

_KNOWN_AI_GROUP_LABELS = set(AI_GROUP_PARTITION_IDS) | {"Default"}

# Fields whose value determines what already-written rows *mean*. If any of
# these changes, existing rows on S3 are incompatible with newly-computed rows
# (e.g. agg_group buckets of 30 vs 100 bets), so an incremental append would
# silently corrupt the dataset. The runner refuses to append on drift in these
# fields and requires --overwrite. Cosmetic fields (date_start, output_prefix,
# lookback_days) are deliberately excluded: changing them does not invalidate
# rows already written.
SEMANTIC_FIELDS: tuple[str, ...] = (
    "game_id",
    "ai_groups",
    "selected_groups",
    "partition_cols",
    "session_length",
    "max_session_interval_seconds",
    "streak_threshold_seconds",
    "max_session_gap_seconds",
    "drop_incomplete_tail_groups",
    "extra_where_clauses",
)


@dataclass(frozen=True)
class GameFeatureConfig:
    """Per-game knobs for the feature engineering pipeline.

    Required fields:
        game_id: e.g. ``"SS02"``. Inserted into the ``game_id = '...'`` WHERE clause.
        output_prefix: last path segment under ``DEFAULT_ETL_OUTPUT/jobs/``,
            e.g. ``"output_ss02_feature_engineer"``. The runner appends
            ``features_enriched/`` and ``features_grouped/`` for the two datasets.
        date_start: ``YYYY-MM-DD``; used as ``ETLScheduler.default_start_date``.
        date_end: ``YYYY-MM-DD``; exclusive upper bound on bet timestamps.
        ai_groups: tuple of labels the ``ai_group`` column can take for this
            game. Every game uses the same column name; for games with no AB
            test partitions, set to ``("AI", "Default")`` and rely on
            ``selected_groups`` + the partition_ab filter to drop AI rows.
        selected_groups: subset of ``ai_groups`` to include via the WHERE filter.

    Optional fields:
        partition_cols: extra GROUP BY / aggregation key columns beyond the
            baseline ``(user_id, ai_group, session_group, agg_group)``.
            ``("math_table_id",)`` for ss02/ss03; ``()`` for ss01.
        session_length: rows per agg_group (``ROUND((rn - 1) / N)``).
        max_session_interval_seconds: gap threshold that splits a user's bets
            into new session_groups.
        streak_threshold_seconds: gap threshold used by streak / win_streak /
            lose_streak window functions.
        max_session_gap_seconds: ``delta_t_seconds_nogap`` excludes gaps above
            this value (defaults to 1 hour).
        drop_incomplete_tail_groups: when ``True``, ``stats_base`` adds
            ``HAVING COUNT(user_id) = session_length`` so the final partial
            ``agg_group`` per session is dropped.
        extra_where_clauses: additional AND-conjuncted WHERE clauses appended to
            ``user_bets`` (e.g. ``("t.script_id = 'giftShop'",)`` for ss01).
        requires_full_history: when ``True``, the runner emits a warning if
            ``--overwrite`` was not passed. Set this when the SQL semantics
            change (new columns, threshold updates) and existing data on S3
            needs a full recomputation. With stable session IDs +
            ``effective_lookback_days()`` the normal incremental run is safe,
            so the default is ``False``.
        lookback_days: explicit override for ETLScheduler lookback. ``None`` ->
            derive from ``max_session_interval_seconds`` via
            ``effective_lookback_days()``.
    """

    game_id: str
    output_prefix: str
    date_start: str
    date_end: str
    ai_groups: tuple[str, ...]
    selected_groups: tuple[str, ...]
    partition_cols: tuple[str, ...] = ()
    session_length: int = 100
    max_session_interval_seconds: int = 60 * 60 * 24 * 7
    streak_threshold_seconds: int = 200
    max_session_gap_seconds: int = 60 * 60
    drop_incomplete_tail_groups: bool = False
    extra_where_clauses: tuple[str, ...] = ()
    requires_full_history: bool = False
    # Explicit override for ETLScheduler lookback. ``None`` -> derive from
    # ``max_session_interval_seconds`` via ``effective_lookback_days()``.
    lookback_days: int | None = None

    def __post_init__(self) -> None:
        if not self.ai_groups:
            raise ValueError("ai_groups must contain at least one label")
        unknown_ai = set(self.ai_groups) - _KNOWN_AI_GROUP_LABELS
        if unknown_ai:
            raise ValueError(
                f"ai_groups contains unknown labels: {sorted(unknown_ai)}. "
                f"Allowed: {sorted(_KNOWN_AI_GROUP_LABELS)}."
            )
        unknown_sel = set(self.selected_groups) - set(self.ai_groups)
        if unknown_sel:
            raise ValueError(
                f"selected_groups contains labels not in ai_groups: {sorted(unknown_sel)}. "
                f"ai_groups: {self.ai_groups}."
            )

    def effective_lookback_days(self) -> int:
        """ETLScheduler lookback derived from ``max_session_interval_seconds``.

        Formula: ``ceil(max_session_interval_seconds / 86400) + 1`` -- one full
        ``max_session_interval`` plus a one-day safety buffer. Examples:

        * 12-hour ``max_session_interval_seconds`` -> 2 days
        * 7-day ``max_session_interval_seconds`` -> 8 days

        Any session that started within the lookback window has its first bet
        captured by the query, so ``session_start_ts`` (and therefore
        ``session_start_date`` + ``session_group``) is computed correctly.
        Sessions that began *before* the lookback window are not re-queried;
        their existing rows in S3 stay untouched.

        ``lookback_days`` on the config overrides this if explicitly set
        (e.g. for users known to bet continuously beyond the formula's buffer).
        """
        if self.lookback_days is not None:
            return self.lookback_days
        return math.ceil(self.max_session_interval_seconds / 86400) + 1

    def to_dict(self) -> dict:
        """Full config as a JSON-serializable dict (tuples become lists)."""
        return json.loads(json.dumps(asdict(self)))

    def semantic_signature(self) -> dict:
        """Subset of the config (``SEMANTIC_FIELDS``) that, if changed, makes
        already-written rows incompatible with new rows. Used by the runner's
        drift guard to decide whether an incremental append is safe.
        """
        full = self.to_dict()
        return {field: full[field] for field in SEMANTIC_FIELDS}
