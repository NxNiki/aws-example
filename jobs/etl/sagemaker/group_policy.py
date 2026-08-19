"""User-group policy for EVERY game, in one place (slots + fish_hunter).

Single source of truth for how bets/bullets are assigned to dashboard
cohorts: the id/digit cutovers, the per-game label rules, and the user-day
collapse priorities. Jobs import the builders below and never hardcode
group logic. Reference docs: ``docs/ab_group_policy.md`` (slots) and
``docs/fish_group_tag_policy.md`` (fish_hunter).

Ships to the Spark container via ``submit_py_files`` next to
``spark_etl_common.py``; keep it dependency-free.
"""

# --------------------------------------------------------------------------
# Slot machines: ab_group
# --------------------------------------------------------------------------

AI_GROUP_ID = "jojpin-9mokha-rexQug"
AB_TEST_GROUP_A = "4f1a46ca-7baa-4452-9a40-ef21d9b33b57"
AB_TEST_GROUP_B = "4a04df21-c749-4808-8e55-3a0b74c084d2"

# The cold data stores partition_ab as binary JSON (b'["<group-id>"]'), not a
# parquet list, so the first element is extracted via get_json_object.
PARTITION_AB_FIRST = "get_json_object(CAST(t.partition_ab AS STRING), '$[0]')"

# Digit-policy cutovers: the shared orders dataset uses the EMPIRICAL serving
# flip; SS03's dashboard uses the game team's ANNOUNCED start (product
# decision — the ~1.5h in between keeps stale partition_ab labels).
AB_GROUP_DIGIT_POLICY_START_BJ = "2026-08-04 05:30:00"
SS03_AB_GROUP_ANNOUNCED_START_UTC = "2026-08-03 23:00:00"

# Per-game policy. ab_tests: whether the game HAS AB test arms (without them
# the A/B labels fold into Default). user_day_groups: ONE label per user-day
# (re-derived under the announced cutover; a single higher-tier bet claims
# the whole session-day) instead of the stored per-bet label — SS03's
# dashboard policy; the run/cohort variants opt out in the job.
SLOT_GROUP_POLICY = {
    "SS01": {"ab_tests": False, "user_day_groups": False},
    "SS01A": {"ab_tests": False, "user_day_groups": False},
    "SS02": {"ab_tests": False, "user_day_groups": False},
    "SS03": {"ab_tests": True, "user_day_groups": True},
    "SS06": {"ab_tests": True, "user_day_groups": False},
}

# User-day collapse priority (highest first; 'Default' is the implicit last).
SLOT_DAY_GROUP_PRIORITY = ("AI", "AB_TEST_A", "AB_TEST_B")


def ab_group_sql(ab_label_expr: str, user_id_expr: str, bj_ts_expr: str, include_ab_tests: bool = True) -> str:
    """CASE labeling a bet's AB group under the EMPIRICAL timestamp-gated
    policy (the slot_orders_ab_group dataset's stored label).

    ``ab_label_expr`` is the first partition_ab id (e.g. PARTITION_AB_FIRST),
    ``user_id_expr`` the numeric user id, ``bj_ts_expr`` a TIMESTAMP in
    Beijing time. Games without AB test groups (include_ab_tests=False)
    collapse the test digits / legacy test ids into Default."""
    digit = f"CAST({user_id_expr} AS BIGINT) % 10"
    if include_ab_tests:
        digit_case = f"""CASE
                WHEN {digit} >= 8 THEN 'AI'
                WHEN {digit} >= 6 THEN 'AB_TEST_B'
                WHEN {digit} >= 4 THEN 'AB_TEST_A'
                ELSE 'Default'
            END"""
        legacy_tests = f"""
                WHEN {ab_label_expr} = '{AB_TEST_GROUP_A}' THEN 'AB_TEST_A'
                WHEN {ab_label_expr} = '{AB_TEST_GROUP_B}' THEN 'AB_TEST_B'"""
    else:
        digit_case = f"CASE WHEN {digit} >= 8 THEN 'AI' ELSE 'Default' END"
        legacy_tests = ""
    return f"""CASE
            WHEN {bj_ts_expr} >= TIMESTAMP '{AB_GROUP_DIGIT_POLICY_START_BJ}' THEN {digit_case}
            ELSE CASE
                WHEN {ab_label_expr} = '{AI_GROUP_ID}' THEN 'AI'{legacy_tests}
                ELSE 'Default'
            END
        END"""


def ss03_bet_ab_group_sql(ab_label_expr: str, user_id_expr: str, utc_ts_expr: str) -> str:
    """Row-level SS03 dashboard AB label under the ANNOUNCED cutover.

    ``utc_ts_expr`` must be a UTC TIMESTAMP. New era: user-id last digit
    (0-3 Default, 4-5 A, 6-7 B, 8-9 AI). Old era: partition_ab[0] id mapping
    (4f1a... = A serves the shi-family tables that digit-4/5 users continue,
    4a04... = B the BGadj_v3 family). The caller collapses via
    ``slot_day_group_case``."""
    digit = f"CAST({user_id_expr} AS BIGINT) % 10"
    return f"""CASE
            WHEN {utc_ts_expr} >= TIMESTAMP '{SS03_AB_GROUP_ANNOUNCED_START_UTC}' THEN CASE
                WHEN {digit} >= 8 THEN 'AI'
                WHEN {digit} >= 6 THEN 'AB_TEST_B'
                WHEN {digit} >= 4 THEN 'AB_TEST_A'
                ELSE 'Default'
            END
            WHEN {ab_label_expr} = '{AI_GROUP_ID}' THEN 'AI'
            WHEN {ab_label_expr} = '{AB_TEST_GROUP_A}' THEN 'AB_TEST_A'
            WHEN {ab_label_expr} = '{AB_TEST_GROUP_B}' THEN 'AB_TEST_B'
            ELSE 'Default'
        END"""


def slot_stored_label_case(game_id: str) -> str:
    """The stored slot_orders_ab_group label, folded to the game's group
    vocabulary: games without AB test groups fold AB_TEST_A/B into Default."""
    if SLOT_GROUP_POLICY[game_id]["ab_tests"]:
        return "t.ab_group"
    return "CASE WHEN t.ab_group IN ('AB_TEST_A', 'AB_TEST_B') THEN 'Default' ELSE t.ab_group END AS ab_group"


def slot_day_group_case(label_expr: str) -> str:
    """One AB label per (user, activity_date): a single bet in a higher tier
    claims the whole user-day, priority AI > AB_TEST_A > AB_TEST_B > Default
    (old-era assignment was per-bet; new-era digit labels collapse as a
    no-op)."""
    day = "PARTITION BY t.user_id, t.activity_date"
    branches = "\n                ".join(
        f"WHEN MAX(CASE WHEN {label_expr} = '{g}' THEN 1 ELSE 0 END) OVER ({day}) = 1 THEN '{g}'"
        for g in SLOT_DAY_GROUP_PRIORITY
    )
    return f"""CASE
                {branches}
                ELSE 'Default'
            END"""


# --------------------------------------------------------------------------
# fish_hunter (FM01): group_tag
# --------------------------------------------------------------------------

# Personalized-retention (个性化挽留) launch: 2026-07-30 16:00 PST. Only the
# digit rule is gated on it; strategy branches apply to all history.
FISH_RETENTION_POLICY_START_UTC = "2026-07-31 00:00:00"

# Branch order of fish_group_tag_sql, which doubles as the user-day collapse
# priority ('default' is the implicit last tier). boost_pool outranks
# everything for legacy data; the strategy no longer exists in new data.
FISH_GROUP_TAG_PRIORITY = ("boost_pool", "dynamic_rtp", "risk_control", "retention")


def fish_group_tag_sql(strategy_expr: str, user_id_expr: str, utc_ts_expr: str) -> str:
    """CASE expression labeling a single bullet's ``group_tag``.

    ``utc_ts_expr`` must be a UTC TIMESTAMP (the retention gate is specified
    in UTC). The ``substr`` tests are the escaped ``LIKE '.._FISHING\\_%'``
    (literal underscore); ``'%RISK_CONTROL%'`` also matches the legacy
    ``RISK_CONTROLLED`` strategy, and the whole DYNAMIC_RTP family shares
    the ``dynamic_rtp`` label. CR_FISHING_* is the personalized-retention
    treatment itself (served exclusively to digit-0/1 users, including a
    small canary in the hour before the official launch), so it labels
    ``retention`` regardless of the gate."""
    return f"""CASE
            WHEN {strategy_expr} = 'BOOST_POOL' THEN 'boost_pool'
            WHEN {strategy_expr} IN ('DYNAMIC_RTP', 'DYNAMIC_RTP_V2', 'DYNAMIC_RTP_V3') THEN 'dynamic_rtp'
            WHEN substr({strategy_expr}, 1, 11) = 'RC_FISHING_' THEN 'risk_control'
            WHEN {strategy_expr} LIKE '%RISK_CONTROL%' THEN 'risk_control'
            WHEN substr({strategy_expr}, 1, 11) = 'CR_FISHING_' THEN 'retention'
            WHEN {utc_ts_expr} >= TIMESTAMP '{FISH_RETENTION_POLICY_START_UTC}'
                AND CAST({user_id_expr} AS BIGINT) % 10 IN (0, 1) THEN 'retention'
            ELSE 'default'
        END"""


def fish_group_tag_day_case(tag_expr: str) -> str:
    """Collapse a user-day's row-level tags to ONE label (use inside a
    GROUP BY user, day aggregate). A single bullet in a higher tier claims
    the whole user-day; priority is the row CASE's branch order."""
    branches = "\n            ".join(
        f"WHEN MAX(CASE WHEN {tag_expr} = '{tag}' THEN 1 ELSE 0 END) > 0 THEN '{tag}'"
        for tag in FISH_GROUP_TAG_PRIORITY
    )
    return f"""CASE
            {branches}
            ELSE 'default'
        END"""
