"""User-group policy CONFIG for every game — pure data, no logic.

Declares, per policy key, HOW A GROUP IS DEFINED: an ordered list of
branches (first match wins), each naming the group label, the database
column it reads, the values/match rule, and the effective UTC date range.
``group_policy_sql.py`` compiles these declarations into SQL; ETL scripts
carry no policy parameters or per-game logic.

Kept as a JSON-shaped dict in a .py file (not .json/.yaml) because the
SageMaker Spark container receives code via ``--py-files`` only — a .py
config imports with zero parsing and allows comments. Do not add functions
here. Reference docs: docs/ab_group_policy.md, docs/fish_group_tag_policy.md.

Branch fields:
  label   -- the group label this branch assigns
  column  -- logical database column the rule reads (the SQL builder binds
             it to a concrete expression per job)
  op      -- "in" | "prefix" | "contains" | "last_digit_in"
  values  -- list the rule matches against
  from/to -- effective UTC timestamp range [from, to); omitted = unbounded

Policy fields:
  default           -- label when no branch matches
  fold              -- stored-label remapping (games without their own arms)
  collapse_priority -- ONE label per (user, session-day): a single bet in a
                       higher-priority group claims the whole user-day;
                       omitted = the stored/derived label stays per-bet
"""

# partition_ab is stored as binary JSON (b'["<group-id>"]'); this expression
# extracts the first element (bind it to the "partition_ab_label" column
# when reading the raw warehouse; derived datasets store it as a column).
PARTITION_AB_FIRST = "get_json_object(CAST(t.partition_ab AS STRING), '$[0]')"

_AI_ID = "jojpin-9mokha-rexQug"
_A_ID = "4f1a46ca-7baa-4452-9a40-ef21d9b33b57"  # serves the shi-family tables (continues into digits 4-5)
_B_ID = "4a04df21-c749-4808-8e55-3a0b74c084d2"  # serves the BGadj_v3 family (continues into digits 6-7)

# Empirical digit-policy serving flip: 2026-08-04 05:30 Beijing.
AB_GROUP_DIGIT_POLICY_START_UTC = "2026-08-03 21:30:00"
# The game team's ANNOUNCED start (2026-08-03 16:00 PDT) — SS03's dashboard
# uses this instead; the ~1.5h in between keeps stale partition_ab labels.
SS03_AB_GROUP_ANNOUNCED_START_UTC = "2026-08-03 23:00:00"
# Personalized-retention (个性化挽留) launch, fish_hunter.
FISH_RETENTION_POLICY_START_UTC = "2026-07-31 00:00:00"

_DIGIT_BRANCHES_EMPIRICAL = [
    {
        "label": "AI",
        "column": "user_id",
        "op": "last_digit_in",
        "values": [8, 9],
        "from": AB_GROUP_DIGIT_POLICY_START_UTC,
    },
    {
        "label": "AB_TEST_B",
        "column": "user_id",
        "op": "last_digit_in",
        "values": [6, 7],
        "from": AB_GROUP_DIGIT_POLICY_START_UTC,
    },
    {
        "label": "AB_TEST_A",
        "column": "user_id",
        "op": "last_digit_in",
        "values": [4, 5],
        "from": AB_GROUP_DIGIT_POLICY_START_UTC,
    },
    {
        "label": "Default",
        "column": "user_id",
        "op": "last_digit_in",
        "values": [0, 1, 2, 3],
        "from": AB_GROUP_DIGIT_POLICY_START_UTC,
    },
]

_ID_BRANCHES = [
    {"label": "AI", "column": "partition_ab_label", "op": "in", "values": [_AI_ID]},
    {"label": "AB_TEST_A", "column": "partition_ab_label", "op": "in", "values": [_A_ID]},
    {"label": "AB_TEST_B", "column": "partition_ab_label", "op": "in", "values": [_B_ID]},
]

GROUP_POLICY = {
    # The stored row-level ab_group written into slot_orders_ab_group for all
    # five games (empirical cutover; per-game folding happens in the stats
    # policies below).
    "SLOT_ORDERS": {
        "column_name": "ab_group",
        "default": "Default",
        "branches": _DIGIT_BRANCHES_EMPIRICAL + [{**b, "to": AB_GROUP_DIGIT_POLICY_START_UTC} for b in _ID_BRANCHES],
    },
    # Stats-job policies: games without their own AB arms fold the test
    # labels into Default; SS03's dashboard re-derives per bet under the
    # announced cutover and collapses to one label per user-day.
    "SS01": {"column_name": "ab_group", "source": "stored", "fold": {"AB_TEST_A": "Default", "AB_TEST_B": "Default"}},
    "SS01A": {"column_name": "ab_group", "source": "stored", "fold": {"AB_TEST_A": "Default", "AB_TEST_B": "Default"}},
    "SS02": {"column_name": "ab_group", "source": "stored", "fold": {"AB_TEST_A": "Default", "AB_TEST_B": "Default"}},
    "SS06": {"column_name": "ab_group", "source": "stored"},
    "SS03": {
        "column_name": "ab_group",
        "default": "Default",
        "branches": [
            {
                "label": "AI",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [8, 9],
                "from": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
            {
                "label": "AB_TEST_B",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [6, 7],
                "from": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
            {
                "label": "AB_TEST_A",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [4, 5],
                "from": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
            {
                "label": "Default",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [0, 1, 2, 3],
                "from": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
            *[{**b, "to": SS03_AB_GROUP_ANNOUNCED_START_UTC} for b in _ID_BRANCHES],
        ],
        "collapse_priority": ["AI", "AB_TEST_A", "AB_TEST_B"],
    },
    # fish_hunter group_tag, derived per bullet into fish_bullets_group_tag.
    # Strategy branches apply to ALL history (risk_control predates the
    # retention launch); CR_FISHING_* is the retention treatment itself
    # (100% digit-0/1 users incl. the pre-launch canary); only the digit
    # rule is gated on the launch.
    "FM01": {
        "column_name": "group_tag",
        "default": "default",
        "branches": [
            {"label": "boost_pool", "column": "strategy_name", "op": "in", "values": ["BOOST_POOL"]},
            {
                "label": "dynamic_rtp",
                "column": "strategy_name",
                "op": "in",
                "values": ["DYNAMIC_RTP", "DYNAMIC_RTP_V2", "DYNAMIC_RTP_V3"],
            },
            {"label": "risk_control", "column": "strategy_name", "op": "prefix", "values": ["RC_FISHING_"]},
            {"label": "risk_control", "column": "strategy_name", "op": "contains", "values": ["RISK_CONTROL"]},
            {"label": "retention", "column": "strategy_name", "op": "prefix", "values": ["CR_FISHING_"]},
            {
                "label": "retention",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [0, 1],
                "from": FISH_RETENTION_POLICY_START_UTC,
            },
        ],
        "collapse_priority": ["boost_pool", "dynamic_rtp", "risk_control", "retention"],
    },
}
