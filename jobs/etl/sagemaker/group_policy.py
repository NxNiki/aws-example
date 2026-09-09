"""User-group policy CONFIG for every game — pure data, no logic.

Each policy declares HOW A GROUP IS DEFINED: an ordered list of branches
(first match wins), each naming the group label, the database column it
reads, the values/match rule, and the effective UTC range.
``group_policy_sql.py`` compiles these declarations into SQL; ETL scripts
carry no policy parameters or per-game logic.

Branch fields: label / column / op ("in" | "prefix" | "contains" |
"last_digit_in") / values / optional from-to (UTC, [from, to)).
Policy fields: ``default`` (label when nothing matches); ``fold`` (stored
per-bet label remapping for games without their own AB arms);
``collapse: True`` (ONE label per user-day — a bet in a higher group claims
the user's whole session-day; priority = the order labels first appear in
``branches``).

Only SS03 and fish_hunter define their own branches/cutovers; the other
slot games read the stored per-bet label from slot_orders_ab_group.
Kept as a JSON-shaped dict in a .py file because the SageMaker container
receives code via ``--py-files`` only. Reference docs:
docs/ab_group_policy.md, docs/fish_group_tag_policy.md.
"""

# partition_ab is stored as binary JSON (b'["<group-id>"]'); this expression
# extracts the first element (bind it to the "partition_ab_label" column
# when reading the raw warehouse; derived datasets store it as a column).
PARTITION_AB_FIRST = "get_json_object(CAST(t.partition_ab AS STRING), '$[0]')"

# Digit-policy start used by the stored per-bet label of ALL five slot games
# (empirically located; see docs/ab_group_policy.md).
AB_GROUP_DIGIT_POLICY_START_UTC = "2026-08-03 21:30:00"
# The game team's ANNOUNCED start — SS03's dashboard uses this instead.
SS03_AB_GROUP_ANNOUNCED_START_UTC = "2026-08-03 23:00:00"
# Personalized-retention (个性化挽留) launch, fish_hunter.
FISH_RETENTION_POLICY_START_UTC = "2026-07-31 00:00:00"

_AI_ID = "jojpin-9mokha-rexQug"
_A_ID = "4f1a46ca-7baa-4452-9a40-ef21d9b33b57"
_B_ID = "4a04df21-c749-4808-8e55-3a0b74c084d2"

GROUP_POLICY = {
    # The stored row-level ab_group written into slot_orders_ab_group for all
    # five games.
    "SLOT_ORDERS": {
        "column_name": "ab_group",
        "default": "Default",
        "branches": [
            {
                "label": "AI",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [8, 9],
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
                "label": "AB_TEST_B",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [6, 7],
                "from": AB_GROUP_DIGIT_POLICY_START_UTC,
            },
            {
                "label": "Default",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [0, 1, 2, 3],
                "from": AB_GROUP_DIGIT_POLICY_START_UTC,
            },
            {
                "label": "AI",
                "column": "partition_ab_label",
                "op": "in",
                "values": [_AI_ID],
                "to": AB_GROUP_DIGIT_POLICY_START_UTC,
            },
            {
                "label": "AB_TEST_A",
                "column": "partition_ab_label",
                "op": "in",
                "values": [_A_ID],
                "to": AB_GROUP_DIGIT_POLICY_START_UTC,
            },
            {
                "label": "AB_TEST_B",
                "column": "partition_ab_label",
                "op": "in",
                "values": [_B_ID],
                "to": AB_GROUP_DIGIT_POLICY_START_UTC,
            },
        ],
    },
    # Games reading the stored per-bet label; without their own AB arms the
    # test labels fold into Default.
    "SS01": {"column_name": "ab_group", "fold": {"AB_TEST_A": "Default", "AB_TEST_B": "Default"}},
    "SS01A": {"column_name": "ab_group", "fold": {"AB_TEST_A": "Default", "AB_TEST_B": "Default"}},
    "SS02": {"column_name": "ab_group", "fold": {"AB_TEST_A": "Default", "AB_TEST_B": "Default"}},
    "SS06": {"column_name": "ab_group"},
    # SS03's dashboard re-derives the label under the announced cutover and
    # keeps ONE label per user-day.
    "SS03": {
        "column_name": "ab_group",
        "default": "Default",
        "collapse": True,
        "branches": [
            {
                "label": "AI",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [8, 9],
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
                "label": "AB_TEST_B",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [6, 7],
                "from": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
            {
                "label": "Default",
                "column": "user_id",
                "op": "last_digit_in",
                "values": [0, 1, 2, 3],
                "from": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
            {
                "label": "AI",
                "column": "partition_ab_label",
                "op": "in",
                "values": [_AI_ID],
                "to": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
            {
                "label": "AB_TEST_A",
                "column": "partition_ab_label",
                "op": "in",
                "values": [_A_ID],
                "to": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
            {
                "label": "AB_TEST_B",
                "column": "partition_ab_label",
                "op": "in",
                "values": [_B_ID],
                "to": SS03_AB_GROUP_ANNOUNCED_START_UTC,
            },
        ],
    },
    # fish_hunter group_tag, derived per bullet into fish_bullets_group_tag.
    # Strategy branches apply to ALL history; CR_FISHING_* is the retention
    # treatment itself; only the digit rule is gated on the launch.
    "FM01": {
        "column_name": "group_tag",
        "default": "default",
        "collapse": True,
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
    },
}
