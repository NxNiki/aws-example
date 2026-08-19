"""Compile ``group_policy.GROUP_POLICY`` declarations into SQL.

The config file declares WHAT a group is (label, database column, values,
effective UTC range); this module is the only place that turns those
declarations into SQL expressions. ETL scripts call these builders and
carry no policy parameters. Ships via ``submit_py_files`` next to
``group_policy.py``; keep it dependency-free.
"""

from group_policy import GROUP_POLICY


def _branch_condition(branch: dict, col_exprs: dict, ts_expr: str) -> str:
    col = col_exprs[branch["column"]]
    op, values = branch["op"], branch["values"]
    if op == "in":
        cond = f"{col} IN ({', '.join(repr(v) if isinstance(v, str) else str(v) for v in values)})"
    elif op == "prefix":
        cond = " OR ".join(f"substr({col}, 1, {len(v)}) = '{v}'" for v in values)
    elif op == "contains":
        cond = " OR ".join(f"{col} LIKE '%{v}%'" for v in values)
    elif op == "last_digit_in":
        cond = f"CAST({col} AS BIGINT) % 10 IN ({', '.join(str(v) for v in values)})"
    else:
        raise ValueError(f"unknown op: {op}")
    parts = [f"({cond})" if " OR " in cond else cond]
    if branch.get("from"):
        parts.append(f"{ts_expr} >= TIMESTAMP '{branch['from']}'")
    if branch.get("to"):
        parts.append(f"{ts_expr} < TIMESTAMP '{branch['to']}'")
    return " AND ".join(parts)


def group_label_sql(policy_key: str, col_exprs: dict, ts_expr: str) -> str:
    """Row-level CASE assigning the policy's group label (first matching
    branch wins). ``col_exprs`` binds the config's logical column names to
    concrete SQL expressions; ``ts_expr`` must be a UTC TIMESTAMP (the
    effective ranges are declared in UTC)."""
    policy = GROUP_POLICY[policy_key]
    whens = "\n            ".join(
        f"WHEN {_branch_condition(b, col_exprs, ts_expr)} THEN '{b['label']}'" for b in policy["branches"]
    )
    return f"""CASE
            {whens}
            ELSE '{policy["default"]}'
        END"""


def slot_stored_label_case(policy_key: str, stored_expr: str = "t.ab_group") -> str:
    """The stored per-bet label, remapped through the policy's ``fold`` (games
    without their own AB arms fold the test labels into Default)."""
    policy = GROUP_POLICY[policy_key]
    fold = policy.get("fold")
    if not fold:
        return stored_expr
    by_target: dict = {}
    for src, target in fold.items():
        by_target.setdefault(target, []).append(src)
    whens = " ".join(
        f"WHEN {stored_expr} IN ({', '.join(repr(s) for s in sorted(srcs))}) THEN '{target}'"
        for target, srcs in sorted(by_target.items())
    )
    return f"CASE {whens} ELSE {stored_expr} END AS {policy['column_name']}"


def collapse_case_over(policy_key: str, label_expr: str) -> str:
    """ONE label per (user, activity_date) as a window expression: a single
    bet in a higher-priority group claims the whole user-day."""
    policy = GROUP_POLICY[policy_key]
    day = "PARTITION BY t.user_id, t.activity_date"
    whens = "\n                ".join(
        f"WHEN MAX(CASE WHEN {label_expr} = '{g}' THEN 1 ELSE 0 END) OVER ({day}) = 1 THEN '{g}'"
        for g in policy["collapse_priority"]
    )
    return f"""CASE
                {whens}
                ELSE '{policy["default"]}'
            END"""


def collapse_case_groupby(policy_key: str, label_expr: str) -> str:
    """ONE label per user-day inside a GROUP BY (user, day) aggregate."""
    policy = GROUP_POLICY[policy_key]
    whens = "\n            ".join(
        f"WHEN MAX(CASE WHEN {label_expr} = '{g}' THEN 1 ELSE 0 END) > 0 THEN '{g}'"
        for g in policy["collapse_priority"]
    )
    return f"""CASE
            {whens}
            ELSE '{policy["default"]}'
        END"""


def slot_grouping(game_id: str, row_level: bool = False):
    """A slot game's complete grouping recipe for the stats job.

    Returns ``(source_col, bet_label_case, day_collapse_case)``: when the
    last two are None the stored per-bet label applies (via
    ``slot_stored_label_case``); otherwise each bet is labeled by the game's
    declared branches and every (user, session-day) collapses.
    ``row_level=True`` (the run/cohort variants, i.e. the AI-combo dataset)
    always uses the stored per-bet label."""
    policy = GROUP_POLICY[game_id]
    if row_level or policy.get("source") == "stored" or "collapse_priority" not in policy:
        return "t.ab_group", None, None
    label = group_label_sql(
        game_id,
        {"partition_ab_label": "t.partition_ab_label", "user_id": "t.user_id"},
        "t.created_at",
    )
    return "t.partition_ab_label", label, collapse_case_over(game_id, "t.bet_ab_group")
