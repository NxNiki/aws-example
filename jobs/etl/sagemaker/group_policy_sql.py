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


def _collapse_priority(policy: dict) -> list:
    """Collapse priority = the order labels first appear in ``branches``
    (the default label is the implicit last tier)."""
    seen: list = []
    for b in policy["branches"]:
        if b["label"] != policy["default"] and b["label"] not in seen:
            seen.append(b["label"])
    return seen


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


def _fold_expr(policy: dict, stored_expr: str) -> str:
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
    return f"CASE {whens} ELSE {stored_expr} END"


def collapse_case_over(policy_key: str, label_expr: str, day_expr: str) -> str:
    """ONE label per (user, day) as a window expression: a bet in a
    higher-priority group claims the user's whole day."""
    policy = GROUP_POLICY[policy_key]
    day = f"PARTITION BY t.user_id, {day_expr}"
    whens = "\n                ".join(
        f"WHEN MAX(CASE WHEN {label_expr} = '{g}' THEN 1 ELSE 0 END) OVER ({day}) = 1 THEN '{g}'"
        for g in _collapse_priority(policy)
    )
    return f"""CASE
                {whens}
                ELSE '{policy["default"]}'
            END"""


def collapse_case_groupby(policy_key: str, label_expr: str) -> str:
    """ONE label per user-day inside a GROUP BY (user, day) aggregate."""
    policy = GROUP_POLICY[policy_key]
    whens = "\n            ".join(
        f"WHEN MAX(CASE WHEN {label_expr} = '{g}' THEN 1 ELSE 0 END) > 0 THEN '{g}'" for g in _collapse_priority(policy)
    )
    return f"""CASE
            {whens}
            ELSE '{policy["default"]}'
        END"""


def slot_group_expr(game_id: str, day_expr: str, row_level: bool = False) -> str:
    """The COMPLETE ``ab_group`` expression for a slot stats query — the one
    policy entry point the job interpolates (alias it ``AS ab_group``).

    Games without their own branches read the stored per-bet label (folded
    per their policy). A game declaring branches + ``collapse: True`` gets
    its label re-derived per bet and collapsed to ONE label per
    (user, ``day_expr``). ``row_level=True`` (the run/cohort variants, i.e.
    the AI-combo dataset) always uses the stored per-bet label."""
    policy = GROUP_POLICY[game_id]
    if row_level or "branches" not in policy:
        return _fold_expr(policy, "t.ab_group")
    label = group_label_sql(
        game_id,
        {"partition_ab_label": "t.partition_ab_label", "user_id": "t.user_id"},
        "t.created_at",
    )
    if not policy.get("collapse"):
        return label
    return collapse_case_over(game_id, label, day_expr)
