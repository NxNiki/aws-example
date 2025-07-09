from textwrap import dedent, indent
from typing import List, Union

from bituslabs_ds.athena_utils import execute_query


def generate_query_for_table(fishes_indices: Union[List[int], range], table_names: Union[List[str], str]) -> str:
    """
    Generates an SQL query for fish hunter data with dynamic columns using f-strings.

    Returns:
        str: The generated SQL query.
    """

    if isinstance(table_names, str):
        table_names = [table_names]

    query_for_fish_index = []
    for table_name in table_names:
        for fishes_index in fishes_indices:
            if table_name == "hunter_logs_suspicious_clean_data":
                get_user_id = True
            else:
                get_user_id = False

            # add a blank line so that extra indent is not added to the 1st line:
            query = dedent(
                f"""\
                
                SELECT *
                FROM (
                    SELECT
                        {generate_query_columns(fishes_index, get_user_id, 24)}
                    FROM
                        {table_name}
                ) t
                WHERE
                    fishid is NOT NULL AND fishtype is NOT NULL AND fishcost IS NOT NULL
                """
            )

            query_for_fish_index.append(query)

    query = f"\nUNION ALL\n".join(query_for_fish_index)

    query = dedent(
        f"""\
        CREATE OR REPLACE VIEW all_hunter_logs_merged AS
        {indent(query, prefix=" " * 8)}
    """
    )

    return query


def generate_query_columns(fishes_index: int, get_user_id: bool = True, n_indents: int = 0) -> str:

    columns = (
        [
            "COALESCE(user_id, fields['name']) AS loginname",
            "CAST(date AS DATE) AS date",
            "device_id",
            "event_type",
            "event_value",
            "event_time AS billtime",
            "fields['enter-room'] AS fields_enter_room",
            "TRY_CAST(fields['credit-seq'] AS INT) AS creditseq",
            f"COALESCE( fields['fish'], SPLIT_PART(SPLIT_PART(fields['fishes'], ';', {fishes_index}), '/', {1}) ) AS fishid",
            f"COALESCE( fields['fish-type'], SPLIT_PART(SPLIT_PART(fields['fishes'], ';', {fishes_index}), '/', {2}) ) AS fishtype",
            f"COALESCE( fields['cost'], SPLIT_PART(SPLIT_PART(fields['fishes'], ';', {fishes_index}), '/', {3}) ) AS fishcost",
            "fields['p'] AS fields_p",
            "fields['betx'] AS betx",
            "fields['game-type'] AS fields_game_type",
            "fields['room'] AS roomid",
            "TRY_CAST(fields['prize'] AS DOUBLE) AS fields_prize",
            "fields['c'] AS fields_c",
            "TRY_CAST(fields['scene'] AS VARCHAR) AS sceneid",
            "fields['weapon-level'] AS fields_weapon_level",
            "TRY_CAST(fields['seat'] AS INT) AS fields_seat",
            "TRY_CAST(fields['avatar'] AS INT) AS fields_avatar",
            "fields['device'] AS device",
            "fields['release-wallet-session-error'] AS fields_release_wallet_session_error",
            "TRY_CAST(fields['frame'] AS INT) AS fields_frame",
            "fields['logout'] AS fields_logout",
            "fields['collected'] AS fields_collected",
            "fields['collection'] AS fields_collection",
            "fields['power'] AS fields_power",
            "fields['login'] AS fields_login",
            "fields['maximum'] AS fields_maximum",
            "fields['ip'] AS fields_ip",
            "fields['x'] AS fields_x",
            "fields['high'] AS fields_high",
            "fields['on-frozen-hit'] AS fields_on_frozen_hit",
            "fields['send-weapon-list'] AS fields_send_weapon_list",
            "fields['on-fish-hit'] AS fields_on_fish_hit",
            "fields['remain'] AS fields_remain",
            "fields['change-seat'] AS fields_change_seat",
            "fields['currency'] AS currency",
            "fields['weapon'] AS weaponid",
            "TRY_CAST(fields['credit'] AS DOUBLE) AS fields_credit",
            "TRY_CAST(fields['transaction'] AS BIGINT) AS fields_transaction",
            "fields['count'] AS fields_count",
            "fields['low'] AS fields_low",
            "fields['bullet-cost'] AS fields_bullet_cost",
            "fields['b'] AS fields_b",
            "fields['tick'] AS fields_tick",
            "fields['start'] AS fields_start",
            "fields['on-weapon-hit'] AS fields_on_weapon_hit",
            "fields['game'] AS game_type",
            "fields['get-jackpot-summary'] AS fields_get_jackpot_summary",
            "TRY_CAST(fields['level'] AS INT) AS fields_level",
            "TRY_CAST(fields['leave'] AS INT) AS fields_leave",
            "fields['elapsed'] AS fields_elapsed",
            "TRY_CAST(fields['hunter-vip'] AS INT) AS fields_hunter_vip",
            "fields['database'] AS fields_database",
            "fields['end'] AS fields_end",
            "fields['requested'] AS fields_requested",
            "fields['reason'] AS fields_reason",
            "fields['stamp'] AS fields_stamp",
        ]
        + [f"SPLIT_PART(fields['detail'], '/', {i}) AS fields_detail_{i}" for i in range(1, 10)]
        + [f"SPLIT_PART(fields['amount'], '/', {i}) AS fields_amount_{i}" for i in range(1, 8)]
    )

    if not get_user_id:
        columns[0] = "NULL AS loginname"

    # add indents except for the 1st line:
    indent = " " * n_indents
    columns_str = f",\n{indent}".join(columns)

    return columns_str


if __name__ == "__main__":

    sql_query = generate_query_for_table(
        range(1, 11),
        ["hunter_logs_clean_data", "hunter_logs_suspicious_clean_data", "hunter_logs_suspicious_clean_data_total"],
    )
    # execute_query(sql_query, database="ag_share_data")
    print(sql_query)
