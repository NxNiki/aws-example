"""
prepare slot machine data to reproduce math table
"""

from bituslabs_ds.athena_utils import execute_query

athena_database = "ag_share_data"

# Athena athena_query
query = """
    SELECT type, column1, column2, column3, column4, column5, column6, column7, column8
    FROM sampled_slotorders_timeblocks
    WHERE bet_amount <= 500
    ORDER BY bet_time
"""
data_output = f"ds-data-slotorders/slotorders_wucaishen_for_mathtable.csv"

df = execute_query(query, athena_database, data_output)
print(df.head(5))

query = """
    select 
        type, 
        column1, column2, column3, column4, column5, column6, column7, column8,
        game,
        bet_time,
        bet_amount,
        payout,
        sum(payout) over (partition by productid, loginname, timeblock) as overall_payout
    from sampled_slotorders_timeblocks
    where bet_amount <= 500
    order by bet_time
"""
data_output = f"ds-data-slotorders/slotorders_wucaishen_for_mathtable_fulldata.csv"

df = execute_query(query, athena_database, data_output)
print(df.head(5))
