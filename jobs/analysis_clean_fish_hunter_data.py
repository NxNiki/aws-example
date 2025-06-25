"""
compare stats of old and new data for the fish hunter game and match column names.
"""

from bituslabs_ds.athena_utils import execute_query
from bituslabs_ds.config import S3_BUCKET, setup_logging

setup_logging("./.log")

query = """
    SELECT *
    FROM agfish_test.hunterorders 
    TABLESAMPLE SYSTEM (.01);
"""


file_path_hunter_orders = "./output/hunter_orders.csv"
s3_uri_hunter_orders = f"s3://{S3_BUCKET}/fish_hunter/hunter_orders.csv"
hunter_orders = execute_query(
    query, "agfish_test", s3_file_path=s3_uri_hunter_orders, local_cache=file_path_hunter_orders
)

print(hunter_orders.describe())
