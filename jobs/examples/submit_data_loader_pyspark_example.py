import logging
import os

from bituslabs_ds.emr_utils import submit_spark_step
from bituslabs_ds.s3_utils import upload_file_to_s3

if __name__ == "__main__":

    # Setup logging
    os.makedirs("../.log", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(".log/emr_job_submit.log"), logging.StreamHandler()],
    )

    # === Configuration ===
    LOCAL_SCRIPT = "data_loader_pyspark_example.py"
    BUCKET = "bituslabs-team-ai"
    S3_KEY = "data_loader_pyspark_example.py"
    DATA_SOURCE = f"s3://{BUCKET}/xin-config/food_establishment_data.csv"
    OUTPUT_URI = f"s3://{BUCKET}/xin-config/restaurant_violation_results"
    CLUSTER_ID = "j-WD67QQS5JW2Z"  # Your EMR Cluster ID
    REGION = "us-west-2"

    # === Upload and Submit ===
    s3_script_uri = upload_file_to_s3(LOCAL_SCRIPT, BUCKET, S3_KEY)
    if s3_script_uri:
        submit_spark_step(CLUSTER_ID, s3_script_uri, DATA_SOURCE, OUTPUT_URI, region=REGION)
