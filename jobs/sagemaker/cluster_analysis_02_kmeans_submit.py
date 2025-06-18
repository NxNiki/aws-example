from datetime import datetime

from sagemaker.estimator import Estimator

from bituslabs_ds.config import S3_BUCKET

role = "arn:aws:iam::338568447110:role/SageMakerExecutionRole"
image_uri = "338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-sagemaker:latest"
print(f"image_uri: {image_uri}")

time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    entry_point="cluster_analysis_02_kmeans.py",  # <-- this is a local path to your script
    source_dir=".",  # optional: directory containing train.py
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{S3_BUCKET}/ds-data-kmeans/kmeans_{time_tag}",
)

estimator.fit({"train": "s3://your-bucket/train/"})
