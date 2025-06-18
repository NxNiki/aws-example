from sagemaker.estimator import Estimator

role = "arn:aws:iam::338568447110:role/SageMakerExecutionRole"
image_uri = "338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-sagemaker:latest"
print(f"image_uri: {image_uri}")

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    entry_point="cluster_analysis_02_kmeans.py",  # <-- this is a local path to your script
    source_dir=".",  # optional: directory containing train.py
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://your-bucket/output/",
)

estimator.fit({"train": "s3://your-bucket/train/"})
