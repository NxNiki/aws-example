# aws-example

## how to use:

If you use peotry within a conda environment, make sure to aovid creating virtual environment with peotry:
```
poetry config virtualenvs.create false --local
```
Otherwise, conda and peotry will use different virtual environments!

## install aws-cli:

https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

This is required to run aws commands locally to upload/download data from s3, submit sagemaker/EMR jobs to aws.
After installation, run `aws configure` to setup your aws access key, secret key, etc. 
You may need to create an IAM user to generate aws access key for yourself.

## jobs:

The jobs folder contains scripts with following patterns:

- `data_loader_*.py`: to download data from s3 bucket
- `data_process_*.py`: to process data and upload result to s3
- `analysis_*.py`: run statistic analysis and make plots.


### connect to s3 from local:

Ensure your IAM role has `AmazonS3FullAccess` permission.

run script [s3_utils.py](https://github.com/NxNiki/aws-example/blob/main/src/bituslabs_ds/s3_utils.py) to upload and download data from s3.

Check data in result folder on s3 to verify code run sucessfully.

### run sql query to fetch data with AWS Athena:

Add `AmazonAthenaFullAccess` to your IAM role.

run script [athena_utils.py](https://github.com/NxNiki/aws-example/blob/main/src/bituslabs_ds/athena_utils.py)

### submit script two AWS EMR:

Create an EMR cluster on AWS.

Add `AmazonEMRFullAccessPolicy_v2` to your IAM role.

run script [submit_data_loader_pyspark_example.py](https://github.com/NxNiki/aws-example/blob/main/jobs/submit_data_loader_pyspark_example.py) to ensure you can submit EMR jobs.

Check data in result folder on s3 to verify code run sucessfully.

https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html




