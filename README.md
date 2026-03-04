# aws-example

## how to use:

Use conda to create and activate the virtual environment, run in the project root directory:
```
conda env create -f environment.yml --prune
conda activate aws-example
```
Use peotry within a conda environment, make sure to avoid creating virtual environment with peotry:
```
poetry config virtualenvs.create false --local
poetry install
```
Otherwise, conda and peotry will use different virtual environments!

Setup pre-commit:
```
pre-commit install
```
This command will install the pre-commit hook into your .git/hooks directory. From now on, pre-commit will automatically run the defined hooks every time you try to git commit.

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

### examples:

scripts under `jobs/examples` can run locally or directly be submitted to MER/sagemaker without configuring environments. Use these example to ensure you have aws account configured correctly.


### connect to s3 from local:

Ensure your IAM role has `AmazonS3FullAccess` permission.

run script [data_loader_s3_example.py](https://github.com/NxNiki/aws-example/blob/788559a678e5e0fbd451b77ee6d9e74533f4c595/jobs/examples/data_loader_s3_example.py) to upload and download data from s3.

Check data in result folder on s3 to verify code run sucessfully.

### run sql query to fetch data with AWS Athena:

Add `AmazonAthenaFullAccess` to your IAM role.

run script [athena_utils.py](https://github.com/NxNiki/aws-example/blob/main/src/bituslabs_ds/athena_utils.py)

### run sql query to fetch data from redshift:

Credentials (never commit to git):

1. **Redshift**: Copy `.env.example` to `.env` and set `REDSHIFT_USER` and `REDSHIFT_PASSWORD`.
2. **Bastion** (for Redshift tunnel): Set `BASTION_KEY_PATH` to your `.pem` path, or use `BASTION_KEY_CONTENT` (Secrets Manager on ECS).

Host and port are in `bituslabs_ds.config`. Bastion IP: 13.215.212.244 (ubuntu).

Setup and run an ETL job:

Update user name and password for bastion connection then run:

```bash
cp .env.example .env
# Edit .env if needed, then:
source .env
poetry run python jobs/operation_daily_report/run_daily_report.py
```

Check [etl.py](https://github.com/NxNiki/aws-example/blob/main/src/bituslabs_ds/etl.py) to see sample to execute sql query.


### submit script to AWS EMR:

Create an EMR cluster on AWS.

Add `AmazonEMRFullAccessPolicy_v2` to your IAM role.

run script [data_loader_pyspark_example_submit.py](https://github.com/NxNiki/aws-example/blob/b434e9c2e819772b6df87d5e17876353764dbbbf/jobs/examples/data_loader_pyspark_example_submit.py) to ensure you can submit EMR jobs.

Check data in result folder on s3 to verify code run sucessfully.

https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html

### submit script to AWS sagemaker:

the sagemaker jobs are saved to: [jobs/sagemaker](https://github.com/NxNiki/aws-example/tree/419b3a9d1483d5050e6af26fd3b86ffc9df0cd10/jobs/sagemaker)






