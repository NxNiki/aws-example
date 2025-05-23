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

## connect to s3 from local:

Ensure your IAM role has `AmazonS3FullAccess` permission.

run script [example_read_data_from_s3.py](https://github.com/NxNiki/aws-example/blob/main/example_read_data_from_s3.py) to upload and download data from s3.

Check data in result folder on s3 to verify code run sucessfully.

## run script on emr:

Create an EMR cluster on AWS.

Add `AmazonEMRFullAccessPolicy_v2` to your IAM role.

run script [submit_emr_job.py](https://github.com/NxNiki/aws-example/blob/main/submit_emr_job.py) to ensure you can submit EMR jobs.

Check data in result folder on s3 to verify code run sucessfully.

https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html




