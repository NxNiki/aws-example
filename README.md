# aws-example

## install aws-cli:

https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

This is required to run aws commands locally to upload/download data from s3, submit sagemaker/EMR jobs to aws.
After installation, run `aws configure` to setup your aws access key, secret key, etc. 
You may need to create an IAM user to generate aws access key for yourself.

## connect to s3 from local:

run script [https://github.com/NxNiki/aws-example/blob/main/example_read_data_from_s3.py] (example_read_data_from_s3)

## run script on emr:

https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html

