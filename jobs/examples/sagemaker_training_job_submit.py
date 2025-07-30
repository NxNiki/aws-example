from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.debugger import DebuggerHookConfig, FrameworkProfile, ProfilerConfig, ProfilerRule, Rule, rule_configs
from sagemaker.pytorch import PyTorch

from bituslabs_ds.config import S3_BUCKET, SAGEMAKER_ROLE, setup_logging
from bituslabs_ds.s3_utils import upload_file_to_s3, upload_folder_to_s3
from bituslabs_ds.utils import get_data_from_url

setup_logging(".sagemaker_log", "sagemaker_training_job_submit.log")
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

# s3_path = "s3://sagemaker-us-west-2-338568447110/Project-pytorch-dogImages-20250729-163817/dogImages"
s3_path = "s3://bituslabs-team-ai/Project-pytorch-dogImages-20250729-180144/dogImages"
if not s3_path:
    data_url = "https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"
    get_data_from_url(data_url, "dogImages.zip", "./")

    prefix = "Project-pytorch-dogImages"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # s3_path = sagemaker_session.upload_data("dogImages", key_prefix=f"{prefix}-{timestamp}")
    s3_path, _ = upload_folder_to_s3(Path("./dogImages/"), S3_BUCKET, f"{prefix}-{timestamp}")
    print("input spec (in this case, just an S3 path): {}".format(s3_path))


rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
]

hook_config = DebuggerHookConfig(hook_parameters={"train.save_interval": "500", "eval.save_interval": "50"})

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500, framework_profile_params=FrameworkProfile(num_steps=10)
)

hyperparameters = {
    "_tuning_objective_metric": '"average test loss"',
    "batch-size": '"256"',
    "lr": "0.005",
}

estimator = PyTorch(
    role=SAGEMAKER_ROLE,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    entry_point="sagemaker_training_job.py",
    framework_version="2.0",
    py_version="py310",
    hyperparameters=hyperparameters,
    debugger_hook_config=hook_config,
    profiler_config=profiler_config,
    rules=rules,
)

estimator.fit(
    {
        "train": f"{s3_path}/train",
        "valid": f"{s3_path}/valid",
        "test": f"{s3_path}/test",
    },
    wait=True,
)
