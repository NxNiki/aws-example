import os
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.debugger import (
    CollectionConfig,
    DebuggerHookConfig,
    FrameworkProfile,
    ProfilerConfig,
    ProfilerRule,
    Rule,
    rule_configs,
)
from sagemaker.pytorch import PyTorch

from bituslabs_ds.config import IMAGE_URI, LOCAL_ROOT, S3_BUCKET, SAGEMAKER_ROLE, setup_logging
from bituslabs_ds.s3_utils import upload_file_to_s3, upload_folder_to_s3
from bituslabs_ds.utils import get_data_from_url

setup_logging(f"{LOCAL_ROOT}/.sagemaker_log", "sagemaker_training_job_submit.log")
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()


def prepare_data(s3_path):
    if not s3_path:
        data_url = "https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"
        get_data_from_url(data_url, "dogImages.zip", "./")

        prefix = "Project-pytorch-dogImages"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # s3_path = sagemaker_session.upload_data("dogImages", key_prefix=f"{prefix}-{timestamp}")
        s3_path, _ = upload_folder_to_s3(Path("./dogImages/"), S3_BUCKET, f"{prefix}-{timestamp}")
        print("input spec (in this case, just an S3 path): {}".format(s3_path))

    return s3_path


if __name__ == "__main__":

    # s3_path = "s3://sagemaker-us-west-2-338568447110/Project-pytorch-dogImages-20250729-163817/dogImages"
    s3_path = "s3://bituslabs-team-ai/Project-pytorch-dogImages-20250729-180144/dogImages"
    prepare_data(s3_path)

    # adding too many rules may result in file size error (<2GB)
    rules = [
        Rule.sagemaker(rule_configs.vanishing_gradient()),
        Rule.sagemaker(rule_configs.overfit()),
        # Rule.sagemaker(rule_configs.overtraining()),
        # Rule.sagemaker(rule_configs.poor_weight_initialization()),
        # Rule.sagemaker(rule_configs.loss_not_decreasing()),
        ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    ]

    hook_config = DebuggerHookConfig(
        hook_parameters={"train.save_interval": "5000", "eval.save_interval": "1000"},
        # Add CollectionConfigurations to control what is saved
        collection_configs=[
            # Define a custom collection for only the FC layer's weights
            CollectionConfig(
                name="fc_weights",
                parameters={
                    "include_regex": "model.fc.0.(weight|bias)",  # Regex for your new FC layer's weights and biases
                    "save_interval": "5000",  # Less frequent saves for these tensors
                },
            ),
            # Define a custom collection for only the FC layer's gradients
            CollectionConfig(
                name="fc_gradients",
                parameters={
                    "include_regex": "model.fc.0.(weight|bias).grad",  # Regex for your new FC layer's gradients
                    "save_interval": "5000",
                },
            ),
        ],
    )

    profiler_config = ProfilerConfig(
        system_monitor_interval_millis=5000, framework_profile_params=FrameworkProfile(num_steps=5, start_step=500)
    )

    hyperparameters = {
        "_tuning_objective_metric": '"average test loss"',
        "batch-size": 256,
        "lr": 0.005,
        "epochs": 20,
    }

    estimator = PyTorch(
        image_uri=IMAGE_URI,
        role=SAGEMAKER_ROLE,
        instance_count=1,  # make sure this does not exceed the instance quota, and the job script needs to config distributed training.
        instance_type="ml.g4dn.xlarge",
        entry_point="sagemaker_training_job.py",
        source_dir=f"{LOCAL_ROOT}/jobs/examples",  # aviod large files in the scourc_dir or it takes long time to transfer to instance.
        framework_version="2.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        debugger_hook_config=hook_config,
        profiler_config=profiler_config,
        rules=rules,
        max_run=36000,  # 10 hours (in seconds)
        keep_alive_period_in_seconds=1800,
    )

    # Input channels with FullyReplicated setting
    train_input = sagemaker.inputs.TrainingInput(
        s3_data=f"{s3_path}/train/",
        # distribution="FullyReplicated",
    )

    validation_input = sagemaker.inputs.TrainingInput(
        s3_data=f"{s3_path}/valid/",
        # distribution="FullyReplicated",
    )

    test_input = sagemaker.inputs.TrainingInput(
        s3_data=f"{s3_path}/test/",
        # distribution="FullyReplicated",
    )

    estimator.fit(
        {
            "train": train_input,
            # "valid": validation_input,
            "test": test_input,
        },
        wait=False,
    )
