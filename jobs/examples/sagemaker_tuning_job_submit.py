import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import CategoricalParameter, ContinuousParameter, HyperparameterTuner, IntegerParameter
from sagemaker_training_job_submit import prepare_data

from bituslabs_ds.config import SAGEMAKER_ROLE

if __name__ == "__main__":

    s3_path = "s3://bituslabs-team-ai/Project-pytorch-dogImages-20250729-180144/dogImages"
    prepare_data(s3_path)

    estimator = PyTorch(
        entry_point="sagemaker_tuning_job.py",
        role=SAGEMAKER_ROLE,
        py_version="py310",
        framework_version="2.0",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        max_run=3600 * 10,
        keep_alive_period_in_seconds=1800,
    )

    objective_metric_name = "average test loss"
    objective_type = "Minimize"
    metric_definitions = [{"Name": "average test loss", "Regex": r"Test set: Average loss: ([0-9\.]+)"}]

    hyperparameter_ranges = {
        "lr": ContinuousParameter(0.001, 0.1),
        "batch-size": IntegerParameter(32, 256),
        "epochs": IntegerParameter(5, 20),
    }

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=40,
        max_parallel_jobs=20,  # make sure this does not exceed the instance quota.
        objective_type=objective_type,
    )

    tuner.fit(
        {
            "train": f"{s3_path}/train",
            "valid": f"{s3_path}/valid",
            "test": f"{s3_path}/test",
        },
        wait=False,
    )
