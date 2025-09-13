"""
after the sagemaker tuning job, use this script to get the hyperparameters of the best model.
"""

from sagemaker.tuner import HyperparameterTuner

if __name__ == "__main__":
    # Load the tuner from the previous job
    tuner = HyperparameterTuner.attach("pytorch-training-250826-1400")
    print("Tuning job name:", tuner.latest_tuning_job.name)
    best_estimator = tuner.best_estimator()
    # Get the hyperparameters of the best trained model
    print(best_estimator.hyperparameters())
