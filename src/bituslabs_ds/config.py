import logging
import os
from datetime import datetime
from pathlib import Path

REGION = "us-west-2"
S3_BUCKET = "bituslabs-team-ai"
SAGEMAKER_ROLE = "arn:aws:iam::338568447110:role/SageMakerExecutionRole"
IMAGE_URI = "338568447110.dkr.ecr.us-west-2.amazonaws.com/bituslabs-ds-sagemaker:latest"
DEFAULT_MAX_JOBS = 4
DEFAULT_ATHENA_OUTPUT = f"s3://{S3_BUCKET}/athena-results"
LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent


def get_cpu_cores(logical=True, default=1):
    """
    Returns the number of available CPU cores, with support for diverse environments
    like local, Docker, SageMaker, EMR, etc.

    Parameters:
        logical (bool): Whether to return logical (hyperthreaded) cores. If False, returns physical cores.
        default (int): Fallback value if no CPU count can be determined.

    Returns:
        int: Number of CPU cores.
    """
    import os

    try:
        num_cores = os.cpu_count()
        if num_cores is not None:
            return num_cores
    except Exception:
        pass

    try:
        import psutil

        num_cores = psutil.cpu_count(logical=logical)
        if num_cores is not None:
            return num_cores
    except ImportError:
        pass

    return default


def setup_logging(output_path: str, log_filename: str = ""):
    if output_path == ".":
        log_dir = os.path.join(output_path, ".log")
    else:
        log_dir = output_path
    os.makedirs(log_dir, exist_ok=True)

    if len(log_filename) == 0:
        time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"job_{time_tag}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
