from pathlib import Path


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


REGION = "us-west-2"
S3_BUCKET = "bituslabs-team-ai"
ATHENA_OUTPUT = f"s3://{S3_BUCKET}/athena-results/"
LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_JOBS = get_cpu_cores()
