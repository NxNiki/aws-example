import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional, Union

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


# =========================================================================
# Multi-Process Logging Components
# =========================================================================

# Global Queue for log records
log_queue: Queue = Queue()  # Global multi-process log queue for interprocess logging


def listener_process(queue, log_path):
    """Listens for log records on the queue and writes them to file/stream."""
    # This process needs its own logger configuration

    # 1. Configure the file and stream handlers (where logs actually go)
    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | PID:%(process)d | %(message)s")

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 2. Set up the root logger to use only the configured handlers
    root = logging.getLogger()
    root.handlers.clear()  # Clear any existing handlers
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # 3. Use a listener to pull records from the queue and send them to the handlers
    listener = logging.handlers.QueueListener(queue, file_handler, stream_handler)
    listener.start()

    # Keep the listener running until shutdown
    try:
        while True:
            time.sleep(0.1)  # Small sleep to prevent busy waiting
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


def setup_logging(
    output_path: Union[str, Path] = ".", log_filename: str = "", multiprocess: bool = False
) -> Optional[Process]:
    """
    Configures the main process and worker processes for safe logging.
    Returns the log listener process.
    """
    if output_path == ".":
        log_dir = os.path.join(output_path, "log")
    else:
        log_dir = str(output_path)
    os.makedirs(log_dir, exist_ok=True)

    if len(log_filename) == 0:
        time_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"job_{time_tag}.log"
    log_path = os.path.join(log_dir, log_filename)

    if multiprocess:
        # 1. Start the separate process that will manage writing to the log file
        log_listener_process = Process(target=listener_process, args=(log_queue, log_path))
        # Make the listener a daemon so it won't keep containers/instances alive after the main process exits
        log_listener_process.daemon = True
        log_listener_process.start()

        # 2. Configure the main process's root logger and all worker loggers
        #    to use the QueueHandler, which sends logs to the queue.
        root = logging.getLogger()
        root.handlers.clear()  # Crucial: clear old FileHandler
        root.setLevel(logging.INFO)

        # All loggers (main and workers) will write to this handler
        queue_handler = logging.handlers.QueueHandler(log_queue)
        root.addHandler(queue_handler)

        return log_listener_process
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        )
        return None
