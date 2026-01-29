# FROM 246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3
# FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-cpu-py310

# --- STAGE 1: Builder Stage ---
FROM python:3.9-slim AS builder

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=off

# Install build dependencies: package manager and compiler tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc-dev \
    # Install curl/wget if needed for dependency download, otherwise omit
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==2.1.3
RUN pip install sagemaker-training

# Set up working directory for the builder
WORKDIR /opt/program

COPY ./pyproject.toml poetry.lock ./
# Use poetry to install dependencies in a clean, isolated environment
# We install to the system using 'virtualenvs.create false'
RUN poetry config virtualenvs.create false && poetry install --only main,dl --no-interaction --no-root

COPY ./src ./src
# --- STAGE 2: Runtime Stage (Final Image) ---
FROM python:3.9-slim AS final

# Copy runtime variables
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE

# Set up the final working directory
WORKDIR /opt/program

# Copy ONLY the installed packages and application code from the builder stage
# This is the key step to shrinking the final image
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/program/src /opt/program/src

# Set Python environment variables
ENV PYTHONPATH="/opt/program/src"

# ENTRYPOINT ["python3"] # Use the more specific training entrypoint
ENTRYPOINT ["python3", "-m", "sagemaker_training.cli.train"]