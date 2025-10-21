# FROM 246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3
# FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-cpu-py310


FROM python:3.9-slim

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=off

# enable script mode so we can submit custom training jobs:
RUN apt-get update && apt-get install -y \
gcc \
g++ \
make \
libc-dev \
&& rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==2.1.3

# Set up working directory
WORKDIR /opt/program

# Copy poetry files
COPY ./pyproject.toml poetry.lock README.md ./

# Copy your actual source code
COPY ./src ./src

# Install dependencies first (excluding dev deps for SageMaker)
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction

# Optional: copy your processing script here too
#COPY processing_script.py ./

RUN pip install sagemaker-training

# Set Python environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONPATH="/opt/program"
#ENV PYTHONPATH="/opt/program:${PYTHONPATH}"

# ENTRYPOINT ["python3"]
ENTRYPOINT ["python3", "-m", "sagemaker_training.cli.train"]