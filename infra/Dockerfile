#FROM 246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3
FROM python:3.9-slim

# Install Poetry
RUN pip install poetry==1.8.2

# Set up working directory
WORKDIR /opt/program

# Copy poetry files
#COPY ./pyproject.toml poetry.lock ./
COPY ./pyproject.toml ./

# Install dependencies first (excluding dev deps for SageMaker)
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction

# Copy your actual source code
COPY ./src/bituslabs_ds ./bituslabs_ds

# Optional: copy your processing script here too
#COPY processing_script.py ./

# Set Python environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONPATH="/opt/program"
#ENV PYTHONPATH="/opt/program:${PYTHONPATH}"

ENTRYPOINT ["python3"]