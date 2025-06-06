#!/bin/bash

# This script is run during the EMR bootstrap phase.
# $1 is the S3 URI to a requirements.txt file or Python wheel package

set -e

echo "Starting bootstrap script"

# Wait for yum lock if necessary
while sudo fuser /var/run/yum.pid >/dev/null 2>&1; do
    echo "Waiting for yum lock..."
    sleep 5
done

# Update yum packages
sudo yum update -y

# Install common utilities (optional)
sudo yum install -y git

# Install Python packages
if [[ $1 == *.txt ]]; then
  echo "Installing Python packages from requirements.txt: $1"

  filename=$(basename "$1")
  aws s3 cp "$1" "/home/hadoop/$filename"
  sudo python3 -m pip install --upgrade --ignore-installed -r "/home/hadoop/$filename"

elif [[ $1 == *.whl ]]; then
  echo "Installing Python wheel package: $1"

  filename=$(basename "$1")
  aws s3 cp "$1" "/home/hadoop/$filename"
  sudo python3 -m pip install --ignore-installed "/home/hadoop/$filename"

else
  echo "Unknown file format. Please provide a .txt or .whl file."
  exit 1
fi

echo "Bootstrap script completed successfully"
