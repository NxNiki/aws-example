#!/bin/bash

# This script is run during the EMR bootstrap phase.
# $1 is the S3 URI to a requirements.txt file or Python wheel package

set -e

echo "Starting bootstrap script"

# Update yum packages
sudo yum update -y

# Install common utilities (optional)
sudo yum install -y git

# Install Python packages
if [[ $1 == *.txt ]]; then
  echo "Installing Python packages from requirements.txt: $1"
  aws s3 cp "$1" /home/hadoop/requirements.txt
  sudo python3 -m pip install -r /home/hadoop/requirements.txt
elif [[ $1 == *.whl ]]; then
  echo "Installing Python wheel package: $1"
  aws s3 cp "$1" /home/hadoop/package.whl
  sudo python3 -m pip install /home/hadoop/package.whl
else
  echo "Unknown file format. Please provide a .txt or .whl file."
  exit 1
fi

echo "Bootstrap script completed successfully"
