#!/bin/bash
# Set the Docker image name and tag
IMAGE_NAME="str_data_bot_ui"
IMAGE_TAG="latest"
# Set the Docker registry URL
# REGISTRY_URL="macom-ai"

# Set the Docker container name

docker build -t $IMAGE_NAME:$IMAGE_TAG .

