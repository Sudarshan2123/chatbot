#!/bin/bash
# Set the Docker image name and tag
IMAGE_NAME="rbi-app-totp"
IMAGE_TAG="latest"
# Set the Docker registry URL
REGISTRY_URL="macom-ai"

# Set the Docker container name

docker build -t $REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG .

