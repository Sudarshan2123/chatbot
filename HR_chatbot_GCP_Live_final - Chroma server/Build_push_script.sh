#!/bin/bash
#loan_delinquency-app
# Set the Docker image name and tag
IMAGE_NAME="hr-live"
IMAGE_TAG="1.0.1"

# Set the Docker registry URL
REGISTRY_URL="gcr.io/macomai-441611"

# Build the Docker image
docker build -t $REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG .

# Push the Docker image to the registry
docker push $REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG

echo "Docker image $REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG has been built and pushed to the registry."
