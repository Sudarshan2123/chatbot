#!/bin/bash

# Set the Docker image name and tag
IMAGE_NAME="rbi-app-totp"
IMAGE_TAG="latest"
# Set the Docker registry URL
REGISTRY_URL="macom-ai"
# Path to the docker-compose.yml file
COMPOSE_FILE="/home/postgres/treafik/docker-compose.yml"

# Check if the Dockerfile exists in the current directory
if [[ ! -f "Dockerfile" ]]; then
    echo "Error: Dockerfile not found in the current directory."
    exit 1
fi

# Check if docker-compose.yml exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "Error: $COMPOSE_FILE does not exist."
    exit 1
fi

# Check if docker and docker-compose are installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    exit 1
fi
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed."
    exit 1
fi

# Build the Docker image
echo "Building Docker image: $REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG"
docker build -t "$REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG" .
if [[ $? -ne 0 ]]; then
    echo "Error: Docker build failed."
    exit 1
fi

# Optionally push the image to the registry (uncomment if needed)
# echo "Pushing Docker image to $REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG"
# docker push "$REGISTRY_URL/$IMAGE_NAME:$IMAGE_TAG"
# if [[ $? -ne 0 ]]; then
#     echo "Error: Docker push failed."
#     exit 1
# fi

# Run docker-compose up
echo "Running docker-compose up for $COMPOSE_FILE..."
docker-compose -f "$COMPOSE_FILE" up -d "$@"
if [[ $? -eq 0 ]]; then
    echo "Docker Compose started successfully."
else
    echo "Error: Failed to start Docker Compose."
    exit 1
fi
