#!/usr/bin/env bash

# Parse arguments for project ID, image name, and tag
while getopts p:i:t: flag
do
    case "${flag}" in
        p) PROJECT_ID=${OPTARG};;
        i) IMAGE_NAME=${OPTARG};;
        t) TAG=${OPTARG};;
    esac
done

# Check if required arguments are provided
if [ -z "$PROJECT_ID" ] || [ -z "$IMAGE_NAME" ] || [ -z "$TAG" ]; then
    echo "Usage: $0 -p <Project ID> -i <Image name> -t <Tag>"
    exit 1
fi

# Tag the Docker image
docker tag 3dteethland_processing docker.synapse.org/${PROJECT_ID}/${IMAGE_NAME}:${TAG}

# Push the Docker image
docker push docker.synapse.org/${PROJECT_ID}/${IMAGE_NAME}:${TAG}
