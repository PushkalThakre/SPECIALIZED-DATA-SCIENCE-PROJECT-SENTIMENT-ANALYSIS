#!/bin/bash
# deploy.sh — Build and deploy the project
set -e

echo "=== Building Docker image ==="
docker build -t ds-project-api:latest -f docker/Dockerfile .

echo "=== Running tests ==="
docker run --rm ds-project-api:latest python -m pytest tests/ -v

echo "=== Starting services ==="
docker-compose up -d

echo "=== API available at http://localhost:8000 ==="
echo "=== API docs at http://localhost:8000/docs ==="
