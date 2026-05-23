#!/bin/bash
# run_pipeline.sh — Full pipeline: preprocess → train → evaluate → serve
set -e

echo "╔══════════════════════════════════════════════╗"
echo "║   Month 5 — Advanced DS Project Pipeline     ║"
echo "╚══════════════════════════════════════════════╝"

echo -e "\n[1/3] Training all models..."
python -m src.training.train_all

echo -e "\n[2/3] Running test suite..."
python -m pytest tests/ -v --tb=short

echo -e "\n[3/3] Starting API server..."
echo "API will be available at: http://localhost:8000"
echo "Swagger docs at:          http://localhost:8000/docs"
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
