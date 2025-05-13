#!/bin/bash

# Absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLRUNS_DIR="$PROJECT_DIR/mlruns"

echo "🔁 Starting MLflow tracking server..."
echo "📂 Backend store: $MLRUNS_DIR"
echo "🌐 Tracking URI: http://127.0.0.1:5000"

# Set tracking URI for MLflow clients
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Run MLflow server
mlflow server \
  --backend-store-uri "file://$MLRUNS_DIR" \
  --default-artifact-root "file://$MLRUNS_DIR/artifacts" \
  --host 127.0.0.1 \
  --port 5000
