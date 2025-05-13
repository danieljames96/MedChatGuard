# utils/mlflow_logger.py

import mlflow
import json
from datetime import datetime

mlflow.set_tracking_uri("http://localhost:5000")

def log_rag_state_to_mlflow(state: dict, run_name: str = None):
    if not run_name:
        run_name = f"langgraph_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log flattened params
        for key, value in state.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)

            elif hasattr(value, "content"):  # e.g., AIMessage
                mlflow.log_param(key, value.content)

            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    mlflow.log_param(f"{key}_{subkey}", str(subval))

            elif isinstance(value, list) and all(isinstance(i, dict) for i in value):
                for i, item in enumerate(value):
                    for subkey, subval in item.items():
                        mlflow.log_param(f"{key}_{i}_{subkey}", str(subval))

            else:
                mlflow.log_param(key, str(value))

        # Save entire state as JSON artifact
        mlflow.log_dict(state, artifact_file="./log/full_rag_state.json")
