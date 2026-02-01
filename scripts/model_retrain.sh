#!/bin/bash

NOTIFY_EMAIL="jleighfields@gmail.com"
PROJECT_DIR="$HOME/Documents/github/spp_weis_price_forecast"

cd "$PROJECT_DIR"
mkdir -p ./scripts/model_retrains/
NOW=$(date -u +"%Y-%m-%dT%H-%M-%S")
OUTPUT_NOTEBOOK="./scripts/model_retrains/${NOW}_model_retrain.ipynb"

if ! uv run papermill ./scripts/model_retrain.ipynb "$OUTPUT_NOTEBOOK" --execution-timeout 3600 2>&1; then
    echo "Model retrain failed at $NOW" | mail -s "Model retrain job FAILED - $NOW" "$NOTIFY_EMAIL"
fi