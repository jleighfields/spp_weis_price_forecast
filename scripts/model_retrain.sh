#!/bin/bash

NOTIFY_EMAIL="jleighfields@gmail.com"
PROJECT_DIR="$HOME/Documents/github/spp_weis_price_forecast"

cd "$PROJECT_DIR"
mkdir -p ./scripts/model_retrains/
NOW=$(date -u +"%Y-%m-%dT%H-%M-%S")
OUTPUT_NOTEBOOK="./scripts/model_retrains/${NOW}_model_retrain.html"

if ! uv run marimo export html ./scripts/model_retrain_marimo.py -o "$OUTPUT_NOTEBOOK" 2>&1; then
    echo "Model retrain failed at $NOW" | mail -s "Model retrain job FAILED - $NOW" "$NOTIFY_EMAIL"
fi
