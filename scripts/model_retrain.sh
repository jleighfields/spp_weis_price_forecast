cd ~/Documents/github/spp_weis_price_forecast
mkdir -p ./scripts/model_retrain/
NOW=$(date -u +"%Y-%m-%dT%H-%M-%S")
uv run papermill ./scripts/model_retrain.ipynb "./scripts/model_retrains/${NOW}_model_retrain.ipynb"