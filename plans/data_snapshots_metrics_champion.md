# Plan: Data Snapshots, Metrics Tracking, and Champion Promotion

## Context
Currently the weekly retrain job always promotes the new model as champion if prediction succeeds — no metrics are compared. There's no link between which data a model was trained on and the model itself. This plan adds:
1. Data snapshots before each retrain for reproducibility
2. Data quality metrics on each snapshot to track drift
3. A `model_train.json` with training/testing metrics per retrain
4. Champion promotion only when the new model outperforms the current champion (re-evaluated on new data)

## File to modify
- `modal_jobs/model_retrain.py` — all changes are here (+ one import from `src/modeling.py`)

No new files. No changes to `src/` modules — `get_ci_err()` in `src/modeling.py` already exists and just needs to be imported.

## Changes

### 1. Data Snapshots (before training)

Move `utc_timestamp` / `folder_time` computation earlier (before `de.create_database()`), then copy the 3 source parquet files to a versioned snapshot path using `s3.copy_object()` (server-side copy, no download needed):

```
data/snapshots/2026-03-02_20-00-00/lmp.parquet
data/snapshots/2026-03-02_20-00-00/mtlf.parquet
data/snapshots/2026-03-02_20-00-00/mtrf.parquet
```

### 2. Data Snapshot Metrics

After snapshotting, compute summary statistics on the source data and save as `data/snapshots/<timestamp>/data_metrics.json`. Uses the DuckDB connection (already open) to query each table efficiently.

```json
{
  "snapshot_timestamp": "2026-03-02T20:00:00Z",
  "tables": {
    "lmp": {
      "row_count": 450000,
      "columns": {
        "LMP": {
          "min": -15.23,
          "max": 312.50,
          "avg": 24.67,
          "median": 21.05,
          "p25": 15.30,
          "p75": 30.12,
          "missing_count": 0
        },
        "timestamp_mst": {
          "min": "2025-03-02 00:00:00",
          "max": "2026-03-02 07:00:00",
          "missing_count": 0
        }
      }
    },
    "mtlf": {
      "row_count": 8926,
      "columns": {
        "MTLF": { "min": ..., "max": ..., "avg": ..., "median": ..., "p25": ..., "p75": ..., "missing_count": 0 }
      }
    },
    "mtrf": {
      "row_count": 8928,
      "columns": {
        "MTRF": { "min": ..., "max": ..., "avg": ..., "median": ..., "p25": ..., "p75": ..., "missing_count": 0 }
      }
    }
  }
}
```

For numeric columns: min, max, avg, median, p25, p75, missing_count. For timestamp columns: min, max, missing_count. Computed via DuckDB `SUMMARIZE` or aggregate queries on the tables already loaded in the connection.

### 3. Compute Model Metrics (after ensemble is created)

Replace the current single-node test prediction (lines 260-279) with evaluation across all test nodes:

- Generate predictions for each node using `loaded_model.predict()` with `train_series` as history, predicting `n=FORECAST_HORIZON` steps into the test period
- Compute per-node **MAE** and **RMSE** from median predictions vs actuals (`sklearn.metrics`)
- Compute per-node **CI coverage error** using existing `modeling.get_ci_err()` (80% prediction interval)
- Average across nodes for aggregate metrics

New imports: `numpy`, `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.mean_squared_error`, `modeling.get_ci_err`

### 4. Upload `model_train.json`

Saved to `model_retrains/<timestamp>/model_train.json` alongside model artifacts:

```json
{
  "training_timestamp": "2026-03-02T20:00:00Z",
  "folder_time": "2026-03-02_20-00-00/",
  "artifact_folder": "model_retrains/2026-03-02_20-00-00/",
  "data_snapshot_path": "data/snapshots/2026-03-02_20-00-00/",
  "new_model_metrics": {
    "avg_mae": 12.34,
    "avg_rmse": 18.56,
    "avg_ci_coverage_error": 2.15,
    "per_node": {
      "PSCO_BHCE": {"mae": 11.2, "rmse": 16.8, "ci_coverage_error": 1.8},
      "PSCO_CRSP": {"mae": 13.5, "rmse": 20.3, "ci_coverage_error": 2.5}
    }
  },
  "champion_metrics_on_new_data": {
    "champion_artifact_folder": "model_retrains/2026-02-23_20-00-00/",
    "avg_mae": 14.10,
    "avg_rmse": 21.30,
    "avg_ci_coverage_error": 4.50,
    "per_node": {
      "PSCO_BHCE": {"mae": 13.0, "rmse": 19.5, "ci_coverage_error": 3.8},
      "PSCO_CRSP": {"mae": 15.2, "rmse": 23.1, "ci_coverage_error": 5.2}
    }
  },
  "promoted": true,
  "promotion_reason": "new MAE 12.34 < champion MAE 14.10",
  "train_test_split": {
    "train_start": "2025-03-02 00:00:00",
    "train_end": "2026-02-09 00:00:00",
    "test_start": "2026-02-09 00:00:00",
    "test_end": "2026-02-23 00:00:00",
    "n_train_rows": 150000,
    "n_test_rows": 5000,
    "n_nodes": 8
  },
  "model_config": {
    "model_type": "tide",
    "n_models": 5,
    "forecast_horizon": 120,
    "input_chunk_length": 168
  }
}
```

### 5. Re-evaluate Champion on New Data & Promote Based on MAE

Replace the current `if pred is not None` promotion (lines 282-294) with a fair comparison on the same data. This catches data drift — a champion trained on old data may perform worse on the latest test set.

**Steps:**
1. Load existing `S3_models/champion.json` from R2 to get `champion_artifact_folder`
2. Download and load the current champion ensemble model (reuse existing `load_model_from_s3` helper)
3. Generate predictions on the **same `train_series` / `test_series`** used for the new model
4. Compute champion's MAE, RMSE, CI coverage error on the new test data
5. Compare `avg_mae`: promote only if new model's MAE is lower than champion's fresh MAE

**Edge cases:**
  - No existing champion (first run / `NoSuchKey`): always promote
  - Legacy champion without model files: always promote
  - Champion fails to load or predict: promote the new model
  - New model failed to compute metrics: do not promote

Both sets of metrics (new model + champion re-evaluation) are recorded in `model_train.json` for audit.

Updated `champion.json` structure:
```json
{
  "champion": "2026-03-02_20-00-00/",
  "champion_artifact_folder": "model_retrains/2026-03-02_20-00-00/",
  "champion_artifact_path": "model_retrains/2026-03-02_20-00-00/",
  "data_snapshot_path": "data/snapshots/2026-03-02_20-00-00/",
  "promoted_at": "2026-03-02T20:45:00Z",
  "metrics": {
    "avg_mae": 12.34,
    "avg_rmse": 18.56,
    "avg_ci_coverage_error": 2.15
  }
}
```

### 6. Force Promotion

Add a `--force-promote` mechanism for cases where the champion must be replaced regardless of metrics — e.g., after a Darts version upgrade (old models are incompatible), a change in input data schema, or a change in model hyperparameters/architecture.

**Implementation:**
- Add an environment variable `FORCE_PROMOTE=true` that skips the champion comparison and always promotes the new model
- In Modal, pass via `env={"FORCE_PROMOTE": "true"}` on a one-off `modal run` (not in the scheduled deploy)
- The marimo notebook checks `os.getenv("FORCE_PROMOTE", "").lower() == "true"` before the comparison step
- When force-promoting, `model_train.json` records `"promoted": true, "promotion_reason": "force promoted"` and skips champion re-evaluation metrics
- The champion's old metrics are still logged as `null` for audit

**Usage:**
```bash
# One-off force promotion (does not affect scheduled deploys)
modal run modal_jobs/model_retrain.py::model_retrain_weekly --env FORCE_PROMOTE=true

# Or locally
FORCE_PROMOTE=true uv run marimo run notebooks/model_training/model_retrain.py
```

### 7. App compatibility

The Shiny app (`app.py`) reads `champion.json` and only uses `champion_artifact_folder` to load models. The new fields are additive — no changes needed to `app.py`.

## Verification
1. `modal run modal_jobs/model_retrain.py::model_retrain_weekly` — run a full retrain
2. Check R2 for:
   - `data/snapshots/<timestamp>/` with 3 parquet files + `data_metrics.json`
   - `model_retrains/<timestamp>/model_train.json` with metrics
   - `S3_models/champion.json` with updated structure including metrics
3. Run Shiny app to confirm it still loads the champion model correctly
4. Test force promotion: `FORCE_PROMOTE=true uv run marimo run notebooks/model_training/model_retrain.py` — verify it skips champion comparison and promotes unconditionally
