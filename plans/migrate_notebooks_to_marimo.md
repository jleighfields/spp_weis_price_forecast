# Plan: Migrate Jupyter Notebooks to Marimo

## Context
The project has 11 Jupyter notebooks in `notebooks/` alongside Modal jobs in `modal_jobs/`. The model retrain notebook and modal job share ~250 lines of duplicated code. Marimo notebooks are pure `.py` files that work as both interactive notebooks (`marimo edit`) and scripts (`python notebook.py`), giving git-friendly diffs, reproducible execution, and eliminating code duplication between notebooks and jobs.

## Key Pattern: Modal Jobs Run Marimo Notebooks Directly

Since marimo notebooks are `.py` files with an importable `app` object, Modal jobs become thin wrappers that import and run the notebook:

```python
# modal_jobs/model_retrain.py
@modal_app.function(image=image, schedule=..., gpu="A10G", ...)
def model_retrain_weekly():
    from notebooks.model_training.model_retrain import app as notebook_app
    outputs, defs = notebook_app.run()
    # defs contains all cell-defined variables if needed for logging
```

This means:
- **One source of truth** — the marimo notebook *is* the job logic, no duplication
- **Modal jobs become ~5-line wrappers** — just configure resources/schedule and run the notebook
- **Interactive debugging** — `marimo edit notebooks/model_training/model_retrain.py`
- **Script mode** — `python notebooks/model_training/model_retrain.py` runs headlessly

## Prerequisites
- Add `marimo` to `pyproject.toml` dev dependencies
- `uv sync` to install
- Add `marimo` to Modal image `.pip_install()` in both modal jobs
- Add `.add_local_dir("notebooks", remote_path="/root/notebooks")` to Modal images

## Migration Order (highest priority first)

### Phase 1: Model Retrain (HIGH — eliminates major code duplication)

**Current files:**
- `notebooks/model_training/model_retrain.ipynb` — interactive notebook
- `modal_jobs/model_retrain.py` — ~250 lines duplicated from notebook

**Action:**
1. `marimo convert notebooks/model_training/model_retrain.ipynb -o notebooks/model_training/model_retrain.py`
2. Refactor marimo notebook: fix variable reuse, mutations, ensure all retrain logic lives in the notebook cells
3. Replace `modal_jobs/model_retrain.py` body with:
   ```python
   def model_retrain_weekly():
       import sys
       sys.path.insert(0, "/root")
       from notebooks.model_training.model_retrain import app as notebook_app
       notebook_app.run()
   ```
4. Update Modal image to include `marimo` in `.pip_install()` and `.add_local_dir("notebooks", remote_path="/root/notebooks")`
5. Delete `model_retrain.ipynb`
6. Test: `marimo edit notebooks/model_training/model_retrain.py` (interactive), `modal run modal_jobs/model_retrain.py` (production)

### Phase 2: Data Collection Hourly + Daily (MEDIUM — eliminates notebook/job duplication)

**Current files:**
- `notebooks/data_collection/data_collection_hourly.ipynb`
- `notebooks/data_collection/data_collection_daily.ipynb`
- `modal_jobs/data_collection.py` — nearly identical to notebooks

**Action:**
1. Convert both notebooks with `marimo convert`
2. Refactor variable reuse patterns (single-definition per variable in marimo)
3. Replace `modal_jobs/data_collection.py` function bodies to import and run the marimo notebooks:
   ```python
   def collect_hourly():
       import sys
       sys.path.insert(0, "/root")
       from notebooks.data_collection.data_collection_hourly import app as notebook_app
       notebook_app.run()
   ```
   Same pattern for `collect_daily`.
4. Update Modal image to `.add_local_dir("notebooks", remote_path="/root/notebooks")`
5. Delete `.ipynb` files

### Phase 3: Remaining Data Collection Notebooks (LOW — infrequently used utilities)

**Files:**
- `notebooks/data_collection/data_collection_5_min.ipynb`
- `notebooks/data_collection/data_collection_backfill.ipynb`
- `notebooks/data_collection/data_collection_rebuild.ipynb`
- `notebooks/data_collection/data_collection_testing.ipynb`
- `notebooks/data_collection/data_collection_weather.ipynb`

**Action:**
1. Convert each with `marimo convert`
2. Fix variable reuse/mutation patterns
3. Delete `.ipynb` files

### Phase 4: Model Training + App Notebooks (LOW — exploration/tuning tools)

**Files:**
- `notebooks/model_training/model.ipynb` — Optuna hyperparameter tuning
- `notebooks/model_training/model_ensemble.ipynb` — Ensemble creation
- `notebooks/app/app_testing.ipynb` — Shiny app testing

**Action:**
1. Convert each with `marimo convert`
2. Fix variable reuse/mutation patterns
3. Delete `.ipynb` files

## Marimo Refactoring Notes

Common patterns to fix after `marimo convert`:
- **Duplicate variable names across cells** — rename to unique names or consolidate into one cell
- **DataFrame mutations** (`df["col"] = val`) — move to same cell as `df` definition, or use `df = df.assign(col=val)`
- **Magic commands** (`%pip`, `%cd`) — replace with standard Python
- **Prefix temporary variables** with `_` so they don't leak across cells

## Verification
- For each converted notebook: `marimo edit <file>` to verify interactive mode works
- For modal jobs: `modal run modal_jobs/data_collection.py::collect_hourly` and `modal run modal_jobs/model_retrain.py::model_retrain_weekly`
- `modal deploy` both apps after Phase 1 and 2

## Project Structure After Migration

```
├── notebooks/
│   ├── data_collection/
│   │   ├── data_collection_hourly.py      # marimo
│   │   ├── data_collection_daily.py       # marimo
│   │   ├── data_collection_5_min.py       # marimo
│   │   ├── data_collection_backfill.py    # marimo
│   │   ├── data_collection_rebuild.py     # marimo
│   │   ├── data_collection_testing.py     # marimo
│   │   └── data_collection_weather.py     # marimo
│   ├── model_training/
│   │   ├── model_retrain.py               # marimo (shared logic with modal job)
│   │   ├── model.py                       # marimo
│   │   └── model_ensemble.py              # marimo
│   └── app/
│       └── app_testing.py                 # marimo
├── modal_jobs/
│   ├── data_collection.py                 # thin wrapper → runs marimo notebooks
│   └── model_retrain.py                   # thin wrapper → runs marimo notebook
```
