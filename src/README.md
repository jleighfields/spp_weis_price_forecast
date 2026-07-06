# `src/` — pipeline modules

Core library code for the SPP West price forecast. The data flows
**collection → engineering → modeling → serving**, and each module owns one
stage. Notebooks (`notebooks/`) and the Shiny app (`app.py`) import from here;
they don't reimplement this logic.

| Module | Responsibility |
|--------|----------------|
| `node_list.py` | Single home for the West/East node lists and West-scoping constants (`STORED_NODES`, `MODEL_APP_NODES`, `WEST_BAA`, `RTO_WEST_LAUNCH`). Deliberately dependency-light (no sklearn/darts) so the Modal collection image can import it. |
| `data_collection.py` | Shared collection helpers (`get_csv_from_url`, `_s3_storage_options`, `set_he`, `ProgressParallel`, …) **plus** the legacy WEIS ETL. The helpers are still imported by `data_collection_im.py`; the WEIS-specific collectors are retired (see `deprecated/weis/`). |
| `data_collection_im.py` | Integrated Marketplace collectors — RTBM 5-min + daily LMP, MTLF, MTRF, `RF_RESERVE_ZONE`, DA LMP — writing to the `data_im/` R2 prefix. Handles the `BAA` column, DST duplicate-hour files, and the daily-rollup publication lag. |
| `data_engineering.py` | Reads `data_im/` via DuckDB, filters to the West BAA (`BAA=='SWPW'`) and `MODEL_APP_NODES`, engineers features (renewable ratios, load-net-of-renewables, rolling diffs, the 2026-04-01 break indicator), and builds the Darts `TimeSeries` the models consume. |
| `modeling.py` | Builds and fits the Darts models (TiDE / TSMixer / TFT) and loads a saved ensemble (`load_ensemble_from_dir`). |
| `parameters.py` | Hyperparameters and run config — `MODEL_NAME`, `TRAIN_START`, forecast horizons, the `*_PARAMS` dicts, encoders. The single source of truth for model config. Imports sklearn/darts, so keep it out of the collection image (that's why node lists live in `node_list.py`). |
| `darts_wrapper.py` | mlflow PyFunc wrapper for serving a Darts model/ensemble. |
| `plotting.py` | Forecast visualizations for the Shiny app. |
| `utils.py` | R2/S3 utilities (listing, champion-checkpoint download). |

## Conventions

- Every parameter value has one home (see the root `CLAUDE.md` §"Single source
  of truth"). Model config → `parameters.py`; node lists / West scoping →
  `node_list.py`; storage config → environment variables.
- Google-style docstrings, `X | None` type hints, `#####`-bar section headers.
- Fast unit tests live in `tests/unit/`; run `uv run pytest tests/unit -q`.
