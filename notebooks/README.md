# `notebooks/` — marimo notebooks

These are [marimo](https://marimo.io) notebooks stored as plain `.py` files, so
they are git-friendly (clean diffs), importable, and **runnable as scripts**:

```bash
marimo edit notebooks/data_collection/data_collection_im_hourly.py   # interactive
python  notebooks/data_collection/data_collection_im_hourly.py       # headless run
```

Each cell is a separate `@app.cell def _(...)` whose parameters are the cell's
dependencies (marimo's reactive graph). Keep real logic in `src/` and call it
from the cells — the notebooks orchestrate, they don't reimplement.

**Modal jobs wrap these notebooks.** A `modal_jobs/*.py` job imports a
notebook's `app` and calls `app.run()` headlessly, so the notebook is the single
source of truth for the job logic (see the Modal-jobs table in the root README).

## Layout

| Folder | Notebooks |
|--------|-----------|
| `data_collection/` | Integrated Marketplace collection: `data_collection_im_hourly.py` (MTLF/MTRF/RF/5-min LMP), `data_collection_im_daily.py` (daily-LMP repair sweep + DA LMP), `data_collection_im_backfill.py` (one-time history backfill), and `data_collection_weather.py` (weather covariate). |
| `model_training/` | `model_retrain.py` (production retrain — trains the ensemble, uploads checkpoints, promotes `champion.json`; set `PROMOTE_CHAMPION=false` to stage without promoting), plus tuning/experiment notebooks (`model.py`, `model_ensemble.py`). |
| `app/` | App-testing notebooks. |

Retired WEIS market-collection notebooks live under `deprecated/weis/`.

## Notes

- Notebooks detect the project root via `pathlib.Path(__file__)` and add `src/`
  to `sys.path`; the `__init__.py` files let Modal import them as packages.
- Local runs read credentials from the gitignored `.env`; Modal runs use the
  `aws-secret` secret.
