"""
Lightweight test app that patches heavy startup loaders with fixture data.

This allows Playwright E2E tests to run without R2/S3 credentials or model
checkpoints. The real app_ui and server are imported from app.py, but
_do_load_data and _do_load_models are monkeypatched before the Shiny App
is constructed.
"""

import sys
import unittest.mock as mock
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries

# Ensure project root is on sys.path so `import app` works when this file
# is launched as a subprocess by create_app_fixture from tests/e2e/.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import app as real_app

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

NODES = ["TEST_NODE_A", "TEST_NODE_B"]
N_HOURS = 72


def _make_lmp_pd() -> pd.DataFrame:
    """Return a fake LMP DataFrame matching the shape app.py expects."""
    end = pd.Timestamp(datetime.now().replace(minute=0, second=0, microsecond=0))
    dates = pd.date_range(end=end, periods=N_HOURS, freq="h")
    rows = []
    for node in NODES:
        for ts in dates:
            rows.append({"timestamp_mst": ts, "unique_id": node, "LMP": float(np.random.default_rng(42).uniform(20, 60))})
    df = pd.DataFrame(rows).set_index("timestamp_mst")
    return df


def _make_all_df_pd() -> pd.DataFrame:
    """Return a fake covariates DataFrame matching all_df_to_pandas() output."""
    end = pd.Timestamp(datetime.now().replace(minute=0, second=0, microsecond=0))
    dates = pd.date_range(end=end, periods=N_HOURS, freq="h")
    rows = []
    rng = np.random.default_rng(42)
    for node in NODES:
        for ts in dates:
            rows.append({
                "timestamp_mst": ts,
                "unique_id": node,
                "LMP": float(rng.uniform(20, 60)),
                "MTLF": float(rng.uniform(1400, 1800)),
                "Wind_Forecast_MW": float(rng.uniform(400, 600)),
                "Solar_Forecast_MW": float(rng.uniform(100, 400)),
                "re_ratio": float(rng.uniform(0.2, 0.5)),
                "re_diff": float(rng.uniform(-50, 50)),
                "load_net_re": float(rng.uniform(800, 1200)),
                "load_net_re_diff": float(rng.uniform(-100, 100)),
                "load_net_re_diff_rolling_2": float(rng.uniform(-100, 100)),
                "load_net_re_diff_rolling_3": float(rng.uniform(-100, 100)),
                "load_net_re_diff_rolling_4": float(rng.uniform(-100, 100)),
                "load_net_re_diff_rolling_6": float(rng.uniform(-100, 100)),
                "Averaged_Actual": float(rng.uniform(20, 60)),
                "lmp_diff": float(rng.uniform(-10, 10)),
                "lmp_diff_rolling_2": float(rng.uniform(-10, 10)),
                "lmp_diff_rolling_3": float(rng.uniform(-10, 10)),
                "lmp_diff_rolling_4": float(rng.uniform(-10, 10)),
                "lmp_diff_rolling_6": float(rng.uniform(-10, 10)),
                "lmp_load_net_re": float(rng.uniform(-500, 500)),
            })
    df = pd.DataFrame(rows).set_index("timestamp_mst")
    return df


def _fake_load_data():
    """Drop-in replacement for app._do_load_data."""
    return _make_all_df_pd(), _make_lmp_pd()


class _FakeModel:
    """Minimal mock that satisfies model.predict() in _run_forecast."""

    def predict(self, *, series, past_covariates, future_covariates, n, num_samples=1):
        start = series.end_time() + pd.Timedelta("1h")
        dates = pd.date_range(start=start, periods=n, freq="h")
        rng = np.random.default_rng(0)
        values = rng.uniform(20, 60, (n, 1, num_samples))
        return TimeSeries.from_times_and_values(times=dates, values=values, columns=["LMP"])


def _fake_load_models():
    """Drop-in replacement for app._do_load_models."""
    return _FakeModel(), pd.Timestamp("2025-01-15 08:00:00")


# ---------------------------------------------------------------------------
# Patch and expose
# ---------------------------------------------------------------------------

real_app._do_load_data = _fake_load_data
real_app._do_load_models = _fake_load_models

# Re-export so Shiny can find the app object
app = real_app.app
