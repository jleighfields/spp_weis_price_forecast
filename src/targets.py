"""Forecast-target definitions — the single source of truth.

Kept dependency-light (no sklearn/darts) so darts-free callers — the
`scripts/r2_promote_champion.py` CLI, and any collection-side code — can import
the target set without pulling in the model stack. `src/parameters.py`
re-exports these so existing `parameters.TARGETS` / `parameters.DEFAULT_TARGET`
callers keep working.

Each target trains an independent model from its own source table into its own
model namespace (`models/<target>/...`):
  'da' — day-ahead auction prices (im/da_lmp.parquet), smooth/predictable
         (the primary/default model)
  'rt' — real-time RTBM prices (im/lmp.parquet), spiky/hard to predict
         (parked until the market matures)
`source_dataset` is the parquet basename under the IM prefix (data_engineering
builds im/<ds>.parquet). `label` is the human market name for the app UI (the
one home for it — the market selector and forecast header both read it).
"""

DEFAULT_TARGET = 'da'
TARGETS = {
    'rt': {'source_dataset': 'lmp', 'model_name': 'spp_west', 'label': 'Real-time'},
    'da': {'source_dataset': 'da_lmp', 'model_name': 'spp_west_da', 'label': 'Day-ahead'},
}
