"""Invariant tests for src/parameters.py constants.

QUANTILES is the single home for the QuantileRegression quantile set, and
three consumers select specific levels by exact float lookup — the eval
harness (0.05/0.5/0.95), the scored bands (0.1/0.9), and the app/plotting
(0.1/0.5/0.9). These guard against an edit that drops, reorders, or drifts a
required level.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import parameters


class TestQuantiles:
    def test_strictly_ascending_no_duplicates(self):
        q = parameters.QUANTILES
        assert q == sorted(q)
        assert len(q) == len(set(q))

    def test_within_open_unit_interval(self):
        assert all(0.0 < x < 1.0 for x in parameters.QUANTILES)

    def test_contains_levels_downstream_code_selects(self):
        # exact-float lookups in the harness / scored bands / plotting
        for level in (0.05, 0.1, 0.5, 0.9, 0.95):
            assert level in parameters.QUANTILES

    def test_symmetric_about_median(self):
        q = set(parameters.QUANTILES)
        assert all(round(1 - x, 6) in q for x in q)


class TestTargets:
    def test_default_target_is_defined(self):
        assert parameters.DEFAULT_TARGET in parameters.TARGETS

    def test_each_target_has_required_fields(self):
        for cfg in parameters.TARGETS.values():
            assert 'source_dataset' in cfg
            assert 'model_name' in cfg

    def test_rt_and_da_targets_present_and_distinct(self):
        rt, da = parameters.TARGETS['rt'], parameters.TARGETS['da']
        assert rt['source_dataset'] == 'lmp'
        assert da['source_dataset'] == 'da_lmp'
        assert rt['model_name'] != da['model_name']

    def test_model_name_matches_default_target(self):
        assert parameters.MODEL_NAME == parameters.TARGETS[parameters.DEFAULT_TARGET]['model_name']


class TestClipQuantiles:
    """Training-data clipping is per target, and one home for each.

    Hyperparameters selected against a tail-suppressed distribution and then
    trained on the raw one are params chosen for a dataset that was never
    served. The study and the retrain previously disagreed — the study clipped,
    the retrain did not — with nothing to catch it.
    """

    def test_every_target_declares_clip_quantiles(self):
        for name, cfg in parameters.TARGETS.items():
            assert 'clip_quantiles' in cfg, name

    def test_bounds_are_ordered_and_within_unit_interval(self):
        for name, cfg in parameters.TARGETS.items():
            q = cfg['clip_quantiles']
            if q is None:  # None is a valid "train on raw prices"
                continue
            lo, hi = q
            assert 0.0 < lo < hi < 1.0, name

    def test_bounds_are_symmetric(self):
        # An asymmetric clip shifts the training distribution's centre, biasing
        # the median the app plots.
        for name, cfg in parameters.TARGETS.items():
            q = cfg['clip_quantiles']
            if q is None:
                continue
            assert round(q[0] + q[1], 9) == 1.0, name

    def test_training_notebooks_read_the_targets_own_bounds(self):
        # Both must read TARGETS[TARGET]['clip_quantiles'], and neither may
        # hardcode bounds locally — a second home is how they drifted before.
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        for name in ('model.py', 'model_retrain.py'):
            src = (root / 'notebooks' / 'model_training' / name).read_text()
            assert 'clip_quantiles=parameters.TARGETS[TARGET]["clip_quantiles"]' in src, name
            assert 'CLIP_OUTLIERS' not in src, f'{name} references a retired constant'

    def test_both_training_call_sites_clip(self):
        # prep_all_df and get_train_test_all both feed training data, so both
        # need the bounds in each notebook.
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        for name in ('model.py', 'model_retrain.py'):
            src = (root / 'notebooks' / 'model_training' / name).read_text()
            assert src.count('clip_quantiles=parameters.TARGETS') == 2, name

    def test_clipping_is_off_by_default_so_display_paths_stay_raw(self):
        # The app calls prep_lmp for the actuals it PLOTS; a clipping default
        # would hide real spikes from users.
        import inspect

        import data_engineering as de

        for fn in (de.prep_lmp, de.prep_all_df, de.get_train_test_all):
            assert inspect.signature(fn).parameters['clip_quantiles'].default is None
