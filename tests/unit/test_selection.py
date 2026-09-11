"""Invariant tests for src/selection.py — the objective-mode table.

The table decides how models rank at three separate points (the Optuna study,
the top-N bake, the promote gate), so a malformed entry does not fail loudly —
it silently ranks by something nobody chose. These guard the shapes that make
the ranking meaningful: aligned intervals/scalers, bands the models are
actually trained to emit, and a score arity that matches the study's objective
count.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import selection


class TestObjectiveTable:
    def test_default_objective_is_defined(self):
        assert selection.DEFAULT_OBJECTIVE in selection.OBJECTIVES

    def test_each_mode_has_required_fields(self):
        for name, mode in selection.OBJECTIVES.items():
            assert set(mode) == {'metrics', 'intervals', 'scalers'}, name
            assert mode['metrics'], f'{name} must rank on at least one metric'

    def test_intervals_and_scalers_are_aligned(self):
        # A misaligned pair would apply a band's weight to a different band —
        # a wrong ranking that nothing else would catch.
        for name, mode in selection.OBJECTIVES.items():
            assert len(mode['intervals']) == len(mode['scalers']), name

    def test_calibration_modes_declare_ci_err_metric(self):
        # Bands without a ci_err metric would be scored and then discarded.
        for name, mode in selection.OBJECTIVES.items():
            has_bands = bool(mode['intervals'])
            assert has_bands == ('ci_err' in mode['metrics']), name

    def test_bands_are_ordered_and_within_unit_interval(self):
        for name, mode in selection.OBJECTIVES.items():
            for q_low, q_high in mode['intervals']:
                assert 0.0 < q_low < q_high < 1.0, name

    def test_restored_total_calibration_weight(self):
        # The formula being restored was `MAE + 0.5 * ci_err` on one band; the
        # per-band weights are a split of that same total, not a multiple of it.
        for name in ('mae_ci', 'crps_ci'):
            assert sum(selection.OBJECTIVES[name]['scalers']) == pytest.approx(0.5)


class TestBandsAreTrainedQuantiles:
    """Every scored band edge must be a level the models actually emit."""

    def _quantiles(self):
        # Imported lazily: parameters pulls in sklearn/darts, which the rest of
        # this module deliberately does not need.
        import parameters

        return parameters.QUANTILES

    def test_diagnostic_band_edges_are_trained_levels(self):
        q = self._quantiles()
        for q_low, q_high in selection.DIAGNOSTIC_BANDS:
            assert q_low in q and q_high in q

    def test_mode_band_edges_are_trained_levels(self):
        q = self._quantiles()
        for name, mode in selection.OBJECTIVES.items():
            for q_low, q_high in mode['intervals']:
                assert q_low in q and q_high in q, name

    def test_mode_bands_are_recorded_as_diagnostics(self):
        # Re-ranking a finished study under a different mode only works if
        # every band a mode ranks on is also recorded on every trial, in every
        # mode — DIAGNOSTIC_BANDS is what gets recorded.
        for name, mode in selection.OBJECTIVES.items():
            for band in mode['intervals']:
                assert band in selection.DIAGNOSTIC_BANDS, name


class TestBandHelpers:
    def test_nominal_is_band_width(self):
        assert selection.band_nominal((0.1, 0.9)) == pytest.approx(0.8)
        assert selection.band_nominal((0.025, 0.975)) == pytest.approx(0.95)

    def test_labels_are_distinct_per_diagnostic_band(self):
        labels = [selection.band_label(b) for b in selection.DIAGNOSTIC_BANDS]
        assert len(labels) == len(set(labels))

    def test_label_names_the_nominal_percent(self):
        assert selection.band_label((0.1, 0.9)) == 'ci_err_80'
        assert selection.band_label((0.025, 0.975)) == 'ci_err_95'

    def test_coverage_error_is_percentage_points_from_nominal(self):
        assert selection.coverage_error(0.8, (0.1, 0.9)) == pytest.approx(0.0)
        assert selection.coverage_error(0.75, (0.1, 0.9)) == pytest.approx(5.0)
        # Over-coverage is penalized as much as under-coverage.
        assert selection.coverage_error(0.85, (0.1, 0.9)) == pytest.approx(5.0)


class TestWeightedCiErr:
    def test_weights_each_band_by_its_scaler(self):
        mode = selection.resolve_mode('mae_ci')
        # 5 points off at 80%, 3 points off at 90%, both weighted 0.25.
        coverages = {(0.1, 0.9): 0.75, (0.05, 0.95): 0.87}
        assert selection.weighted_ci_err(coverages, mode) == pytest.approx(2.0)

    def test_perfect_calibration_scores_zero(self):
        mode = selection.resolve_mode('mae_ci')
        coverages = {(0.1, 0.9): 0.8, (0.05, 0.95): 0.9}
        assert selection.weighted_ci_err(coverages, mode) == pytest.approx(0.0)

    def test_extra_diagnostic_bands_are_ignored(self):
        mode = selection.resolve_mode('mae_ci')
        coverages = {b: 0.5 for b in selection.DIAGNOSTIC_BANDS}
        coverages.update({(0.1, 0.9): 0.8, (0.05, 0.95): 0.9})
        assert selection.weighted_ci_err(coverages, mode) == pytest.approx(0.0)

    def test_mode_without_bands_scores_zero(self):
        mode = selection.resolve_mode('crps')
        assert selection.weighted_ci_err({}, mode) == pytest.approx(0.0)

    def test_missing_ranked_band_raises(self):
        mode = selection.resolve_mode('mae_ci')
        with pytest.raises(KeyError):
            selection.weighted_ci_err({(0.1, 0.9): 0.8}, mode)


class TestSelectionScore:
    def test_two_metric_mode_sums_accuracy_and_calibration(self):
        mode = selection.resolve_mode('mae_ci')
        assert selection.selection_score((6.09, 1.5), mode) == pytest.approx(7.59)

    def test_single_metric_mode_is_the_metric(self):
        mode = selection.resolve_mode('crps')
        assert selection.selection_score((4.43,), mode) == pytest.approx(4.43)

    def test_lower_is_better_for_both_terms(self):
        mode = selection.resolve_mode('mae_ci')
        base = selection.selection_score((6.0, 1.0), mode)
        assert selection.selection_score((7.0, 1.0), mode) > base
        assert selection.selection_score((6.0, 2.0), mode) > base

    def test_arity_mismatch_raises(self):
        # A single-objective study's trials being ranked under a two-metric
        # mode: the order would be meaningless, so refuse rather than guess.
        mode = selection.resolve_mode('mae_ci')
        with pytest.raises(ValueError):
            selection.selection_score((4.43,), mode)
        with pytest.raises(ValueError):
            selection.selection_score((6.09, 1.5), selection.resolve_mode('crps'))


class TestResolveMode:
    def test_returns_the_mode_config(self):
        assert selection.resolve_mode('mae_ci') is selection.OBJECTIVES['mae_ci']

    def test_unknown_mode_raises_and_lists_valid_names(self):
        with pytest.raises(ValueError) as exc:
            selection.resolve_mode('mae_ci80')
        assert 'mae_ci' in str(exc.value)

    def test_env_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(selection.OBJECTIVE_ENV_VAR, raising=False)
        assert selection.mode_name_from_env() == selection.DEFAULT_OBJECTIVE

    def test_env_override_is_read(self, monkeypatch):
        monkeypatch.setenv(selection.OBJECTIVE_ENV_VAR, 'crps')
        assert selection.mode_name_from_env() == 'crps'


class TestStudyName:
    """The study-name format has one home; the notebook and the bake CLI
    both build names from it, and a drift between them would silently read
    the wrong study."""

    def test_includes_target_architecture_and_mode(self):
        assert (
            selection.study_name('spp_west_da', 'tide', 'mae_ci')
            == 'spp_west_da_tide_mae_ci'
        )

    def test_mode_distinguishes_otherwise_identical_studies(self):
        # Trials scored under different objectives are not comparable, and
        # Optuna silently ignores a changed `directions` on an existing study,
        # so the names must not collide.
        names = {
            selection.study_name('spp_west_da', 'tide', m)
            for m in selection.OBJECTIVES
        }
        assert len(names) == len(selection.OBJECTIVES)


class TestWeightCiErrs:
    """The weighting step shared by the study (holds errors) and the gate
    (holds coverages)."""

    def test_weights_each_band_by_its_scaler(self):
        mode = selection.resolve_mode('mae_ci')
        errs = {(0.1, 0.9): 5.0, (0.05, 0.95): 3.0}
        assert selection.weight_ci_errs(errs, mode) == pytest.approx(2.0)

    def test_agrees_with_the_coverage_side_wrapper(self):
        # The two entry points must produce the same number, or the study
        # would optimize a differently-weighted term than the gate decides on.
        mode = selection.resolve_mode('mae_ci')
        coverages = {(0.1, 0.9): 0.75, (0.05, 0.95): 0.87}
        errs = {
            band: selection.coverage_error(cov, band)
            for band, cov in coverages.items()
        }
        assert selection.weight_ci_errs(errs, mode) == pytest.approx(
            selection.weighted_ci_err(coverages, mode)
        )

    def test_mode_without_bands_scores_zero(self):
        assert selection.weight_ci_errs({}, selection.resolve_mode('crps')) == 0.0

    def test_misaligned_mode_raises_rather_than_dropping_a_band(self):
        # zip(strict=True): a 3-band/2-weight mode must not silently score two.
        bad = {'metrics': ('mae', 'ci_err'),
               'intervals': ((0.1, 0.9), (0.05, 0.95), (0.25, 0.75)),
               'scalers': (0.25, 0.25)}
        with pytest.raises(ValueError):
            selection.weight_ci_errs({b: 1.0 for b in bad['intervals']}, bad)


class TestTopN:
    def test_is_a_positive_int(self):
        assert isinstance(selection.TOP_N, int) and selection.TOP_N > 0

    def test_parameters_reexports_the_same_object(self):
        # One home: the bake CLI reads selection.TOP_N (it cannot import
        # parameters), the retrain slices parameters.TOP_N. If these diverged
        # the bake would write N params and the retrain train a different N.
        import parameters

        assert parameters.TOP_N is selection.TOP_N
