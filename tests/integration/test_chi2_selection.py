"""Regression tests for BASTA's chi2-based model selection.

These tests verify that BASTA correctly identifies the best-fit model using
the chi2 statistic — the sum of squared residuals between the observed stellar
parameters and each grid model, weighted by the observational uncertainties:

    chi2 = sum_i [ (obs_i - model_i) / sigma_i ]^2

All tests use BastaHarness to call bastamain.BASTA() directly, bypassing
the XML layer.  Each test builds a minimal synthetic grid and star, runs the
Bayesian fitting, and asserts that the expected model is selected.

Tests cover:
- Single observable, single track: the exact-match model (chi2=0) is selected
- Multiple observables, single track: the exact-match model wins across all observables
- Multiple tracks: the track containing the chi2=0 model beats a distant track
- Star between models: the lower-chi2 model wins when the star falls between grid points
- Joint chi2 with two observables: observation uncertainties correctly weight each dimension,
  so a model that is worse in one observable can still win overall
- Hard parameter limits: models outside user-specified bounds are excluded before chi2
  is evaluated, shifting the best fit to the nearest surviving model
"""
import numpy as np
import pytest

from tests.helpers.bastaharness import BastaHarness
from tests.helpers.builders.starbuilder import StarBuilder
from tests.helpers.models.grid import Grid
from tests.helpers.models.star import Star
from tests.helpers.models.track import Track


@pytest.mark.regression
def test_single_observable_single_track_selects_exact_match_model(monkeypatch, tmp_path):
    """BASTA selects the model with chi2=0 (exact Teff match) as the best fit."""
    track = Track(
        track_id="track001",
        values={
            "age": np.array([1.0, 2.0, 3.0]),
            "Teff": np.array([5700.0, 5800.0, 5900.0]),
        },
    )
    grid = Grid(tracks=[track], header_per_track={}, header_global={})
    star = Star(Teff=5800.0, Teff_err=50.0)

    harness = BastaHarness(monkeypatch)
    harness.run(grid=grid, star=star, tmp_path=tmp_path)

    assert harness.best_fit_model is not None
    _, maxPDF_ind = harness.best_fit_model

    # The 5800 K model (index 1) has chi2=0 — it must be the best fit
    assert maxPDF_ind == 1


@pytest.mark.regression
def test_multi_observable_fit_selects_exact_match(monkeypatch, tmp_path):
    """With 3 observables, the model with chi2=0 across all is selected."""
    track = Track(
        track_id="track001",
        values={
            "age":  np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "Teff": np.array([5600.0, 5700.0, 5800.0, 5900.0, 6000.0]),
            "logg": np.array([4.1,    4.2,    4.3,    4.4,    4.5   ]),
            "FeH":  np.array([-0.3,   -0.1,    0.1,    0.2,    0.3  ]),
        },
    )
    # Star exactly matches model at index 2
    star = StarBuilder().make_star_equal_to_track_index(track, index=2)
    grid = Grid(tracks=[track], header_per_track={}, header_global={})

    harness = BastaHarness(monkeypatch)
    harness.run(grid=grid, star=star, tmp_path=tmp_path)

    _, maxPDF_ind = harness.best_fit_model
    assert maxPDF_ind == 2


@pytest.mark.regression
def test_two_tracks_selects_track_containing_exact_match_model(monkeypatch, tmp_path):
    """BASTA picks the track containing the exact-match model (chi2=0)."""
    track_a = Track(
        track_id="trackA",
        values={
            "age": np.array([1.0, 2.0, 3.0]),
            "Teff": np.array([5700.0, 5800.0, 5900.0]),
        },
    )
    track_b = Track(
        track_id="trackB",
        values={
            "age": np.array([1.0, 2.0, 3.0]),
            "Teff": np.array([5900.0, 6000.0, 6100.0]),
        },
    )
    # Star matches track_a at index 1 exactly; track_b has no close match
    star = StarBuilder().make_star_equal_to_track_index(track_a, index=1)

    grid = Grid(
        tracks=[track_a, track_b],
        header_per_track={},
        header_global={},
    )
    harness = BastaHarness(monkeypatch)
    harness.run(grid=grid, star=star, tmp_path=tmp_path)

    maxPDF_path, maxPDF_ind = harness.best_fit_model
    assert maxPDF_path == "grid/tracks/trackA"
    assert maxPDF_ind == 1


@pytest.mark.regression
def test_star_between_models_selects_nearest_chi2(monkeypatch, tmp_path):
    """When the star falls between two models, the lower-chi2 model is selected."""
    track = Track(
        track_id="track001",
        values={
            "age":  np.array([1.0,    2.0,    3.0,    4.0,    5.0   ]),
            "Teff": np.array([5700.0, 5750.0, 5800.0, 5850.0, 5900.0]),
        },
    )
    builder = StarBuilder()
    # Start exactly halfway between index 2 (5800 K) and index 3 (5850 K): Teff = 5825 K.
    # Shift 5 K toward index 2 so the star is closer to index 2 than index 3.
    star_midpoint = builder.make_star_between_track_indices(track, index1=2, index2=3)
    star = builder.shift_star(star_midpoint, Teff=-5.0)

    harness = BastaHarness(monkeypatch)
    harness.run(
        grid=Grid(tracks=[track], header_per_track={}, header_global={}),
        star=star,
        tmp_path=tmp_path,
    )

    _, maxPDF_ind = harness.best_fit_model
    assert maxPDF_ind == 2  # Teff=5800 wins over Teff=5850


@pytest.mark.regression
def test_two_observables_two_tracks_selects_track_containing_exact_match_model(monkeypatch, tmp_path):
    """With two observables (Teff+FeH) and two tracks, BASTA picks the track with chi2=0."""
    # track_a contains the model that exactly matches the star (index 1: Teff=5800, FeH=0.0)
    track_a = Track(
        track_id="trackA",
        values={
            "age": np.array([1.0,    2.0,    3.0   ]),
            "Teff": np.array([5700.0, 5800.0, 5900.0]),
            "FeH":  np.array([-0.1,    0.0,    0.1  ]),
        },
    )
    # track_b has no model close to the star (all Teff and FeH values are far off)
    track_b = Track(
        track_id="trackB",
        values={
            "age": np.array([1.0,    2.0,    3.0   ]),
            "Teff": np.array([5900.0, 6000.0, 6100.0]),
            "FeH":  np.array([ 0.1,    0.2,    0.3  ]),
        },
    )
    # Star exactly matches track_a at index 1: Teff=5800, FeH=0.0
    star = StarBuilder().make_star_equal_to_track_index(track_a, index=1)

    grid = Grid(tracks=[track_a, track_b], header_per_track={}, header_global={})
    harness = BastaHarness(monkeypatch)
    harness.run(grid=grid, star=star, tmp_path=tmp_path)

    maxPDF_path, maxPDF_ind = harness.best_fit_model
    # track_a, index 1 has joint chi2=0 across both observables; track_b has no close model
    assert maxPDF_path == "grid/tracks/trackA"
    assert maxPDF_ind == 1


@pytest.mark.regression
def test_two_observables_between_models_selects_lowest_joint_chi2(monkeypatch, tmp_path):
    """With two observables, BASTA picks the model minimising joint chi2, not per-observable chi2.

    The star's tight FeH constraint means the model closer in FeH wins,
    even though it has a worse Teff match — joint chi2 correctly weights both observables.
    """
    track = Track(
        track_id="track001",
        values={
            "age":  np.array([1.0,    2.0,    3.0,    4.0,    5.0   ]),
            "Teff": np.array([5700.0, 5750.0, 5800.0, 5850.0, 5900.0]),
            "FeH":  np.array([-0.2,   -0.1,    0.0,    0.1,    0.2  ]),
        },
    )
    # Star has Teff exactly matching index 2 (5800 K) but FeH matching index 3 (0.1).
    # The tight FeH_err (0.01) means FeH drives the chi2.
    #
    # Chi2 at each relevant model:
    #   index 2 (Teff=5800, FeH=0.0): chi2 = 0 + (0.1/0.01)^2 = 100.0   (exact Teff, bad FeH)
    #   index 3 (Teff=5850, FeH=0.1): chi2 = (50/50)^2 + 0    = 1.0     (imperfect Teff, exact FeH)
    #
    # Index 3 wins because the FeH benefit far outweighs the Teff penalty.
    star = Star(Teff=5800.0, Teff_err=50.0, FeH=0.1, FeH_err=0.01)

    harness = BastaHarness(monkeypatch)
    harness.run(
        grid=Grid(tracks=[track], header_per_track={}, header_global={}),
        star=star,
        tmp_path=tmp_path,
    )

    _, maxPDF_ind = harness.best_fit_model
    assert maxPDF_ind == 3  # FeH match at index 3 beats perfect Teff match at index 2


@pytest.mark.regression
def test_two_observables_no_exact_match_picks_model_with_lowest_joint_chi2(monkeypatch, tmp_path):
    """With two observables and no exact match, BASTA picks the model with lowest joint chi2.

    The star sits between two models on both Teff and FeH, but the relative
    uncertainties mean the FeH dimension dominates: the model that is further
    away in Teff but much closer in FeH wins overall.
    """
    track = Track(
        track_id="track001",
        values={
            "age":  np.array([1.0,    2.0,    3.0,    4.0,    5.0   ]),
            "Teff": np.array([5700.0, 5750.0, 5800.0, 5850.0, 5900.0]),
            "FeH":  np.array([-0.2,   -0.1,    0.0,    0.1,    0.2  ]),
        },
    )
    # Star is between index 2 and index 3 on both observables.
    # It is closer to index 2 in Teff (10 K off vs 40 K) but much closer to
    # index 3 in FeH (0.02 off vs 0.08). The tight FeH_err (0.04) means the
    # FeH dimension dominates, so index 3 wins despite its worse Teff match.
    #
    # Chi2 at each relevant model:
    #   index 2 (Teff=5800, FeH=0.0): (10/50)^2 + (0.08/0.04)^2 = 0.04 + 4.00 = 4.04
    #   index 3 (Teff=5850, FeH=0.1): (40/50)^2 + (0.02/0.04)^2 = 0.64 + 0.25 = 0.89
    star = Star(Teff=5810.0, Teff_err=50.0, FeH=0.08, FeH_err=0.04)

    harness = BastaHarness(monkeypatch)
    harness.run(
        grid=Grid(tracks=[track], header_per_track={}, header_global={}),
        star=star,
        tmp_path=tmp_path,
    )

    _, maxPDF_ind = harness.best_fit_model
    assert maxPDF_ind == 3  # better FeH fit wins despite worse Teff fit


@pytest.mark.regression
def test_hard_limit_excludes_models_outside_teff_bounds(monkeypatch, tmp_path):
    """Models outside a hard Teff limit are excluded, shifting the best fit to the nearest surviving model.

    Without limits, the star (Teff=5600) would match index 1 exactly (chi2=0).
    The limit [5750, inf] cuts indices 0 (Teff=5500) and 1 (Teff=5600) because
    they fall below 5750 K.  Of the surviving models, index 2 (Teff=5900) is
    closest to the star and therefore wins.
    """
    track = Track(
        track_id="track001",
        values={
            "age":  np.array([1.0,    2.0,    3.0,    4.0,    5.0   ]),
            "Teff": np.array([5500.0, 5600.0, 5900.0, 6000.0, 6100.0]),
        },
    )
    star = Star(Teff=5600.0, Teff_err=50.0)
    grid = Grid(tracks=[track], header_per_track={}, header_global={})

    harness = BastaHarness(monkeypatch)
    harness.run(
        grid=grid,
        star=star,
        tmp_path=tmp_path,
        overrides={"limits": {"Teff": [5750.0, np.inf, np.inf, np.inf]}},
    )

    _, maxPDF_ind = harness.best_fit_model
    # Indices 0 (5500 K) and 1 (5600 K) are below the 5750 K lower bound and are cut.
    # Index 2 (Teff=5900) is the closest surviving model and wins.
    assert maxPDF_ind == 2
