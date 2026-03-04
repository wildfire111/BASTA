"""Regression tests for BASTA's Bayesian grid weight system.

The logPDF that BASTA maximises (with weights enabled) is:

    logPDF = log(volume_weight) + log(dage) - 0.5 * chi2

where volume_weight is a scalar across-track weight and dage is a per-model
weight proportional to the evolutionary time step at that point.  Together
they encode how much of stellar parameter space each model represents.  Large
weights can shift the selected track away from the chi2 minimum.

Tests cover two complementary scenarios:

- Tie-breaking: when two tracks both contain a chi2=0 model for the star,
  the larger dage weight determines which track wins.

- Override: when the weight difference is large enough to overcome a real
  chi2 disadvantage, the heavier-weighted track is selected even though it
  contains no model close to the star.  This guards against bugs where weights
  are applied after the chi2 winner has already been determined.
"""
import numpy as np
import pytest

from tests.helpers.bastaharness import BastaHarness
from tests.helpers.models.grid import Grid
from tests.helpers.models.star import Star
from tests.helpers.models.track import Track


@pytest.mark.regression
def test_bayesian_weights_favor_track_with_larger_model_weight(monkeypatch, tmp_path):
    """Bayesian grid weights shift track selection toward the track with higher weight.

    Both tracks have an exact chi2=0 match for the star at index 1 (Teff=5800).
    Both have equal across-track weight: volume_weight=1.0.
    The per-model weight (dage) differs:

        trackA: dage = 0.01  →  logPDF = log(1.0) + log(0.01) + 0 ≈ -4.6
        trackB: dage = 1.0   →  logPDF = log(1.0) + log(1.0)  + 0 =  0.0

    trackB wins because 0.0 > -4.6.
    """
    track_a = Track(
        track_id="trackA",
        values={
            "age":           np.array([1.0,    2.0,    3.0   ]),
            "Teff":          np.array([5700.0, 5800.0, 5900.0]),
            "volume_weight": np.array(1.0),   # scalar dataset (across-track weight)
            "dage":          np.array([0.01,  0.01,   0.01  ]),
        },
    )
    track_b = Track(
        track_id="trackB",
        values={
            "age":           np.array([1.0,    2.0,    3.0   ]),
            "Teff":          np.array([5700.0, 5800.0, 5900.0]),
            "volume_weight": np.array(1.0),   # scalar dataset (across-track weight)
            "dage":          np.array([1.0,   1.0,    1.0   ]),
        },
    )
    star = Star(Teff=5800.0, Teff_err=50.0)
    grid = Grid(
        tracks=[track_a, track_b],
        header_per_track={},
        header_global={},
        active_weights=("volume",),  # tells BASTA to load volume_weight + dage
    )

    harness = BastaHarness(monkeypatch)
    harness.run(
        grid=grid,
        star=star,
        tmp_path=tmp_path,
        usebayw=True,
    )

    maxPDF_path, maxPDF_ind = harness.best_fit_model
    # trackB has dage=1.0 vs trackA's dage=0.01; trackB index 1 wins on logPDF
    assert maxPDF_path == "grid/tracks/trackB"
    assert maxPDF_ind == 1


@pytest.mark.regression
def test_bayesian_weights_override_chi2_preference(monkeypatch, tmp_path):
    """Large per-model dage weights select a track that is NOT the closest chi2 match.

    Setup: two tracks.  track_a contains the exact-match model (Teff=5800,
    chi2=0) but has tiny dage weights (0.01).  track_b has no exact match
    (closest model Teff=5700, chi2=4.0) but has enormous dage weights (100.0).

    Arithmetic:
        logPDF = log(volume_weight) + log(dage) - 0.5 * chi2

        track_a best model (index 2): Teff=5800
            chi2   = 0
            logPDF = log(1.0) + log(0.01) - 0 = 0 + (-4.61) = -4.61

        track_b best model (index 0): Teff=5700
            chi2   = (5700 - 5800)^2 / 50^2 = 4.0
            logPDF = log(1.0) + log(100.0) - 2.0 = 0 + 4.61 - 2.0 = +2.61

    chi2 alone picks track_a index 2 (chi2=0); the dage weight difference
    (+9.22 log-units in favour of track_b) far outweighs the chi2 penalty
    (-2.0 log-units), so track_b index 0 wins by a net margin of 7.22.
    """
    track_a = Track(
        track_id="trackA",
        values={
            "age":           np.array([1.0,    2.0,    3.0   ]),
            "Teff":          np.array([5780.0, 5790.0, 5800.0]),
            "volume_weight": np.array(1.0),
            "dage":          np.array([0.01,   0.01,   0.01  ]),
        },
    )
    track_b = Track(
        track_id="trackB",
        values={
            "age":           np.array([1.0,    2.0,    3.0   ]),
            "Teff":          np.array([5700.0, 5600.0, 5500.0]),
            "volume_weight": np.array(1.0),
            "dage":          np.array([100.0,  100.0,  100.0 ]),
        },
    )
    star = Star(Teff=5800.0, Teff_err=50.0)
    grid = Grid(
        tracks=[track_a, track_b],
        header_per_track={},
        header_global={},
        active_weights=("volume",),
    )

    harness = BastaHarness(monkeypatch)
    harness.run(
        grid=grid,
        star=star,
        tmp_path=tmp_path,
        usebayw=True,
    )

    maxPDF_path, maxPDF_ind = harness.best_fit_model
    # chi2 favours track_a index 2 (exact Teff match), but the enormous dage
    # weights on track_b make track_b index 0 win despite its higher chi2
    assert maxPDF_path == "grid/tracks/trackB"
    assert maxPDF_ind == 0
