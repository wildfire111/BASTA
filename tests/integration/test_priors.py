"""Regression tests for BASTA's IMF prior weighting.

The logPDF that BASTA maximises (in the no-weights case) is:

    logPDF = log(prior(massini)) - 0.5 * chi2

where prior(massini) is an Initial Mass Function (IMF) evaluated at the
initial stellar mass of each grid model.  The prior adds a log-probability
term that favours lower-mass stars, and can shift the selected model away
from the chi2 minimum.

Tests cover two complementary scenarios:

- Tie-breaking: when every model has chi2=0 (all models match the star
  equally in observable space), the prior alone determines which model is
  selected.  Under Salpeter (1955), the lowest-mass model always wins.

- Override: when the prior weight difference is large enough to overcome a
  real chi2 disadvantage, the model that is NOT the closest match to the star
  is selected.
"""
import numpy as np
import pytest

from tests.helpers.bastaharness import BastaHarness
from tests.helpers.models.grid import Grid
from tests.helpers.models.star import Star
from tests.helpers.models.track import Track


@pytest.mark.regression
def test_prior_shifts_selection_when_chi2_is_tied(monkeypatch, tmp_path):
    """An IMF prior changes which model wins when all chi2 values are equal.

    All three models have Teff=5800 so the star (Teff=5800) has chi2=0 against
    every model.  With a Salpeter (1955) prior, the log-PDF includes an extra
    term log(massini ** -2.35):

        index 0: log(1.0 ** -2.35) =  0.00
        index 1: log(2.0 ** -2.35) ≈ -1.63
        index 2: log(4.0 ** -2.35) ≈ -3.26

    Index 0 (massini=1.0) has the highest logPDF and is selected.
    """
    track = Track(
        track_id="track001",
        values={
            "age":     np.array([1.0, 2.0, 3.0]),
            "Teff":    np.array([5800.0, 5800.0, 5800.0]),
            "massini": np.array([1.0,    2.0,    4.0   ]),
        },
    )
    star = Star(Teff=5800.0, Teff_err=50.0)
    grid = Grid(tracks=[track], header_per_track={}, header_global={})

    harness = BastaHarness(monkeypatch)
    harness.run(
        grid=grid,
        star=star,
        tmp_path=tmp_path,
        usepriors=("salpeter1955",),
    )

    _, maxPDF_ind = harness.best_fit_model
    # chi2=0 for all; salpeter1955 gives the highest weight to massini=1.0 (index 0)
    assert maxPDF_ind == 0


@pytest.mark.regression
def test_prior_overrides_chi2_preference(monkeypatch, tmp_path):
    """The IMF prior selects a model that is NOT the closest chi2 match.

    Setup: two models in one track.  The star (Teff=5800) matches index 1
    exactly (chi2=0), but index 1 has a very high mass (massini=8.0) that is
    strongly penalised by the Salpeter (1955) IMF.  Index 0 (massini=0.5) has
    a higher chi2 but a much better prior weight, so it wins overall.

    Arithmetic:
        logPDF = log(massini ** -2.35) - 0.5 * chi2

        index 0: Teff=5700, massini=0.5
            chi2      = (5700 - 5800)^2 / 50^2 = 4.0
            log-prior = log(0.5 ** -2.35) = 2.35 * log(2) ≈ +1.63
            logPDF    = 1.63 - 2.0 = -0.37

        index 1: Teff=5800, massini=8.0
            chi2      = 0
            log-prior = log(8.0 ** -2.35) = -2.35 * log(8) ≈ -4.89
            logPDF    = -4.89 - 0 = -4.89

    chi2 alone picks index 1 (chi2=0); the prior contribution (+6.52 in favour
    of index 0) more than offsets the chi2 penalty (+2.0 in favour of index 1),
    so index 0 wins by a net margin of 4.52 log-units.
    """
    track = Track(
        track_id="track001",
        values={
            "age":     np.array([1.0, 2.0]),
            "Teff":    np.array([5700.0, 5800.0]),
            "massini": np.array([0.5,    8.0   ]),
        },
    )
    star = Star(Teff=5800.0, Teff_err=50.0)
    grid = Grid(tracks=[track], header_per_track={}, header_global={})

    harness = BastaHarness(monkeypatch)
    harness.run(
        grid=grid,
        star=star,
        tmp_path=tmp_path,
        usepriors=("salpeter1955",),
    )

    _, maxPDF_ind = harness.best_fit_model
    # chi2 favours index 1 (exact Teff match), but the Salpeter IMF so
    # strongly penalises massini=8.0 that index 0 (massini=0.5) wins instead
    assert maxPDF_ind == 0
