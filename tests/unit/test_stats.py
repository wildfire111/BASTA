import numpy as np

from basta.stats import Trackstats, lowest_chi2, most_likely


def test_most_likely_single_passing_model_returns_its_original_index():
    """With one track and one passing model, returns that model's original index."""
    selectedmodels = {
        "track/a": Trackstats(
            index=np.array([False, False, True]),
            logPDF=np.array([-1.0]),
            chi2=np.array([2.0]),
        )
    }
    best_track, best_index = most_likely(selectedmodels)
    assert best_track == "track/a"
    assert best_index == 2


def test_most_likely_single_track_returns_model_with_highest_logPDF():
    """With multiple passing models, returns the one with highest logPDF."""
    # All three models pass; logPDF is highest at filtered position 1 → original index 1
    selectedmodels = {
        "track/a": Trackstats(
            index=np.array([True, True, True]),
            logPDF=np.array([-5.0, -1.0, -3.0]),
            chi2=np.array([10.0, 2.0, 6.0]),
        )
    }
    best_track, best_index = most_likely(selectedmodels)
    assert best_track == "track/a"
    assert best_index == 1


def test_most_likely_returns_model_with_globally_highest_logPDF():
    """Best model is found across tracks, not just within one."""
    selectedmodels = {
        "track/a": Trackstats(
            index=np.array([True, True]),
            logPDF=np.array([-5.0, -3.0]),
            chi2=np.array([10.0, 6.0]),
        ),
        "track/b": Trackstats(
            index=np.array([True, True]),
            logPDF=np.array([-2.0, -4.0]),
            chi2=np.array([4.0, 8.0]),
        ),
    }
    best_track, best_index = most_likely(selectedmodels)
    assert best_track == "track/b"
    assert best_index == 0


def test_sparse_index_maps_to_original_position():
    """The returned index is in original-track coordinates, not filtered coordinates."""
    # index: [F, F, T, F, T] → passing models at original indices 2 and 4
    # logPDF[1] is higher → that corresponds to original index 4
    selectedmodels = {
        "track/a": Trackstats(
            index=np.array([False, False, True, False, True]),
            logPDF=np.array([-5.0, -1.0]),
            chi2=np.array([10.0, 2.0]),
        )
    }
    best_track, best_index = most_likely(selectedmodels)
    assert best_track == "track/a"
    assert best_index == 4


def test_most_likely_sparse_index_resolved_correctly_when_best_model_is_in_later_track():
    """Sparse indices are resolved correctly when the best model is in a later track."""
    # track/a: passes at original indices 0 and 3, best logPDF = -4.0
    # track/b: passes at original indices 1 and 4, best logPDF = -0.5 (global best)
    selectedmodels = {
        "track/a": Trackstats(
            index=np.array([True, False, False, True]),
            logPDF=np.array([-6.0, -4.0]),
            chi2=np.array([12.0, 8.0]),
        ),
        "track/b": Trackstats(
            index=np.array([False, True, False, False, True]),
            logPDF=np.array([-3.0, -0.5]),
            chi2=np.array([6.0, 1.0]),
        ),
    }
    best_track, best_index = most_likely(selectedmodels)
    assert best_track == "track/b"
    assert best_index == 4


def test_lowest_chi2_single_track():
    """With multiple passing models, returns the one with lowest chi2."""
    selectedmodels = {
        "track/a": Trackstats(
            index=np.array([True, True, True]),
            logPDF=np.array([-5.0, -1.0, -3.0]),
            chi2=np.array([10.0, 2.0, 6.0]),
        )
    }
    best_track, best_index = lowest_chi2(selectedmodels)
    assert best_track == "track/a"
    assert best_index == 1


def test_lowest_chi2_finds_global_minimum_across_multiple_tracks():
    """Finds global minimum chi2 across multiple tracks."""
    selectedmodels = {
        "track/a": Trackstats(
            index=np.array([True, True]),
            logPDF=np.array([-5.0, -3.0]),
            chi2=np.array([10.0, 6.0]),
        ),
        "track/b": Trackstats(
            index=np.array([True, True]),
            logPDF=np.array([-2.0, -4.0]),
            chi2=np.array([4.0, 1.5]),
        ),
    }
    best_track, best_index = lowest_chi2(selectedmodels)
    assert best_track == "track/b"
    assert best_index == 1


def test_lowest_chi2_sparse_index():
    """Sparse bool index maps back to original-track position."""
    # index: [F, F, T, F, T] → passing models at original indices 2 and 4
    # chi2[1] is lower → that corresponds to original index 4
    selectedmodels = {
        "track/a": Trackstats(
            index=np.array([False, False, True, False, True]),
            logPDF=np.array([-5.0, -1.0]),
            chi2=np.array([10.0, 2.0]),
        )
    }
    best_track, best_index = lowest_chi2(selectedmodels)
    assert best_track == "track/a"
    assert best_index == 4
