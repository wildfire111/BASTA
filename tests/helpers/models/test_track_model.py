import pytest
from tests.helpers.models.track import Track
import numpy as np

#Check the construction of the track equality method, which is used in tests to compare expected vs actual data. 
#These are not trivial because they involve comparing numpy arrays, which require special handling.

def test_track_equality():
    track1 = Track(
        track_id="track1",
        values={
            "age": np.array([0, 1, 2]),
            "luminosity": np.array([1, 2, 3]),
        }
    )

    track2 = Track(
        track_id="track1",
        values={
            "age": np.array([0, 1, 2]),
            "luminosity": np.array([1, 2, 3]),
        }
    )

    track3 = Track(
        track_id="track3",
        values={
            "age": np.array([0, 1, 2]),
            "luminosity": np.array([1, 2, 3]),
        }
    )

    track4 = Track(
        track_id="track1",
        values={
            "age": np.array([0, 1, 2]),
            "luminosity": np.array([1, 2, 4]),  # Different luminosity values
        }
    )

    assert track1 == track2
    assert track1 != track3
    assert track1.values_equal(track3)
    assert track1.values_equal(track4) == False