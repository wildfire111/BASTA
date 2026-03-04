import pytest
from tests.helpers.models.grid import Grid
from tests.helpers.models.track import Track
import numpy as np
import h5py

#Check the construction of the grid equality method, which is used in tests to compare expected vs actual data. 
#These are not trivial because they involve comparing numpy arrays, which require special handling.

def test_grid_equality():
    track1 = Track(
        track_id="track1",
        values={
            "age": np.array([0, 1, 2]),
            "luminosity": np.array([1, 2, 3]),
        }
    )

    track2 = Track(
        track_id="track2",
        values={
            "age": np.array([0, 1, 2]),
            "luminosity": np.array([4, 5, 6]),
        }
    )

    track3 = Track(
        track_id="track3",
        values={
            "age": np.array([0, 1, 2]),
            "luminosity": np.array([4, 5, 6]),
        }
    )

    grid1 = Grid(
        tracks=[track1, track2],
        header_per_track={"initial_mass": np.array([1.0, 1.5])},
        header_global={"mixing_length": 1.8},
    )

    grid2 = Grid(
        tracks=[track1, track2],
        header_per_track={"initial_mass": np.array([1.0, 1.5])},
        header_global={"mixing_length": 1.8},
    )

    grid3 = Grid(
        tracks=[track1, track3],  #Same values but different track_id
        header_per_track={"initial_mass": np.array([1.0, 1.5])},
        header_global={"mixing_length": 1.8},
    )

    grid4 = Grid(
        tracks=[track1, track2],
        header_per_track={"initial_mass": np.array([1.0, 1.5])},
        header_global={"mixing_length": 1.9},  #Different global value
    )

    grid5 = Grid(
        tracks=[track1, track2],
        header_per_track={"initial_mass": np.array([2.0, 2.5])},  #Different header_per_track values
        header_global={"mixing_length": 1.8},
    )

    assert grid1 == grid2
    assert grid1 != grid3
    assert grid1 != grid4
    assert grid1 != grid5

def test_grid_to_hdf5(tmp_path):
    track1 = Track(
        track_id="track1",
        values={
            "age": np.array([0, 1, 2]),
            "luminosity": np.array([1, 2, 3]),
        }
    )

    grid = Grid(
        tracks=[track1],
        header_per_track={"initial_mass": np.array([1.0])},
        header_global={"mixing_length": 1.8},
    )

    hdf5_path = tmp_path / "test_grid.hdf5"
    grid.to_hdf5(hdf5_path)

    # Read back the HDF5 file and check contents
    with h5py.File(hdf5_path, "r") as f:
        assert "grid" in f
        assert "tracks" in f["grid"]
        assert "track1" in f["grid"]["tracks"]
        track_group = f["grid"]["tracks"]["track1"]
        assert np.array_equal(track_group["age"][:], np.array([0, 1, 2]))
        assert np.array_equal(track_group["luminosity"][:], np.array([1, 2, 3]))
        
        assert "header" in f
        header_group = f["header"]
        assert np.array_equal(header_group["initial_mass"][:], np.array([1.0]))
        assert header_group["mixing_length"][()] == 1.8