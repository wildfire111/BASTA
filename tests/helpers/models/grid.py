from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import h5py

from tests.helpers.models.track import Track

@dataclass(frozen=True)
class Grid:
    #A data class to hold test grid data, to be written to HDF5 and used for testing.

    tracks: List[Track]

    # Header information
    header_per_track: Dict[str, np.ndarray]
    """
    Maps header field name -> array of length n_tracks.
    Example:
        "initial_mass" -> (n_tracks,)
        "metallicity"  -> (n_tracks,)
    """

    header_global: Dict[str, Any]
    """
    Grid-wide metadata that is not per-track.
    Values may be numeric or string (e.g. eta, gcut, pars_constant).
    Example:
        "eta": 0.0
        "pars_constant": "alphaFe"
    """

    solar_models: Dict[str, Dict[str, object]] = field(default_factory=dict)
    """
    Optional group to hold solar model data if necessary for test
    """

    library_type: str = "GarstecTracks"
    version: str = "test"
    buildtime: str = "test"
    active_weights: Optional[Tuple[str, ...]] = None

    def to_hdf5(self, path: str) -> str:
        #Builds an HDF5 file from this Grid instance and returns the path to the file. This is used in tests to create test grid files from Grid instances.
        #Necessary because the current main BASTA main wrapper reads from HDF5 files, so we need a way to create those files from our test data.
        with h5py.File(path, "w") as f:
            #Builds grid and tracks
            grid_group = f.create_group("grid")
            tracks_group = grid_group.create_group("tracks")
            for track in self.tracks:
                track_group = tracks_group.create_group(track.track_id)
                for name, array in track.values.items():
                    track_group.create_dataset(
                        name,
                        data=array,
                        dtype=array.dtype,
                    )
            #Builds header
            header_group = f.create_group("header")
            for name, array in self.header_per_track.items():
                header_group.create_dataset(name, data=array, dtype=array.dtype)
            for name, value in self.header_global.items():
                header_group.create_dataset(name, data=value)
            header_group.create_dataset("library_type", data=self.library_type)
            header_group.create_dataset("version", data=self.version)
            header_group.create_dataset("buildtime", data=self.buildtime)
            if self.active_weights is not None:
                header_group.create_dataset(
                    "active_weights",
                    data=np.array([w.encode("utf-8") for w in self.active_weights]),
                )
            #Builds solar_models (optional)
            if self.solar_models:
                solar_group = f.create_group("solar_models")
                for model_name, datasets in self.solar_models.items():
                    model_group = solar_group.create_group(model_name)
                    for dataset_name, value in datasets.items():
                        model_group.create_dataset(dataset_name, data=value)
        return path
        
            
    def __eq__(self, other) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        
        if len(self.tracks) != len(other.tracks):
            return False
        
        for track_self, track_other in zip(self.tracks, other.tracks):

            if track_self != track_other:
                return False

        if set(self.header_per_track.keys()) != set(other.header_per_track.keys()):
            return False
        
        for key in self.header_per_track.keys():
            if not np.array_equal(self.header_per_track[key], other.header_per_track[key]):
                return False
        
        if self.header_global != other.header_global:
            return False
        
        if self.solar_models != other.solar_models:
            return False

        if self.library_type != other.library_type:
            return False
        if self.version != other.version:
            return False
        if self.buildtime != other.buildtime:
            return False

        if self.active_weights != other.active_weights:
            return False

        return True