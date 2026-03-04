from tests.helpers.models import grid, track
from typing import Any, Dict
import numpy as np

class GridBuilder:
    #A helper class to build test grid data for use in tests.

    def __init__(self):
        self.tracks = []
        self.header_per_track = {}
        self.header_global = {}
        self.solar_models = {}

    def add_track(self, track_id: str, values: Dict[str, np.ndarray]):

        #Track must have at least one timestep
        if len(values) == 0:
            raise ValueError("Track values must contain at least one timestep.")

        new_track = track.Track(track_id=track_id, values=values)
        self.tracks.append(new_track)

    def set_header_per_track(self, header_name: str, values: np.ndarray):

        #Header must have at least one value
        if len(values) == 0:
            raise ValueError("Header values must contain at least one value.")
        
        self.header_per_track[header_name] = values

    def set_header_global(self, name: str, value: Any):

        #Value must not be None
        if value is None:
            raise ValueError("Header global value cannot be None.")

        self.header_global[name] = value

    def add_solar_model(self, model_name: str, model_data: Dict[str, object]):
        self.solar_models[model_name] = model_data

    def build(self) -> grid.Grid:
        #Check that we have at least one track
        if len(self.tracks) == 0:
            raise ValueError("At least one track must be added to the grid.")
        
        #Check that all tracks have the same set of value keys
        value_keys = set(self.tracks[0].values.keys())
        for track in self.tracks[1:]:
            if set(track.values.keys()) != value_keys:
                raise ValueError(f"All tracks must have the same set of value keys. Track '{track.track_id}' has different keys.")
            
        #Check that all there is a header for each track if set_header_per_track is used
        n_tracks = len(self.tracks)
        for header_name, values in self.header_per_track.items():
            if len(values) != n_tracks:
                raise ValueError(f"Header '{header_name}' has {len(values)} values but there are {n_tracks} tracks.")
        
        return grid.Grid(
            tracks=self.tracks,
            header_per_track=self.header_per_track,
            header_global=self.header_global,
            solar_models=self.solar_models
        )