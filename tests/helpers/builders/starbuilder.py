from tests.helpers.models.star import Star
from tests.helpers.models.track import Track

class StarBuilder:
    """Builds star objects from test data. This is used in tests to track behaviour of the prediction code by 
    building observables that work off the track data in the test grids"""

    TEST_DEFAULT_ERR = { #This is built from the BASTA example observables file.
    "Teff_err": 50.0,
    "FeH_err": 0.026,
    "logg_err": 0.07,
    }

    TRACK_FIELDS = ("Teff", "FeH", "logg")

    def make_star_equal_to_track_index(self, track: Track, index: int = -1) -> Star:
        """
        Make a Star whose observables equal the Track values at `index`.
        Only fills fields that exist in both Track.values and Star.
        """
        observables = {}
        for key in self.TRACK_FIELDS:
            if key not in track.values:
                continue
            observables[key] = track.values[key][index]
            err_key = f"{key}_err"
            observables[err_key] = self.TEST_DEFAULT_ERR[err_key]

        return Star(**observables)
    
    def make_star_between_track_indices(self, track: Track, index1: int, index2: int) -> Star:
        """
        Make a Star whose observables are the average of the Track values at `index1` and `index2`.
        Only fills fields that exist in both Track.values and Star.
        """
        observables = {}
        for key in self.TRACK_FIELDS:
            if key not in track.values:
                continue
            observables[key] = (track.values[key][index1] + track.values[key][index2]) / 2
            err_key = f"{key}_err"
            observables[err_key] = self.TEST_DEFAULT_ERR[err_key]

        return Star(**observables)
    

    def make_star_between_two_tracks(self, track1: Track, track2: Track, index1: int, index2: int) -> Star:
        """
        Make a Star whose observables are the average of the values at `index1` and `index2` of `track1` and `track2`.
        Only fills fields that exist in both Track.values and Star.
        """
        observables = {}
        for key in self.TRACK_FIELDS:
            if key not in track1.values or key not in track2.values:
                continue
            observables[key] = (track1.values[key][index1] + track2.values[key][index2]) / 2
            err_key = f"{key}_err"
            observables[err_key] = self.TEST_DEFAULT_ERR[err_key]

        return Star(**observables)
    
    def shift_star(self, star: Star, **shifts) -> Star:
        """
        Return a new Star with one or more observable values shifted by the given amounts.

        Only observable fields (not error fields) may be shifted.
        Pass keyword arguments matching Star field names, e.g.:
            shift_star(star, Teff=20.0)   # increases Teff by 20 K
            shift_star(star, Teff=-10.0, FeH=0.01)
        """
        for field in shifts:
            if field not in self.TRACK_FIELDS:
                raise ValueError(
                    f"Cannot shift field '{field}'. "
                    f"Only observable fields {self.TRACK_FIELDS} may be shifted."
                )
        current_values = {
            field: getattr(star, field)
            for field in vars(Star())
            if getattr(star, field) is not None
        }
        for field, amount in shifts.items():
            current_values[field] = current_values[field] + amount
        return Star(**current_values)