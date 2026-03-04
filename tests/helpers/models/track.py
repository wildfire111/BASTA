from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass(frozen=True)
class Track:
    #A data class to represent a single stellar track
    track_id: str
    values: Dict[str, np.ndarray]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Track):
            return NotImplemented
        
        if self.track_id != other.track_id:
            return False
        
        if set(self.values.keys()) != set(other.values.keys()):
            return False
        
        for key in self.values.keys():
            if not np.array_equal(self.values[key], other.values[key]):
                return False
        
        return True
    
    def values_equal(self, other) -> bool:
        if not isinstance(other, Track):
            return NotImplemented
        
        if set(self.values.keys()) != set(other.values.keys()):
            return False
        
        for key in self.values.keys():
            if not np.array_equal(self.values[key], other.values[key]):
                return False
        
        return True
