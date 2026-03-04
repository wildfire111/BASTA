from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Star:
    Teff: Optional[float] = None
    Teff_err: Optional[float] = None

    FeH: Optional[float] = None
    FeH_err: Optional[float] = None

    logg: Optional[float] = None
    logg_err: Optional[float] = None