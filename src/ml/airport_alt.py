"""Approach / decision-height altitude conventions (OpenSky baro is metres MSL)."""

from __future__ import annotations

FT_TO_M = 0.3048

# Still "approaching" when at or below this height above airport elevation.
APPROACH_AGL_FT = 2000.0
APPROACH_AGL_M = APPROACH_AGL_FT * FT_TO_M

# Default CAT I decision height above the runway (AGL). Landing = reaching
# decision_height_m MSL = elev_m + this AGL (or an explicit CSV override).
DEFAULT_DECISION_HEIGHT_AGL_FT = 200.0
DEFAULT_DECISION_HEIGHT_AGL_M = DEFAULT_DECISION_HEIGHT_AGL_FT * FT_TO_M


def decision_height_msl_m(elev_m: float, dh_agl_ft: float = DEFAULT_DECISION_HEIGHT_AGL_FT) -> float:
    """Absolute barometric altitude (metres MSL) at which the flight is landed."""
    return float(elev_m) + float(dh_agl_ft) * FT_TO_M
