"""Shared astronomical constants and unit conversions."""

import numpy as np
from scipy.constants import au, parsec

R_SUN_M = 6.957e8
R_EARTH_M = 6.3781e6
T_SUN_K = 5778.0
AU_M = au
PARSEC_M = parsec
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
OBLIQUITY_RAD = np.radians(23.4392911)
