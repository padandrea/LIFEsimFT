"""Reference case of Dannert et al. (2025), Table 1, in SI units."""

import numpy as np

COLLECTOR_POSITIONS_M = np.array(
    [[-7.25, -43.5], [-7.25, 43.5], [7.25, -43.5], [7.25, 43.5]]
)
APERTURE_DIAMETER_M = 3.0
PHOTON_CONVERSION_EFFICIENCY = 0.035
PHASE_LEFT_RAD = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
PHASE_RIGHT_RAD = np.array([0.0, 3 * np.pi / 2, np.pi, np.pi / 2])

WAVELENGTH_M = 10e-6
BANDWIDTH_M = 0.3e-6

TOTAL_TIME_S = 16 * 86400.0
N_ROTATIONS = 15
N_SAMPLES = 2325

DISTANCE_M = 10 * 3.0857e16
STAR_RADIUS_M = 6.957e8
STAR_TEMPERATURE_K = 5778.0
PLANET_RADIUS_M = 6.3781e6
PLANET_SEPARATION_M = 1.495978707e11
PLANET_TEMPERATURE_K = 255.0
