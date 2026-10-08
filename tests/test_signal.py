import numpy as np
import pytest

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions, rotation_angles
from lifesimft.signal import planet_photon_rate

EARTH_AT_10_PC_RAD = 4.848e-7
AMPLITUDES = np.ones(4)
ATOL = 1e-10


@pytest.fixture
def positions() -> np.ndarray:
    angles = rotation_angles(ref.N_ROTATIONS, ref.N_SAMPLES)
    return rotate_positions(ref.COLLECTOR_POSITIONS_M, angles)


def rate(positions, phases, theta):
    return planet_photon_rate(
        flux_density=1.0,
        amplitudes=AMPLITUDES,
        phases=phases,
        baselines=baselines(positions),
        planet_position=np.asarray(theta),
        wavelength=ref.WAVELENGTH_M,
        bandwidth=1.0,
    )


def test_on_axis_source_is_nulled(positions):
    for phases in (ref.PHASE_LEFT_RAD, ref.PHASE_RIGHT_RAD):
        np.testing.assert_allclose(rate(positions, phases, [0, 0]), 0.0, atol=ATOL)


def test_rate_is_non_negative(positions):
    n = rate(positions, ref.PHASE_LEFT_RAD, [EARTH_AT_10_PC_RAD, 0])
    assert np.all(n >= -ATOL)


def test_matches_direct_field_sum(positions):
    theta = np.array([EARTH_AT_10_PC_RAD, 0.3 * EARTH_AT_10_PC_RAD])
    k = 2 * np.pi / ref.WAVELENGTH_M
    field = (
        AMPLITUDES * np.exp(1j * (ref.PHASE_LEFT_RAD + k * (positions @ theta)))
    ).sum(axis=1)
    np.testing.assert_allclose(
        rate(positions, ref.PHASE_LEFT_RAD, theta), np.abs(field) ** 2, atol=ATOL
    )


def test_right_output_mirrors_left(positions):
    theta = np.array([EARTH_AT_10_PC_RAD, 0.0])
    np.testing.assert_allclose(
        rate(positions, ref.PHASE_RIGHT_RAD, theta),
        rate(positions, ref.PHASE_LEFT_RAD, -theta),
        atol=ATOL,
    )


def test_differential_is_odd(positions):
    theta = np.array([EARTH_AT_10_PC_RAD, 0.0])

    def differential(t):
        return rate(positions, ref.PHASE_LEFT_RAD, t) - rate(
            positions, ref.PHASE_RIGHT_RAD, t
        )

    np.testing.assert_allclose(differential(theta), -differential(-theta), atol=ATOL)
