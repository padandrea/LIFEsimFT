import numpy as np
import pytest

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions, rotation_angles
from lifesimft.signal import (
    local_zodi_photon_rate,
    planet_photon_rate,
    star_photon_rate,
)
from lifesimft.sources import blackbody_flux_density, local_zodi_radiance

EARTH_AT_10_PC_RAD = 4.848e-7
AMPLITUDES = np.ones(4)
AMPLITUDE_REF = np.sqrt(
    np.pi * (ref.APERTURE_DIAMETER_M / 2) ** 2 * ref.PHOTON_CONVERSION_EFFICIENCY / 4
)
STAR_ANGULAR_RADIUS = ref.STAR_RADIUS_M / ref.DISTANCE_M
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


def star_rate(positions, phases, angular_radius):
    return star_photon_rate(
        flux_density=blackbody_flux_density(
            ref.WAVELENGTH_M,
            ref.STAR_TEMPERATURE_K,
            ref.STAR_RADIUS_M,
            ref.DISTANCE_M,
        ),
        amplitudes=np.full(4, AMPLITUDE_REF),
        phases=phases,
        baselines=baselines(positions),
        angular_radius=angular_radius,
        wavelength=ref.WAVELENGTH_M,
        bandwidth=ref.BANDWIDTH_M,
    )


def test_point_star_is_nulled(positions):
    n = star_rate(positions, ref.PHASE_LEFT_RAD, 0.0)
    np.testing.assert_allclose(n, 0.0, atol=1e-6)


def test_star_leakage_constant_and_equal_in_both_outputs(positions):
    n_left = star_rate(positions, ref.PHASE_LEFT_RAD, STAR_ANGULAR_RADIUS)
    n_right = star_rate(positions, ref.PHASE_RIGHT_RAD, STAR_ANGULAR_RADIUS)
    np.testing.assert_allclose(n_left, n_left[0], rtol=1e-9)
    np.testing.assert_allclose(n_left, n_right, rtol=1e-9)


def test_star_leakage_matches_small_disk_formula(positions):
    flux = blackbody_flux_density(
        ref.WAVELENGTH_M, ref.STAR_TEMPERATURE_K, ref.STAR_RADIUS_M, ref.DISTANCE_M
    )
    nulling_baseline = 14.5
    expected = (
        2
        * AMPLITUDE_REF**2
        * ref.BANDWIDTH_M
        * flux
        * (np.pi * STAR_ANGULAR_RADIUS * nulling_baseline / ref.WAVELENGTH_M) ** 2
    )
    n = star_rate(positions, ref.PHASE_LEFT_RAD, STAR_ANGULAR_RADIUS)
    assert n[0] == pytest.approx(expected, rel=1e-3)


def test_local_zodi_rate_matches_inlifesim():
    rate = local_zodi_photon_rate(
        radiance=local_zodi_radiance(ref.WAVELENGTH_M, ref.ECLIPTIC_LATITUDE_RAD),
        amplitudes=np.full(4, AMPLITUDE_REF),
        wavelength=ref.WAVELENGTH_M,
        aperture_diameter=ref.APERTURE_DIAMETER_M,
        bandwidth=ref.BANDWIDTH_M,
    )
    assert rate == pytest.approx(10.061779016, rel=1e-4)
