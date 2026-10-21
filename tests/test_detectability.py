"""Tests for the generalized planet detectability calculation."""

from dataclasses import replace

import pytest

from lifesimft import reference as ref
from lifesimft.detectability import (
    Planet,
    Star,
    calculate_detection_time,
    calculate_snr,
)


@pytest.fixture
def reference_system():
    """Sun-Earth reference system from Dannert et al. (2025)."""
    star = Star(
        temperature_k=ref.STAR_TEMPERATURE_K,
        radius_m=ref.STAR_RADIUS_M,
        distance_m=ref.DISTANCE_M,
        luminosity_lsun=ref.STAR_LUMINOSITY_LSUN,
        ecliptic_latitude_rad=ref.ECLIPTIC_LATITUDE_RAD,
        zodi_level=ref.ZODI_LEVEL,
    )

    planet = Planet(
        temperature_k=ref.PLANET_TEMPERATURE_K,
        radius_m=ref.PLANET_RADIUS_M,
        position_m=(ref.PLANET_SEPARATION_M, 0.0),
    )

    return star, planet


def test_reference_snr(reference_system):
    """Reproduce the validated 16-day reference calculation."""
    star, planet = reference_system

    snr = calculate_snr(star, planet, ref.TOTAL_TIME_S)

    assert snr == pytest.approx(42.18, abs=0.05)


def test_snr_scales_with_sqrt_time(reference_system):
    """Photon-noise-limited SNR must scale as sqrt(t)."""
    star, planet = reference_system

    snr_1h = calculate_snr(star, planet, 3600.0)
    snr_4h = calculate_snr(star, planet, 4 * 3600.0)

    assert snr_4h == pytest.approx(2 * snr_1h)


def test_larger_planet_has_higher_snr(reference_system):
    """Increasing planet radius should increase the signal."""
    star, planet = reference_system

    larger_planet = Planet(
        temperature_k=planet.temperature_k,
        radius_m=2 * planet.radius_m,
        position_m=planet.position_m,
    )

    snr_earth = calculate_snr(star, planet)
    snr_large = calculate_snr(star, larger_planet)

    assert snr_large > snr_earth


def test_colder_planet_has_lower_snr(reference_system):
    """A colder blackbody planet should be harder to detect."""
    star, planet = reference_system

    colder_planet = Planet(
        temperature_k=200.0,
        radius_m=planet.radius_m,
        position_m=planet.position_m,
    )

    snr_earth = calculate_snr(star, planet)
    snr_cold = calculate_snr(star, colder_planet)

    assert snr_cold < snr_earth


@pytest.mark.parametrize("integration_time", [0.0, -3600.0])
def test_invalid_integration_time(reference_system, integration_time):
    """Zero or negative observation times must be rejected."""
    star, planet = reference_system

    with pytest.raises(ValueError):
        calculate_snr(star, planet, integration_time)


def test_detection_time_reaches_threshold(reference_system):
    """Calculated integration time must reach the requested SNR."""
    star, planet = reference_system

    t_det = calculate_detection_time(star, planet, snr_threshold=5.0)
    snr = calculate_snr(star, planet, integration_time=t_det)

    assert snr == pytest.approx(5.0)


def test_higher_threshold_requires_more_time(reference_system):
    """Integration time scales with the square of the SNR threshold."""
    star, planet = reference_system

    t5 = calculate_detection_time(star, planet, snr_threshold=5.0)
    t10 = calculate_detection_time(star, planet, snr_threshold=10.0)

    assert t10 == pytest.approx(4 * t5)


def test_invalid_snr_threshold(reference_system):
    """Detection threshold must be positive."""
    star, planet = reference_system

    with pytest.raises(ValueError):
        calculate_detection_time(star, planet, snr_threshold=0.0)


def test_greater_distance_requires_more_time(reference_system):
    """A system at 20 pc should require more time than at 10 pc."""
    star, planet = reference_system

    farther_star = replace(star, distance_m=2 * star.distance_m)

    t_near = calculate_detection_time(star, planet)
    t_far = calculate_detection_time(farther_star, planet)

    assert t_far > t_near


def test_larger_planet_requires_less_time(reference_system):
    """A larger planet should require less integration time."""
    star, planet = reference_system

    larger_planet = replace(planet, radius_m=2 * planet.radius_m)

    t_earth = calculate_detection_time(star, planet)
    t_large = calculate_detection_time(star, larger_planet)

    assert t_large < t_earth


def test_colder_planet_requires_more_time(reference_system):
    """A colder planet should require more integration time."""
    star, planet = reference_system

    colder_planet = replace(planet, temperature_k=200.0)

    t_earth = calculate_detection_time(star, planet)
    t_cold = calculate_detection_time(star, colder_planet)

    assert t_cold > t_earth
