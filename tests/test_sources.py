import numpy as np
import pytest
from scipy.constants import au, c, h, k, sigma

from lifesimft import reference as ref
from lifesimft.sources import (
    blackbody_flux_density,
    exozodi_radiance,
    local_zodi_radiance,
    planck_photon_radiance,
    uniform_disk_visibility,
)


def test_planck_rayleigh_jeans_limit():
    wavelength, temperature = 1e-2, 5778.0
    expected = 2 * k * temperature / (h * wavelength**3)
    assert planck_photon_radiance(wavelength, temperature) == pytest.approx(
        expected, rel=1e-3
    )


def test_sun_gives_solar_constant():
    wavelengths = np.logspace(-8, -3, 20000)
    photon_flux = blackbody_flux_density(
        wavelengths, ref.STAR_TEMPERATURE_K, ref.STAR_RADIUS_M, au
    )
    energy_flux = np.trapezoid(photon_flux * h * c / wavelengths, wavelengths)
    expected = sigma * ref.STAR_TEMPERATURE_K**4 * (ref.STAR_RADIUS_M / au) ** 2
    assert energy_flux == pytest.approx(expected, rel=1e-3)
    assert energy_flux == pytest.approx(1361, rel=0.01)


def test_visibility_is_one_at_zero_baseline():
    assert uniform_disk_visibility(np.zeros((1, 2)), 10e-6, 1e-8)[0] == 1.0


def test_visibility_small_argument_expansion():
    baseline = np.array([[14.5, 0.0]])
    theta = ref.STAR_RADIUS_M / ref.DISTANCE_M
    x = 2 * np.pi * theta * 14.5 / ref.WAVELENGTH_M
    assert uniform_disk_visibility(baseline, ref.WAVELENGTH_M, theta)[
        0
    ] == pytest.approx(1 - x**2 / 8, rel=1e-8)


def test_visibility_first_null():
    first_zero_of_j1 = 3.8317059702
    wavelength, theta = 10e-6, 1e-8
    length = first_zero_of_j1 * wavelength / (2 * np.pi * theta)
    v = uniform_disk_visibility(np.array([[length, 0.0]]), wavelength, theta)
    assert v[0] == pytest.approx(0.0, abs=1e-9)


def test_local_zodi_at_ecliptic_pole():
    wavelength = ref.WAVELENGTH_M
    spectrum = planck_photon_radiance(wavelength, 265.0) + 0.22 * (
        0.00465047 / 1.5
    ) ** 2 * planck_photon_radiance(wavelength, 5777.0)
    expected = 4e-8 * spectrum * np.sqrt(2)
    assert local_zodi_radiance(wavelength, np.pi / 2) == pytest.approx(expected)


def test_exozodi_at_reference_radius():
    theta = au / ref.DISTANCE_M
    expected = 7.12e-8 * planck_photon_radiance(ref.WAVELENGTH_M, 278.3)
    radiance = exozodi_radiance(theta, ref.WAVELENGTH_M, 1.0, ref.DISTANCE_M, 1.0)
    assert radiance == pytest.approx(expected)


def test_exozodi_is_zero_outside_the_disk():
    theta = np.array([0.01, 20.0]) * au / ref.DISTANCE_M
    radiance = exozodi_radiance(theta, ref.WAVELENGTH_M, 1.0, ref.DISTANCE_M, 1.0)
    np.testing.assert_array_equal(radiance, 0.0)


def test_exozodi_scales_linearly_with_zodi_level():
    theta = np.linspace(0.1, 5.0, 50) * au / ref.DISTANCE_M
    one = exozodi_radiance(theta, ref.WAVELENGTH_M, 1.0, ref.DISTANCE_M, 1.0)
    three = exozodi_radiance(theta, ref.WAVELENGTH_M, 3.0, ref.DISTANCE_M, 1.0)
    np.testing.assert_allclose(three, 3 * one)
