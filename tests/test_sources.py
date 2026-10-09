import numpy as np
import pytest
from scipy.constants import au, c, h, k, sigma
from scipy.integrate import quad

from lifesimft import reference as ref
from lifesimft.sources import (
    blackbody_flux_density,
    exozodi_radiance,
    exozodi_visibility,
    local_zodi_radiance,
    planck_photon_radiance,
    point_source_visibility,
    uniform_disk_visibility,
)

HFOV_RAD = ref.WAVELENGTH_M / (2 * ref.APERTURE_DIAMETER_M)
EXOZODI_ARGS = (ref.WAVELENGTH_M, 1.0, ref.DISTANCE_M, 1.0, HFOV_RAD)


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


def test_point_source_visibility_is_a_pure_phase():
    baselines = np.array([[14.5, 0.0], [0.0, 87.0], [-7.3, 40.1]])
    theta = np.array([4.8e-7, -1.2e-7])
    v = point_source_visibility(baselines, ref.WAVELENGTH_M, theta)
    np.testing.assert_allclose(np.abs(v), 1.0)


def test_on_axis_point_source_has_unit_visibility():
    baselines = np.array([[14.5, 0.0], [0.0, 87.0]])
    v = point_source_visibility(baselines, ref.WAVELENGTH_M, np.zeros(2))
    np.testing.assert_allclose(v, 1.0)


def test_exozodi_visibility_at_zero_baseline_is_total_flux():
    theta_min = (278.3 / 1500.0) ** 2 * au / ref.DISTANCE_M
    total, _ = quad(
        lambda t: exozodi_radiance(t, *EXOZODI_ARGS[:-1]) * 2 * np.pi * t,
        theta_min,
        HFOV_RAD,
    )
    assert exozodi_visibility(0.0, *EXOZODI_ARGS) == pytest.approx(total, rel=1e-4)


def test_exozodi_is_partially_resolved_on_nulling_baseline():
    v = exozodi_visibility(np.array([0.0, 14.5]), *EXOZODI_ARGS)
    assert 0 < v[1] < v[0]


def test_exozodi_visibility_scales_linearly_with_zodi_level():
    lengths = np.array([0.0, 14.5, 87.0])
    one = exozodi_visibility(lengths, *EXOZODI_ARGS)
    three = exozodi_visibility(lengths, ref.WAVELENGTH_M, 3.0, *EXOZODI_ARGS[2:])
    np.testing.assert_allclose(three, 3 * one)


def test_exozodi_visibility_matches_inlifesim_with_kennedy_normalization():
    inlifesim = np.array([4.8632491e7, 2.1179369e7])
    kennedy_over_inlifesim = 0.034422617777777775**-0.34 * 7.12e-8 / 7.11889e-8
    v = exozodi_visibility(np.array([0.0, 14.5]), *EXOZODI_ARGS)
    np.testing.assert_allclose(v, kennedy_over_inlifesim * inlifesim, rtol=0.03)
