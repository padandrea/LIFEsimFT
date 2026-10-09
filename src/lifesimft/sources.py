"""Photon fluxes and visibilities of astrophysical sources, Dannert et al. (2025), App. B.2."""

import numpy as np
from scipy.constants import au, c, h, k
from scipy.special import j0, j1


def planck_photon_radiance(wavelength: float, temperature: float) -> float:
    """Blackbody spectral photon radiance.

    :param wavelength: in m
    :param temperature: in K
    :return: photon radiance in ph s^-1 m^-2 m^-1 sr^-1
    """

    z = h * c / (wavelength * k * temperature)
    return 2 * c / wavelength**4 / np.expm1(z)


def blackbody_flux_density(
    wavelength: float, temperature: float, radius: float, distance: float
) -> float:
    """Spectral photon flux density at the observer from a blackbody sphere.

    :param wavelength: in m
    :param temperature: in K
    :param radius: radius of the sphere in m
    :param distance: distance to the observer in m
    :return: photon flux density in ph s^-1 m^-2 m^-1
    """
    return (
        planck_photon_radiance(wavelength, temperature)
        * np.pi
        * radius**2
        / distance**2
    )


def point_source_visibility(
    baselines: np.ndarray, wavelength: float, position: np.ndarray
) -> np.ndarray:
    """Fourier transform of a unit point source at angular position theta.

    V(x_jk) = exp(i 2 pi / lambda * x_jk . theta)

    A pure phase with |V| = 1: an unresolved source has the same amplitude at every
    baseline, and its offset from the optical axis only shifts the phase. The sign
    follows Eq. B12 with x_jk = x_j - x_k.

    :param baselines: baseline vectors x_jk in m, shape (..., 2)
    :param wavelength: in m
    :param position: angular offset theta from the optical axis in rad, shape (2,)
    :return: complex visibility, dimensionless, shape (...)
    """
    return np.exp(1j * 2 * np.pi / wavelength * (baselines @ position))


def uniform_disk_visibility(
    baselines: np.ndarray, wavelength: float, angular_radius: float
) -> np.ndarray:
    """Normalized Fourier transform of a uniform disk, 2 J1(x) / x (Eq. B24).

    :param baselines: baseline vectors in m, shape (..., 2)
    :param wavelength: in m
    :param angular_radius: angular radius of the disk in rad
    :return: visibility, dimensionless, shape (...); 1 for an unresolved source
    """
    x_jk = np.linalg.norm(baselines, axis=-1)
    x = 2 * np.pi * x_jk * angular_radius / wavelength
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 1.0, 2 * j1(safe) / safe)


def local_zodi_radiance(
    wavelength: float,
    ecliptic_latitude: float,
    ecliptic_longitude: float = 0.75 * np.pi,
) -> float:
    """Spectral photon radiance of the local zodiacal light (Dannert et al. 2022).

    :param wavelength: in m
    :param ecliptic_latitude: ecliptic latitude b of the target in rad
    :param ecliptic_longitude: ecliptic longitude l relative to the Sun in rad
    :return: photon radiance in ph s^-1 m^-2 m^-1 sr^-1
    """
    tau = 4e-8
    temperature_eff = 265.0
    temperature_sun = 5777.0
    albedo = 0.22
    sun_radius_over_1p5_au = 0.00465047 / 1.5

    spectrum = planck_photon_radiance(
        wavelength, temperature_eff
    ) + albedo * sun_radius_over_1p5_au**2 * planck_photon_radiance(
        wavelength, temperature_sun
    )
    cos_b, sin_b = np.cos(ecliptic_latitude), np.sin(ecliptic_latitude)
    elongation = np.arccos(np.cos(ecliptic_longitude) * cos_b)
    geometry = np.sqrt(
        np.pi
        / elongation
        / (sin_b**2 + (0.6 * (wavelength / 11e-6) ** -0.4 * cos_b) ** 2)
    )
    return tau * spectrum * geometry


def exozodi_radiance(
    angular_separation: np.ndarray,
    wavelength: float,
    zodi_level: float,
    distance: float,
    luminosity: float,
) -> np.ndarray:
    """Spectral photon radiance of a face-on exozodiacal disk (Kennedy et al. 2015).

    I(r) = Sigma(r) B(lambda, T(r)),
    T(r) = 278.3 K L^(1/4) (r / au)^(-1/2)                      (Eq. 2),
    Sigma(r) = z Sigma_0 (r / r_0)^(-alpha), r_0 = sqrt(L) au   (Eq. 3),
    with Sigma_0 = 7.12e-8, alpha = 0.34, and zero outside the radii where
    T = 1500 K (sublimation) and T = 88 K.

    :param angular_separation: angle from the star in rad, any shape
    :param wavelength: in m
    :param zodi_level: number of zodis z, dimensionless
    :param distance: distance to the star in m
    :param luminosity: stellar luminosity in solar luminosities
    :return: photon radiance in ph s^-1 m^-2 m^-1 sr^-1, same shape as
        angular_separation
    """
    sigma_zero = 7.12e-8
    alpha = 0.34
    r_au = np.asarray(angular_separation) * distance / au
    r_0 = np.sqrt(luminosity)
    r_in = (278.3 / 1500.0) ** 2 * r_0
    r_out = (278.3 / 88.0) ** 2 * r_0
    inside = (r_au >= r_in) & (r_au <= r_out)
    r_safe = np.where(inside, r_au, r_0)
    temperature = 278.3 * luminosity**0.25 / np.sqrt(r_safe)
    sigma = zodi_level * sigma_zero * (r_safe / r_0) ** -alpha
    return np.where(
        inside, sigma * planck_photon_radiance(wavelength, temperature), 0.0
    )


def exozodi_visibility(
    baseline_lengths: np.ndarray,
    wavelength: float,
    zodi_level: float,
    distance: float,
    luminosity: float,
    max_angle: float,
    n_grid: int = 4000,
) -> np.ndarray:
    """Fourier transform of the face-on exozodi disk at the given baseline lengths.

    I~(q) = integral I(theta) J0(2 pi q theta) 2 pi theta d theta,  q = |x_jk| / lambda

    integrated with the trapezoidal rule from the sublimation radius to max_angle.
    Not normalized: I~(0) is the total disk flux density inside max_angle.

    :param baseline_lengths: |x_jk| in m, any shape
    :param wavelength: in m
    :param zodi_level: number of zodis z, dimensionless
    :param distance: distance to the star in m
    :param luminosity: stellar luminosity in solar luminosities
    :param max_angle: outer integration limit in rad, e.g. the field of view lambda / 2D
    :param n_grid: number of angular grid points
    :return: photon flux density in ph s^-1 m^-2 m^-1, same shape as baseline_lengths
    """
    theta_min = (278.3 / 1500.0) ** 2 * np.sqrt(luminosity) * au / distance
    theta = np.linspace(theta_min, max_angle, n_grid)
    radiance = exozodi_radiance(theta, wavelength, zodi_level, distance, luminosity)
    q = np.asarray(baseline_lengths) / wavelength
    integrand = (
        radiance * j0(2 * np.pi * q[..., np.newaxis] * theta) * 2 * np.pi * theta
    )
    return np.trapezoid(integrand, theta, axis=-1)
