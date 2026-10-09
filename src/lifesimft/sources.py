"""Photon fluxes and visibilities of astrophysical sources, Dannert et al. (2025), App. B.2."""

import numpy as np
from scipy.constants import c, h, k
from scipy.special import j1


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
