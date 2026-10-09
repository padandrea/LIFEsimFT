"""Photon rates in an interferometer output, Dannert et al. (2025), Eq. B12-B14 and B19.

Each source's Fourier transform is evaluated at the baselines x_jk / lambda and
summed over collector pairs (van Cittert-Zernike theorem, Eq. B19).
"""

import numpy as np

from lifesimft.sources import (
    exozodi_visibility,
    point_source_visibility,
    uniform_disk_visibility,
)


def pair_sum(
    amplitudes: np.ndarray, phases: np.ndarray, visibility: np.ndarray
) -> np.ndarray:
    """Interferometric pair sum of Eq. B19 for a source with known visibility.

    sum_jk A_j A_k Re[exp(i (phi_j - phi_k)) V(x_jk)]

    :param amplitudes: collector amplitude responses A_j in m, shape (n_collectors,)
    :param phases: beam-combiner phase of each collector in rad, shape (n_collectors,)
    :param visibility: source Fourier transform at the baselines, real or complex,
        shape (n_t, n_collectors, n_collectors)
    :return: pair sum in m^2, shape (n_t,)
    """
    delta_phase = phases[:, np.newaxis] - phases[np.newaxis, :]
    amplitude_product = amplitudes[:, np.newaxis] * amplitudes[np.newaxis, :]
    return np.sum(
        amplitude_product * np.real(np.exp(1j * delta_phase) * visibility),
        axis=(1, 2),
    )


def planet_photon_rate(
    flux_density: float,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    baselines: np.ndarray,
    planet_position: np.ndarray,
    wavelength: float,
    bandwidth: float,
) -> np.ndarray:
    """Photon rate from a point-source planet in one interferometer output.

    n(t) = bandwidth * flux_density
           * sum_jk A_j A_k cos(phi_j - phi_k + 2 pi / wavelength * x_jk(t) . theta)

    :param flux_density: planet spectral photon flux density in ph s^-1 m^-2 m^-1
    :param amplitudes: collector amplitude responses A_j in m, shape (n_collectors,)
    :param phases: beam-combiner phase of each collector in rad, shape (n_collectors,)
    :param baselines: x_jk in m, shape (n_t, n_collectors, n_collectors, 2)
    :param planet_position: planet offset from the star on sky in rad, shape (2,)
    :param wavelength: wavelength in m
    :param bandwidth: width of the wavelength bin in m
    :return: photon rate in ph s^-1, shape (n_t,)
    """
    visibility = point_source_visibility(baselines, wavelength, planet_position)
    return bandwidth * flux_density * pair_sum(amplitudes, phases, visibility)


def star_photon_rate(
    flux_density: float,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    baselines: np.ndarray,
    angular_radius: float,
    wavelength: float,
    bandwidth: float,
) -> np.ndarray:
    """Photon rate from a uniform-disk star centred on the optical axis (Eq. B25).

    n(t) = bandwidth * flux_density
           * sum_jk A_j A_k cos(phi_j - phi_k) V(|x_jk(t)|)

    :param flux_density: stellar spectral photon flux density in ph s^-1 m^-2 m^-1
    :param amplitudes: collector amplitude responses A_j in m, shape (n_collectors,)
    :param phases: beam-combiner phase of each collector in rad, shape (n_collectors,)
    :param baselines: x_jk in m, shape (n_t, n_collectors, n_collectors, 2)
    :param angular_radius: angular radius of the stellar disk in rad
    :param wavelength: wavelength in m
    :param bandwidth: width of the wavelength bin in m
    :return: photon rate in ph s^-1, shape (n_t,)
    """
    visibility = uniform_disk_visibility(baselines, wavelength, angular_radius)
    return bandwidth * flux_density * pair_sum(amplitudes, phases, visibility)


def field_of_view_solid_angle(wavelength: float, aperture_diameter: float) -> float:
    """Solid angle of the single-mode field of view of one collector.

    Omega = pi * (wavelength / (2 * aperture_diameter))^2

    :param wavelength: wavelength in m
    :param aperture_diameter: collector diameter in m
    :return: solid angle in sr
    """
    return np.pi * (wavelength / (2 * aperture_diameter)) ** 2


def local_zodi_photon_rate(
    radiance: float,
    amplitudes: np.ndarray,
    wavelength: float,
    aperture_diameter: float,
    bandwidth: float,
) -> float:
    """Photon rate from the local zodiacal light in one interferometer output.

    n = bandwidth * radiance * Omega * sum_j A_j^2

    :param radiance: local-zodi spectral photon radiance in ph s^-1 m^-2 m^-1 sr^-1
    :param amplitudes: collector amplitude responses A_j in m, shape (n_collectors,)
    :param wavelength: wavelength in m
    :param aperture_diameter: collector diameter in m
    :param bandwidth: width of the wavelength bin in m
    :return: photon rate in ph s^-1, constant in time
    """
    return (
        bandwidth
        * radiance
        * field_of_view_solid_angle(wavelength, aperture_diameter)
        * np.sum(amplitudes**2)
    )


def exozodi_photon_rate(
    zodi_level: float,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    baselines: np.ndarray,
    wavelength: float,
    bandwidth: float,
    distance: float,
    luminosity: float,
    aperture_diameter: float,
) -> np.ndarray:
    """Photon rate from a face-on exozodi disk in one interferometer output.

    n = bandwidth * sum_jk A_j A_k cos(phi_j - phi_k) I~_ez(|x_jk|)

    The disk is radially symmetric, so I~_ez depends only on the baseline lengths,
    which do not change under rotation: the rate is constant in time and is computed
    from the first time step only.

    :param zodi_level: number of zodis z, dimensionless
    :param amplitudes: collector amplitude responses A_j in m, shape (n_collectors,)
    :param phases: beam-combiner phase of each collector in rad, shape (n_collectors,)
    :param baselines: x_jk in m, shape (n_t, n_collectors, n_collectors, 2)
    :param wavelength: wavelength in m
    :param bandwidth: width of the wavelength bin in m
    :param distance: distance to the star in m
    :param luminosity: stellar luminosity in solar luminosities
    :param aperture_diameter: collector diameter in m, sets the field of view
    :return: photon rate in ph s^-1, shape (n_t,), constant in time
    """
    lengths = np.linalg.norm(baselines[0], axis=-1)
    max_angle = wavelength / (2 * aperture_diameter)
    visibility = exozodi_visibility(
        baseline_lengths=lengths,
        wavelength=wavelength,
        zodi_level=zodi_level,
        distance=distance,
        luminosity=luminosity,
        max_angle=max_angle,
    )
    rate = bandwidth * pair_sum(amplitudes, phases, visibility[np.newaxis])
    return np.full(baselines.shape[0], rate)
