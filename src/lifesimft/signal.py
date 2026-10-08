"""Planet photon rate in an interferometer output, Dannert et al. (2025), Eq. B12, B13, and B14."""

import numpy as np


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
    geometric_phase = 2 * np.pi / wavelength * (baselines @ planet_position)
    delta_phase = phases[:, np.newaxis] - phases[np.newaxis, :]
    amplitude_product = amplitudes[:, np.newaxis] * amplitudes[np.newaxis, :]
    sum_over_pairs = np.sum(
        amplitude_product * np.cos(delta_phase + geometric_phase), axis=(1, 2)
    )
    return bandwidth * flux_density * sum_over_pairs
