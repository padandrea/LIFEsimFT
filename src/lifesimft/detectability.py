"""Expected planet detection SNR for arbitrary stellar systems."""

from dataclasses import dataclass

import numpy as np

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions, rotation_angles
from lifesimft.signal import (
    exozodi_photon_rate,
    local_zodi_photon_rate,
    planet_photon_rate,
    star_photon_rate,
)
from lifesimft.sources import blackbody_flux_density, local_zodi_radiance
from lifesimft.spectral import wavelength_bins
from lifesimft.statistics import test_statistic


@dataclass(frozen=True)
class Star:
    temperature_k: float
    radius_m: float
    distance_m: float
    luminosity_lsun: float
    ecliptic_latitude_rad: float
    zodi_level: float = 1.0


@dataclass(frozen=True)
class Planet:
    temperature_k: float
    radius_m: float
    position_m: tuple[float, float]


def calculate_snr(
    star: Star,
    planet: Planet,
    integration_time: float = 3600.0,
) -> float:
    """Expected total matched-filter SNR, photon noise only.

    All dimensional inputs use SI units.
    Planet position is the projected sky-plane offset in metres.
    Integration time is in seconds.
    """
    if integration_time <= 0:
        raise ValueError("integration_time must be positive")

    amplitudes = np.full(
        4,
        np.sqrt(
            np.pi
            * (ref.APERTURE_DIAMETER_M / 2) ** 2
            * ref.PHOTON_CONVERSION_EFFICIENCY
            / 4
        ),
    )

    angles = rotation_angles(ref.N_ROTATIONS, ref.N_SAMPLES)
    b_full = baselines(rotate_positions(ref.COLLECTOR_POSITIONS_M, angles))
    b_one = baselines(rotate_positions(ref.COLLECTOR_POSITIONS_M, np.zeros(1)))

    wavelengths, widths = wavelength_bins(
        ref.WAVELENGTH_MIN_M,
        ref.WAVELENGTH_MAX_M,
        ref.SPECTRAL_RESOLUTION,
    )

    position_rad = np.asarray(planet.position_m) / star.distance_m
    snr_squared = 0.0

    for wavelength, bandwidth in zip(wavelengths, widths):
        star_flux = blackbody_flux_density(
            wavelength, star.temperature_k, star.radius_m, star.distance_m
        )
        planet_flux = blackbody_flux_density(
            wavelength, planet.temperature_k, planet.radius_m, star.distance_m
        )

        common = dict(
            amplitudes=amplitudes,
            baselines=b_full,
            planet_position=position_rad,
            wavelength=wavelength,
            bandwidth=bandwidth,
            flux_density=planet_flux,
        )

        left = planet_photon_rate(phases=ref.PHASE_LEFT_RAD, **common)
        right = planet_photon_rate(phases=ref.PHASE_RIGHT_RAD, **common)

        stellar_noise = star_photon_rate(
            star_flux,
            amplitudes,
            ref.PHASE_LEFT_RAD,
            b_one,
            star.radius_m / star.distance_m,
            wavelength,
            bandwidth,
        )[0]

        local_noise = local_zodi_photon_rate(
            local_zodi_radiance(wavelength, star.ecliptic_latitude_rad),
            amplitudes,
            wavelength,
            ref.APERTURE_DIAMETER_M,
            bandwidth,
        )

        exozodi_noise = exozodi_photon_rate(
            star.zodi_level,
            amplitudes,
            ref.PHASE_LEFT_RAD,
            b_one,
            wavelength,
            bandwidth,
            star.distance_m,
            star.luminosity_lsun,
            ref.APERTURE_DIAMETER_M,
        )[0]

        background = stellar_noise + local_noise + exozodi_noise

        t_bin = test_statistic(
            left - right,
            left.mean() + background,
            right.mean() + background,
            integration_time,
        )
        snr_squared += t_bin**2

    return float(np.sqrt(snr_squared))


def calculate_detection_time(
    star: Star,
    planet: Planet,
    snr_threshold: float = 5.0,
) -> float:
    """Integration time required to reach the expected detection SNR.

    :param star: Host star parameters
    :param planet: Planet parameters
    :param snr_threshold: Required detection significance
    :return: Integration time in seconds
    """
    if not np.isfinite(snr_threshold) or snr_threshold <= 0:
        raise ValueError("snr_threshold must be positive and finite")

    reference_time = 3600.0
    snr_1h = calculate_snr(star, planet, reference_time)

    if snr_1h == 0:
        return np.inf

    return reference_time * (snr_threshold / snr_1h) ** 2
