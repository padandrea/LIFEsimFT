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
from lifesimft.statistics import test_statistic

AMPLITUDES = np.full(
    4,
    np.sqrt(
        np.pi
        * (ref.APERTURE_DIAMETER_M / 2) ** 2
        * ref.PHOTON_CONVERSION_EFFICIENCY
        / 4
    ),
)


def source_rates(phases: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Planet and stellar photon rates in one output for the reference case.

    :param phases: beam-combiner phases of the output in rad, shape (n_collectors,)
    :param b: baselines in m, shape (n_t, n_collectors, n_collectors, 2)
    :return: planet rate and star rate in ph s^-1, each shape (n_t,)
    """
    common = dict(
        amplitudes=AMPLITUDES,
        phases=phases,
        baselines=b,
        wavelength=ref.WAVELENGTH_M,
        bandwidth=ref.BANDWIDTH_M,
    )
    planet = planet_photon_rate(
        flux_density=blackbody_flux_density(
            ref.WAVELENGTH_M,
            ref.PLANET_TEMPERATURE_K,
            ref.PLANET_RADIUS_M,
            ref.DISTANCE_M,
        ),
        planet_position=np.array([ref.PLANET_SEPARATION_M / ref.DISTANCE_M, 0.0]),
        **common,
    )
    star = star_photon_rate(
        flux_density=blackbody_flux_density(
            ref.WAVELENGTH_M, ref.STAR_TEMPERATURE_K, ref.STAR_RADIUS_M, ref.DISTANCE_M
        ),
        angular_radius=ref.STAR_RADIUS_M / ref.DISTANCE_M,
        **common,
    )
    return planet, star


def local_zodi_rate() -> float:
    """Local-zodi photon rate per output for the reference case.

    :return: photon rate in ph s^-1
    """
    return local_zodi_photon_rate(
        radiance=local_zodi_radiance(ref.WAVELENGTH_M, ref.ECLIPTIC_LATITUDE_RAD),
        amplitudes=AMPLITUDES,
        wavelength=ref.WAVELENGTH_M,
        aperture_diameter=ref.APERTURE_DIAMETER_M,
        bandwidth=ref.BANDWIDTH_M,
    )


def exozodi_rate(b: np.ndarray) -> float:
    """Exozodi photon rate per output for the reference case.

    Constant in time and equal in both outputs, so one value from the left output.

    :param b: baselines in m, shape (n_t, n_collectors, n_collectors, 2)
    :return: photon rate in ph s^-1
    """
    return exozodi_photon_rate(
        zodi_level=ref.ZODI_LEVEL,
        amplitudes=AMPLITUDES,
        phases=ref.PHASE_LEFT_RAD,
        baselines=b,
        wavelength=ref.WAVELENGTH_M,
        bandwidth=ref.BANDWIDTH_M,
        distance=ref.DISTANCE_M,
        luminosity=ref.STAR_LUMINOSITY_LSUN,
        aperture_diameter=ref.APERTURE_DIAMETER_M,
    )[0]


def t_with_noise(
    planet_left: np.ndarray,
    planet_right: np.ndarray,
    star_left: np.ndarray,
    star_right: np.ndarray,
    background: float,
) -> float:
    """Test statistic T with photon noise from all sources in each output.

    :param planet_left: planet rate in the left output in ph s^-1, shape (n_t,)
    :param planet_right: planet rate in the right output in ph s^-1, shape (n_t,)
    :param star_left: stellar leakage in the left output in ph s^-1, shape (n_t,)
    :param star_right: stellar leakage in the right output in ph s^-1, shape (n_t,)
    :param background: local-zodi plus exozodi rate per output in ph s^-1
    :return: T, dimensionless
    """
    return test_statistic(
        differential_signal=planet_left - planet_right,
        mean_rate_left=(planet_left + star_left).mean() + background,
        mean_rate_right=(planet_right + star_right).mean() + background,
        total_time=ref.TOTAL_TIME_S,
    )


def main() -> None:
    angles = rotation_angles(ref.N_ROTATIONS, ref.N_SAMPLES)
    b = baselines(rotate_positions(ref.COLLECTOR_POSITIONS_M, angles))
    planet_left, star_left = source_rates(ref.PHASE_LEFT_RAD, b)
    planet_right, star_right = source_rates(ref.PHASE_RIGHT_RAD, b)

    zodi = local_zodi_rate()
    exozodi = exozodi_rate(b)
    signal = (planet_left, planet_right, star_left, star_right)
    t_star = t_with_noise(*signal, 0.0)
    t_zodi = t_with_noise(*signal, zodi)
    t_all = t_with_noise(*signal, zodi + exozodi)

    print(f"stellar leakage per output: {star_left.mean():.2f} ph/s")
    print(f"local zodi per output:      {zodi:.2f} ph/s")
    print(f"exozodi per output:         {exozodi:.2f} ph/s")
    print(f"T star only:          {t_star:.2f}")
    print(f"T + local zodi:       {t_zodi:.2f}")
    print(f"T + local + exozodi:  {t_all:.2f}")
    print(f"total ratio:          {t_star / t_all:.3f}")


if __name__ == "__main__":
    main()
