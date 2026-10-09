"""Photon rates per output of LIFEsimFT and InLIFEsim for the reference case at 10 um.

Requires InLIFEsim (https://github.com/fdannert/InLIFEsim, commit 9505676),
installed with: pip install -e ../InLIFEsim
"""

import timeit

import numpy as np
from inlifesim.sources import (
    create_exozodi,
    create_localzodi,
    create_planet,
    create_star,
)

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions
from lifesimft.signal import (
    exozodi_photon_rate,
    local_zodi_photon_rate,
    star_photon_rate,
)
from lifesimft.sources import (
    blackbody_flux_density,
    exozodi_visibility,
    local_zodi_radiance,
)

WL_BINS = np.array([ref.WAVELENGTH_M])
WL_WIDTHS = np.array([ref.BANDWIDTH_M])
DISTANCE_PC = 10.0
IMAGE_SIZE = 512
AMPLITUDES = np.full(
    4,
    np.sqrt(
        np.pi
        * (ref.APERTURE_DIAMETER_M / 2) ** 2
        * ref.PHOTON_CONVERSION_EFFICIENCY
        / 4
    ),
)


def inlifesim_baselines() -> np.ndarray:
    """Baseline matrix in InLIFEsim layout, shape (2, n_collectors, n_collectors)."""
    x = ref.COLLECTOR_POSITIONS_M[:, 0]
    y = ref.COLLECTOR_POSITIONS_M[:, 1]
    return np.array((np.subtract.outer(x, x).T, np.subtract.outer(y, y).T))


def inlifesim_exozodi_map(bl: np.ndarray) -> np.ndarray:
    """Exozodi Fourier transform from InLIFEsim, shape (1, n_collectors, n_collectors)."""
    hfov = WL_BINS / (2 * ref.APERTURE_DIAMETER_M)
    hfov_mas = hfov * 3600000.0 * 180.0 / np.pi
    rad_pix = 2 * hfov / IMAGE_SIZE
    au_pix = 2 * hfov_mas / IMAGE_SIZE / 1e3 * DISTANCE_PC
    x_map = np.tile(np.arange(IMAGE_SIZE), (IMAGE_SIZE, 1))
    radius_map = np.hypot(x_map - (IMAGE_SIZE - 1) / 2, x_map.T - (IMAGE_SIZE - 1) / 2)
    r_au = radius_map[np.newaxis] * au_pix[:, np.newaxis, np.newaxis]
    return create_exozodi(
        WL_BINS,
        WL_WIDTHS,
        1.0,
        ref.ZODI_LEVEL,
        r_au,
        IMAGE_SIZE,
        au_pix,
        rad_pix,
        radius_map,
        bl,
        hfov,
    )


def inlifesim_rates() -> dict[str, float]:
    """InLIFEsim photon rates per output in ph s^-1 and planet flux density."""
    bl = inlifesim_baselines()
    phi = ref.PHASE_LEFT_RAD
    delta = np.cos(phi[:, np.newaxis] - phi[np.newaxis, :])
    weights = AMPLITUDES[:, np.newaxis] * AMPLITUDES[np.newaxis, :] * delta
    b_star = create_star(
        WL_BINS,
        WL_WIDTHS,
        ref.STAR_TEMPERATURE_K,
        1.0,
        DISTANCE_PC,
        bl,
        ref.COLLECTOR_POSITIONS_M,
        4,
    )[1]
    flux_lz = create_localzodi(WL_BINS, WL_WIDTHS, lat=ref.ECLIPTIC_LATITUDE_RAD)
    omega = np.pi * (WL_BINS / (2 * ref.APERTURE_DIAMETER_M)) ** 2
    flux_planet = create_planet(
        WL_BINS, WL_WIDTHS, ref.PLANET_TEMPERATURE_K, 1.0, DISTANCE_PC
    )
    return {
        "star leakage": float(np.sum(weights * b_star[0])),
        "local zodi": float(flux_lz[0] * omega[0] * np.sum(AMPLITUDES**2)),
        "exozodi": float(np.sum(weights * inlifesim_exozodi_map(bl)[0])),
        "planet flux density": float(flux_planet[0] / WL_WIDTHS[0]),
    }


def lifesimft_rates() -> dict[str, float]:
    """LIFEsimFT photon rates per output in ph s^-1 and planet flux density."""
    b = baselines(rotate_positions(ref.COLLECTOR_POSITIONS_M, np.zeros(1)))
    star = star_photon_rate(
        flux_density=blackbody_flux_density(
            ref.WAVELENGTH_M, ref.STAR_TEMPERATURE_K, ref.STAR_RADIUS_M, ref.DISTANCE_M
        ),
        angular_radius=ref.STAR_RADIUS_M / ref.DISTANCE_M,
        amplitudes=AMPLITUDES,
        phases=ref.PHASE_LEFT_RAD,
        baselines=b,
        wavelength=ref.WAVELENGTH_M,
        bandwidth=ref.BANDWIDTH_M,
    )[0]
    local_zodi = local_zodi_photon_rate(
        radiance=local_zodi_radiance(ref.WAVELENGTH_M, ref.ECLIPTIC_LATITUDE_RAD),
        amplitudes=AMPLITUDES,
        wavelength=ref.WAVELENGTH_M,
        aperture_diameter=ref.APERTURE_DIAMETER_M,
        bandwidth=ref.BANDWIDTH_M,
    )
    exozodi = exozodi_photon_rate(
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
    planet = blackbody_flux_density(
        ref.WAVELENGTH_M,
        ref.PLANET_TEMPERATURE_K,
        ref.PLANET_RADIUS_M,
        ref.DISTANCE_M,
    )
    return {
        "star leakage": float(star),
        "local zodi": float(local_zodi),
        "exozodi": float(exozodi),
        "planet flux density": float(planet),
    }


def time_exozodi(n_repeats: int = 5) -> None:
    """Runtime of the exozodi Fourier transform at 10 um, InLIFEsim vs LIFEsimFT.

    InLIFEsim sums a 2D pixel map; LIFEsimFT integrates the 1D Hankel transform.
    The fastest of n_repeats runs is reported.

    :param n_repeats: number of timed runs per code
    """
    bl = inlifesim_baselines()
    lengths = np.hypot(bl[0], bl[1])
    max_angle = ref.WAVELENGTH_M / (2 * ref.APERTURE_DIAMETER_M)
    t_inlifesim = min(
        timeit.repeat(lambda: inlifesim_exozodi_map(bl), number=1, repeat=n_repeats)
    )
    t_lifesimft = min(
        timeit.repeat(
            lambda: exozodi_visibility(
                lengths,
                ref.WAVELENGTH_M,
                ref.ZODI_LEVEL,
                ref.DISTANCE_M,
                ref.STAR_LUMINOSITY_LSUN,
                max_angle,
            ),
            number=1,
            repeat=n_repeats,
        )
    )
    print(f"exozodi transform, InLIFEsim ({IMAGE_SIZE} px): {t_inlifesim * 1e3:.1f} ms")
    print(f"exozodi transform, LIFEsimFT (Hankel):  {t_lifesimft * 1e3:.1f} ms")
    print(f"speed-up: {t_inlifesim / t_lifesimft:.0f}x")


def main() -> None:
    ours = lifesimft_rates()
    with np.errstate(divide="ignore", invalid="ignore"):
        theirs = inlifesim_rates()
    print(f"{'quantity':<22}{'LIFEsimFT':>14}{'InLIFEsim':>14}{'ratio':>10}")
    for key in ours:
        ratio = ours[key] / theirs[key]
        print(f"{key:<22}{ours[key]:>14.4g}{theirs[key]:>14.4g}{ratio:>10.4f}")
    print()
    with np.errstate(divide="ignore", invalid="ignore"):
        time_exozodi()


if __name__ == "__main__":
    main()
