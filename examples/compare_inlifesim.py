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
from lifesimft.spectral import wavelength_bins

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


def inlifesim_exozodi_map(
    bl: np.ndarray, wl_bins: np.ndarray, wl_widths: np.ndarray
) -> np.ndarray:
    """Exozodi Fourier transform from InLIFEsim, shape (n_bins, n_collectors, n_collectors)."""
    hfov = wl_bins / (2 * ref.APERTURE_DIAMETER_M)
    hfov_mas = hfov * 3600000.0 * 180.0 / np.pi
    rad_pix = 2 * hfov / IMAGE_SIZE
    au_pix = 2 * hfov_mas / IMAGE_SIZE / 1e3 * DISTANCE_PC
    x_map = np.tile(np.arange(IMAGE_SIZE), (IMAGE_SIZE, 1))
    radius_map = np.hypot(x_map - (IMAGE_SIZE - 1) / 2, x_map.T - (IMAGE_SIZE - 1) / 2)
    r_au = radius_map[np.newaxis] * au_pix[:, np.newaxis, np.newaxis]
    return create_exozodi(
        wl_bins,
        wl_widths,
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


def inlifesim_rates(
    wl_bins: np.ndarray, wl_widths: np.ndarray
) -> dict[str, np.ndarray]:
    """InLIFEsim photon rates per output in ph s^-1 and planet flux density, per bin."""
    bl = inlifesim_baselines()
    phi = ref.PHASE_LEFT_RAD
    delta = np.cos(phi[:, np.newaxis] - phi[np.newaxis, :])
    weights = AMPLITUDES[:, np.newaxis] * AMPLITUDES[np.newaxis, :] * delta
    b_star = create_star(
        wl_bins,
        wl_widths,
        ref.STAR_TEMPERATURE_K,
        1.0,
        DISTANCE_PC,
        bl,
        ref.COLLECTOR_POSITIONS_M,
        4,
    )[1]
    flux_lz = create_localzodi(wl_bins, wl_widths, lat=ref.ECLIPTIC_LATITUDE_RAD)
    omega = np.pi * (wl_bins / (2 * ref.APERTURE_DIAMETER_M)) ** 2
    flux_planet = create_planet(
        wl_bins, wl_widths, ref.PLANET_TEMPERATURE_K, 1.0, DISTANCE_PC
    )
    b_ez = inlifesim_exozodi_map(bl, wl_bins, wl_widths)
    return {
        "star leakage": np.sum(weights * b_star, axis=(1, 2)),
        "local zodi": flux_lz * omega * np.sum(AMPLITUDES**2),
        "exozodi": np.sum(weights * b_ez, axis=(1, 2)),
        "planet flux density": flux_planet / wl_widths,
    }


def lifesimft_rates(wavelength: float, bandwidth: float) -> dict[str, float]:
    """LIFEsimFT photon rates per output in ph s^-1 and planet flux density.

    :param wavelength: bin centre in m
    :param bandwidth: bin width in m
    """
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
        timeit.repeat(
            lambda: inlifesim_exozodi_map(bl, WL_BINS, WL_WIDTHS),
            number=1,
            repeat=n_repeats,
        )
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


def compare_band() -> None:
    """Ratio LIFEsimFT / InLIFEsim per quantity over all bins of the reference band."""
    centers, widths = wavelength_bins(
        ref.WAVELENGTH_MIN_M, ref.WAVELENGTH_MAX_M, ref.SPECTRAL_RESOLUTION
    )
    ours = [lifesimft_rates(wl, dwl) for wl, dwl in zip(centers, widths)]
    theirs = inlifesim_rates(centers, widths)
    print(f"{'quantity':<22}{'min ratio':>12}{'max ratio':>12}   ({centers.size} bins)")
    for key in theirs:
        ratio = np.array([row[key] for row in ours]) / theirs[key]
        print(f"{key:<22}{ratio.min():>12.4f}{ratio.max():>12.4f}")


def main() -> None:
    ours = lifesimft_rates(ref.WAVELENGTH_M, ref.BANDWIDTH_M)
    with np.errstate(divide="ignore", invalid="ignore"):
        theirs = {
            k: float(v[0]) for k, v in inlifesim_rates(WL_BINS, WL_WIDTHS).items()
        }
    print(f"{'quantity':<22}{'LIFEsimFT':>14}{'InLIFEsim':>14}{'ratio':>10}")
    for key in ours:
        ratio = ours[key] / theirs[key]
        print(f"{key:<22}{ours[key]:>14.4g}{theirs[key]:>14.4g}{ratio:>10.4f}")
    print()
    with np.errstate(divide="ignore", invalid="ignore"):
        compare_band()
        print()
        time_exozodi()


if __name__ == "__main__":
    main()
