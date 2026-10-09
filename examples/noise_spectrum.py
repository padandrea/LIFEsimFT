"""Photon-noise contributions per wavelength bin for the reference case (cf. Fig. 8, top)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

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

AMPLITUDES = np.full(
    4,
    np.sqrt(
        np.pi
        * (ref.APERTURE_DIAMETER_M / 2) ** 2
        * ref.PHOTON_CONVERSION_EFFICIENCY
        / 4
    ),
)
KENNEDY_OVER_INLIFESIM = 0.034422617777777775**-0.34 * 7.12e-8 / 7.11889e-8


def background_rates(
    wavelength: float, bandwidth: float, b: np.ndarray
) -> dict[str, float]:
    """Background photon rates per output in one wavelength bin.

    :param wavelength: bin centre in m
    :param bandwidth: bin width in m
    :param b: baselines of one time step in m, shape (1, n_collectors, n_collectors, 2)
    :return: star, local-zodi and exozodi rates in ph s^-1
    """
    star = star_photon_rate(
        flux_density=blackbody_flux_density(
            wavelength, ref.STAR_TEMPERATURE_K, ref.STAR_RADIUS_M, ref.DISTANCE_M
        ),
        amplitudes=AMPLITUDES,
        phases=ref.PHASE_LEFT_RAD,
        baselines=b,
        angular_radius=ref.STAR_RADIUS_M / ref.DISTANCE_M,
        wavelength=wavelength,
        bandwidth=bandwidth,
    )[0]
    local_zodi = local_zodi_photon_rate(
        radiance=local_zodi_radiance(wavelength, ref.ECLIPTIC_LATITUDE_RAD),
        amplitudes=AMPLITUDES,
        wavelength=wavelength,
        aperture_diameter=ref.APERTURE_DIAMETER_M,
        bandwidth=bandwidth,
    )
    exozodi = exozodi_photon_rate(
        zodi_level=ref.ZODI_LEVEL,
        amplitudes=AMPLITUDES,
        phases=ref.PHASE_LEFT_RAD,
        baselines=b,
        wavelength=wavelength,
        bandwidth=bandwidth,
        distance=ref.DISTANCE_M,
        luminosity=ref.STAR_LUMINOSITY_LSUN,
        aperture_diameter=ref.APERTURE_DIAMETER_M,
    )[0]
    return {"star": star, "local zodi": local_zodi, "exozodi": exozodi}


def planet_rates(
    wavelength: float, bandwidth: float, b_full: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Planet photon rates in the left and right output over the full observation.

    :param wavelength: bin centre in m
    :param bandwidth: bin width in m
    :param b_full: baselines of all time steps in m,
        shape (n_t, n_collectors, n_collectors, 2)
    :return: left and right rate in ph s^-1, each shape (n_t,)
    """
    common = dict(
        flux_density=blackbody_flux_density(
            wavelength, ref.PLANET_TEMPERATURE_K, ref.PLANET_RADIUS_M, ref.DISTANCE_M
        ),
        amplitudes=AMPLITUDES,
        baselines=b_full,
        planet_position=np.array([ref.PLANET_SEPARATION_M / ref.DISTANCE_M, 0.0]),
        wavelength=wavelength,
        bandwidth=bandwidth,
    )
    left = planet_photon_rate(phases=ref.PHASE_LEFT_RAD, **common)
    right = planet_photon_rate(phases=ref.PHASE_RIGHT_RAD, **common)
    return left, right


def bin_test_statistic(
    wavelength: float, bandwidth: float, b_one: np.ndarray, b_full: np.ndarray
) -> float:
    """Test statistic T in one wavelength bin, photon noise from all sources.

    :param wavelength: bin centre in m
    :param bandwidth: bin width in m
    :param b_one: baselines of one time step in m,
        shape (1, n_collectors, n_collectors, 2)
    :param b_full: baselines of all time steps in m,
        shape (n_t, n_collectors, n_collectors, 2)
    :return: T, dimensionless
    """
    background = sum(background_rates(wavelength, bandwidth, b_one).values())
    planet_left, planet_right = planet_rates(wavelength, bandwidth, b_full)
    mean_left = planet_left.mean() + background
    mean_right = planet_right.mean() + background
    return test_statistic(
        planet_left - planet_right, mean_left, mean_right, ref.TOTAL_TIME_S
    )


def main() -> None:
    centers, widths = wavelength_bins(
        ref.WAVELENGTH_MIN_M, ref.WAVELENGTH_MAX_M, ref.SPECTRAL_RESOLUTION
    )
    b_one = baselines(rotate_positions(ref.COLLECTOR_POSITIONS_M, np.zeros(1)))
    b_full = baselines(
        rotate_positions(
            ref.COLLECTOR_POSITIONS_M, rotation_angles(ref.N_ROTATIONS, ref.N_SAMPLES)
        )
    )
    rates = [background_rates(wl, dwl, b_one) for wl, dwl in zip(centers, widths)]
    counts = {
        key: np.sqrt(np.array([r[key] for r in rates]) * ref.TOTAL_TIME_S)
        for key in rates[0]
    }
    counts["exozodi (InLIFEsim norm.)"] = counts["exozodi"] / np.sqrt(
        KENNEDY_OVER_INLIFESIM
    )
    t_bins = np.array(
        [bin_test_statistic(wl, dwl, b_one, b_full) for wl, dwl in zip(centers, widths)]
    )
    t_total = np.sqrt(np.sum(t_bins**2))

    styles = {
        "star": "--",
        "local zodi": "-.",
        "exozodi": ":",
        "exozodi (InLIFEsim norm.)": (0, (1, 3)),
    }
    fig, (ax_noise, ax_t) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6), height_ratios=(2, 1)
    )
    wavelength_um = centers * 1e6
    for key, style in styles.items():
        ax_noise.step(
            wavelength_um, counts[key], where="mid", ls=style, color="gray", label=key
        )
    ax_noise.set_yscale("log")
    ax_noise.set_ylabel("noise count per output, √(n t) (ph)")
    ax_noise.legend()
    ax_t.step(wavelength_um, t_bins, where="mid", color="black")
    ax_t.set_ylabel("T per bin")
    ax_t.text(0.02, 0.85, f"total T = {t_total:.1f}", transform=ax_t.transAxes)
    ax_t.set_xlim(4, 18.5)
    ax_t.xaxis.set_major_locator(MultipleLocator(2))
    ax_t.set_xlabel("wavelength (µm)")
    fig.tight_layout()

    for wl, row in zip(wavelength_um, zip(*counts.values())):
        if abs(wl - 10) < 0.2:
            print(
                f"{wl:.2f} um:", ", ".join(f"{k} {v:.3g}" for k, v in zip(counts, row))
            )
    print(f"T at 10.15 um: {t_bins[np.argmin(abs(wavelength_um - 10.15))]:.2f}")
    print(f"total T over {centers.size} bins: {t_total:.1f}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "noise_spectrum.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
