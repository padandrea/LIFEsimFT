"""Photon-noise contributions per wavelength bin for the reference case (cf. Fig. 8, top)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions
from lifesimft.signal import (
    exozodi_photon_rate,
    local_zodi_photon_rate,
    star_photon_rate,
)
from lifesimft.sources import blackbody_flux_density, local_zodi_radiance
from lifesimft.spectral import wavelength_bins

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


def main() -> None:
    centers, widths = wavelength_bins(
        ref.WAVELENGTH_MIN_M, ref.WAVELENGTH_MAX_M, ref.SPECTRAL_RESOLUTION
    )
    b = baselines(rotate_positions(ref.COLLECTOR_POSITIONS_M, np.zeros(1)))
    rates = [background_rates(wl, dwl, b) for wl, dwl in zip(centers, widths)]
    counts = {
        key: np.sqrt(np.array([r[key] for r in rates]) * ref.TOTAL_TIME_S)
        for key in rates[0]
    }
    counts["exozodi (InLIFEsim norm.)"] = counts["exozodi"] / np.sqrt(
        KENNEDY_OVER_INLIFESIM
    )

    styles = {
        "star": "--",
        "local zodi": "-.",
        "exozodi": ":",
        "exozodi (InLIFEsim norm.)": (0, (1, 3)),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    wavelength_um = centers * 1e6
    for key, style in styles.items():
        ax.step(
            wavelength_um, counts[key], where="mid", ls=style, color="gray", label=key
        )
    ax.set_yscale("log")
    ax.set_xlim(4, 18.5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("noise count per output, √(n t) (ph)")
    ax.legend()
    fig.tight_layout()

    for wl, row in zip(wavelength_um, zip(*counts.values())):
        if abs(wl - 10) < 0.2:
            print(
                f"{wl:.2f} um:", ", ".join(f"{k} {v:.3g}" for k, v in zip(counts, row))
            )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "noise_spectrum.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
