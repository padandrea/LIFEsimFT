"""LIFEsimFT against InLIFEsim for the reference case: noise and T per wavelength bin.

Requires InLIFEsim (see compare_inlifesim.py).
Run from the repository root: python examples/summary_inlifesim.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compare_inlifesim import (
    AMPLITUDES,
    inlifesim_baselines,
    inlifesim_rates,
    lifesimft_rates,
)
from inlifesim.signal import planet_response
from matplotlib.ticker import MultipleLocator
from noise_spectrum import bin_test_statistic

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions, rotation_angles
from lifesimft.spectral import wavelength_bins
from lifesimft.statistics import test_statistic

COLORS = {"LIFEsimFT": "#2a78d6", "InLIFEsim": "#eb6834"}
SOURCES = {"star leakage": "--", "local zodi": "-.", "exozodi": ":"}
SAMPLES_PER_ROTATION = ref.N_SAMPLES // ref.N_ROTATIONS


def inlifesim_test_statistic(
    wl_bins: np.ndarray, wl_widths: np.ndarray, rates: dict[str, np.ndarray]
) -> np.ndarray:
    """T per bin from InLIFEsim planet response and InLIFEsim noise rates.

    One array rotation is enough: the rms of the periodic signal is the same over
    all rotations.

    :param wl_bins: bin centres in m, shape (n_bins,)
    :param wl_widths: bin widths in m, shape (n_bins,)
    :param rates: InLIFEsim rates per bin from inlifesim_rates
    :return: T per bin, shape (n_bins,)
    """
    phi_rot = rotation_angles(1, SAMPLES_PER_ROTATION)
    theta_p = ref.PLANET_SEPARATION_M / ref.DISTANCE_M
    theta = np.array((-theta_p * np.cos(phi_rot), theta_p * np.sin(phi_rot)))
    flux = rates["planet flux density"] * wl_widths
    common = dict(
        flux_planet=flux,
        A=AMPLITUDES,
        wl_bins=wl_bins,
        bl=inlifesim_baselines(),
        num_a=4,
        theta=theta,
        phi_rot=phi_rot,
    )
    left = planet_response(phi=ref.PHASE_LEFT_RAD, **common)
    right = planet_response(phi=ref.PHASE_RIGHT_RAD, **common)
    background = rates["star leakage"] + rates["local zodi"] + rates["exozodi"]
    return np.array(
        [
            test_statistic(
                left[i] - right[i],
                left[i].mean() + background[i],
                right[i].mean() + background[i],
                ref.TOTAL_TIME_S,
            )
            for i in range(wl_bins.size)
        ]
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
    with np.errstate(divide="ignore", invalid="ignore"):
        rows = [lifesimft_rates(wl, dwl) for wl, dwl in zip(centers, widths)]
        ours = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        theirs = inlifesim_rates(centers, widths)
        t_ours = np.array(
            [
                bin_test_statistic(wl, dwl, b_one, b_full)
                for wl, dwl in zip(centers, widths)
            ]
        )
        t_theirs = inlifesim_test_statistic(centers, widths, theirs)

    fig, (ax_noise, ax_t) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7), height_ratios=(1.4, 1)
    )

    wavelength_um = centers * 1e6
    line_widths = {"InLIFEsim": 1.5, "LIFEsimFT": 1.0}
    for code, values in (("InLIFEsim", theirs), ("LIFEsimFT", ours)):
        for source, style in SOURCES.items():
            ax_noise.step(
                wavelength_um,
                np.sqrt(values[source] * ref.TOTAL_TIME_S),
                where="mid",
                ls=style,
                lw=line_widths[code],
                color=COLORS[code],
                label=f"{code}, {source}",
            )
    ax_noise.set_yscale("log")
    ax_noise.set_ylabel("noise per output, √(n t) (ph)")
    ax_noise.legend(fontsize=8, ncols=2, loc="lower right")
    ax_noise.set_title("(a) photon noise per source", loc="left")

    for code, t_bins in (("InLIFEsim", t_theirs), ("LIFEsimFT", t_ours)):
        total = np.sqrt(np.sum(t_bins**2))
        ax_t.step(
            wavelength_um,
            t_bins,
            where="mid",
            lw=line_widths[code],
            color=COLORS[code],
            label=f"{code}, total S/N = {total:.1f}",
        )
    ax_t.set_xlim(4, 18.5)
    ax_t.xaxis.set_major_locator(MultipleLocator(2))
    ax_t.set_xlabel("wavelength (µm)")
    ax_t.set_ylabel("S/N per bin")
    ax_t.legend(fontsize=8, loc="upper left")
    ax_t.set_title(
        f"(b) S/N (expected T, photon noise only), {ref.PLANET_TEMPERATURE_K:.0f} K blackbody",
        loc="left",
    )

    ratio = t_ours / t_theirs
    print(f"S/N per bin, LIFEsimFT / InLIFEsim: {ratio.min():.4f} to {ratio.max():.4f}")
    print(f"total S/N, LIFEsimFT: {np.sqrt(np.sum(t_ours**2)):.1f}")
    print(f"total S/N, InLIFEsim: {np.sqrt(np.sum(t_theirs**2)):.1f}")

    fig.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "summary_inlifesim.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
