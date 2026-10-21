"""Compare exozodiacal-light photon rates:
LIFEsimFT vs InLIFEsim vs LIFEsim.

All three implementations are imported and called directly.

Run from the LIFEsimFT repository root:

    python examples/compare_exozodi_three.py
"""

import inspect
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the three actual projects directly
# ---------------------------------------------------------------------------
import lifesim
import matplotlib.pyplot as plt
import numpy as np
from inlifesim.sources import create_exozodi

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions
from lifesimft.signal import exozodi_photon_rate
from lifesimft.spectral import wavelength_bins

# ---------------------------------------------------------------------------
# Confirm which source files are actually being executed
# ---------------------------------------------------------------------------

print("LIFEsim:")
print(inspect.getfile(lifesim.PhotonNoiseExozodi))

print("\nInLIFEsim:")
print(inspect.getfile(create_exozodi))

print("\nLIFEsimFT:")
print(inspect.getfile(exozodi_photon_rate))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISTANCE_PC = 10.0
IMAGE_SIZE = 512

AMPLITUDES = np.full(
    4,
    np.sqrt(
        np.pi
        * (ref.APERTURE_DIAMETER_M / 2.0) ** 2
        * ref.PHOTON_CONVERSION_EFFICIENCY
        / 4.0
    ),
)

COLORS = {
    "LIFEsimFT": "#2a78d6",
    "InLIFEsim": "#2ca02c",
    "LIFEsim": "#FD0000",
}


# ---------------------------------------------------------------------------
# InLIFEsim
# ---------------------------------------------------------------------------


def inlifesim_baselines() -> np.ndarray:
    """Baseline matrix in InLIFEsim layout."""

    x = ref.COLLECTOR_POSITIONS_M[:, 0]
    y = ref.COLLECTOR_POSITIONS_M[:, 1]

    return np.array(
        (
            np.subtract.outer(x, x).T,
            np.subtract.outer(y, y).T,
        )
    )


def inlifesim_exozodi_rates(
    wl_bins: np.ndarray,
    wl_widths: np.ndarray,
) -> np.ndarray:
    """Call InLIFEsim's actual create_exozodi() function."""

    bl = inlifesim_baselines()

    hfov = wl_bins / (2.0 * ref.APERTURE_DIAMETER_M)

    hfov_mas = hfov * 3600000.0 * 180.0 / np.pi

    rad_pix = 2.0 * hfov / IMAGE_SIZE

    au_pix = 2.0 * hfov_mas / IMAGE_SIZE / 1e3 * DISTANCE_PC

    x_map = np.tile(
        np.arange(IMAGE_SIZE),
        (IMAGE_SIZE, 1),
    )

    radius_map = np.hypot(
        x_map - (IMAGE_SIZE - 1) / 2.0,
        x_map.T - (IMAGE_SIZE - 1) / 2.0,
    )

    r_au = radius_map[np.newaxis] * au_pix[:, np.newaxis, np.newaxis]

    # ---------------------------------------------------------------
    # ACTUAL InLIFEsim call
    # ---------------------------------------------------------------

    b_ez = create_exozodi(
        wl_bins,
        wl_widths,
        ref.STAR_LUMINOSITY_LSUN,
        ref.ZODI_LEVEL,
        r_au,
        IMAGE_SIZE,
        au_pix,
        rad_pix,
        radius_map,
        bl,
        hfov,
    )

    # Convert InLIFEsim's Fourier matrix into photon rate in one dark output
    phi = ref.PHASE_LEFT_RAD

    delta = np.cos(phi[:, np.newaxis] - phi[np.newaxis, :])

    weights = AMPLITUDES[:, np.newaxis] * AMPLITUDES[np.newaxis, :] * delta

    return np.sum(
        weights * b_ez,
        axis=(1, 2),
    )


# ---------------------------------------------------------------------------
# LIFEsimFT
# ---------------------------------------------------------------------------


def lifesimft_exozodi_rates(
    wl_bins: np.ndarray,
    wl_widths: np.ndarray,
) -> np.ndarray:
    """Call LIFEsimFT's actual exozodi_photon_rate() function."""

    b = baselines(
        rotate_positions(
            ref.COLLECTOR_POSITIONS_M,
            np.zeros(1),
        )
    )

    rates = []

    for wavelength, bandwidth in zip(
        wl_bins,
        wl_widths,
    ):
        # -----------------------------------------------------------
        # ACTUAL LIFEsimFT call
        # -----------------------------------------------------------

        rate = exozodi_photon_rate(
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

        rates.append(rate)

    return np.asarray(rates)


# ---------------------------------------------------------------------------
# LIFEsim
# ---------------------------------------------------------------------------


def lifesim_exozodi_rates(
    wl_bins: np.ndarray,
    wl_widths: np.ndarray,
) -> np.ndarray:
    """Call LIFEsim's actual PhotonNoiseExozodi.noise() method."""

    bus = lifesim.Bus()

    bus.data.options.set_scenario("baseline")

    bus.data.options.set_manual(
        diameter=ref.APERTURE_DIAMETER_M,
        baseline=14.5,
        ratio=6.0,
        quantum_eff=0.7,
        throughput=0.05,
        image_size=IMAGE_SIZE,
    )

    instrument = lifesim.Instrument(name="inst")
    transmission = lifesim.TransmissionMap(name="transm")
    exozodi = lifesim.PhotonNoiseExozodi(name="exo")

    bus.add_module(instrument)
    bus.add_module(transmission)
    bus.add_module(exozodi)

    bus.connect(("inst", "transm"))
    bus.connect(("inst", "exo"))

    instrument.apply_options()

    # Force same wavelength bins as the other two codes
    bus.data.inst["wl_bins"] = np.asarray(wl_bins)
    bus.data.inst["wl_bin_widths"] = np.asarray(wl_widths)

    diameter = ref.APERTURE_DIAMETER_M

    bus.data.inst["hfov"] = wl_bins / (2.0 * diameter)

    bus.data.inst["hfov_mas"] = bus.data.inst["hfov"] * 3600000.0 * 180.0 / np.pi

    bus.data.inst["rad_pix"] = 2.0 * bus.data.inst["hfov"] / IMAGE_SIZE

    bus.data.inst["mas_pix"] = 2.0 * bus.data.inst["hfov_mas"] / IMAGE_SIZE

    bus.data.single["l_sun"] = ref.STAR_LUMINOSITY_LSUN

    bus.data.single["distance_s"] = DISTANCE_PC
    bus.data.single["z"] = ref.ZODI_LEVEL

    # Actual LIFEsim transmission map
    _, _, tm3, _, _ = transmission.transmission_map(map_selection=["tm3"])

    bus.data.inst["t_map"] = tm3

    # ---------------------------------------------------------------
    # ACTUAL LIFEsim call
    # ---------------------------------------------------------------

    exozodi_raw = exozodi.noise(index=None)

    # LIFEsim applies this efficiency later in Instrument.get_snr().
    # Apply it here because LIFEsimFT/InLIFEsim amplitudes already contain
    # the same 0.7 * 0.05 = 0.035 photon conversion efficiency.
    return np.asarray(exozodi_raw) * bus.data.inst["eff_tot"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:

    centers, widths = wavelength_bins(
        ref.WAVELENGTH_MIN_M,
        ref.WAVELENGTH_MAX_M,
        ref.SPECTRAL_RESOLUTION,
    )

    wavelength_um = centers * 1e6

    # ---------------------------------------------------------------
    # Run the three actual implementations
    # ---------------------------------------------------------------

    exozodi_ft = lifesimft_exozodi_rates(
        centers,
        widths,
    )

    exozodi_inlife = inlifesim_exozodi_rates(
        centers,
        widths,
    )

    exozodi_life = lifesim_exozodi_rates(
        centers,
        widths,
    )

    # ---------------------------------------------------------------
    # Ratios
    # ---------------------------------------------------------------

    ratio_ft_life = exozodi_ft / exozodi_life

    ratio_inlife_life = exozodi_inlife / exozodi_life

    print()
    print("Exozodi comparison")
    print("-------------------")

    print(f"LIFEsimFT / LIFEsim: {ratio_ft_life.min():.4f} – {ratio_ft_life.max():.4f}")

    print(
        "InLIFEsim / LIFEsim:"
        f" {ratio_inlife_life.min():.4f}"
        f" – {ratio_inlife_life.max():.4f}"
    )

    print(
        "LIFEsim / InLIFEsim:"
        f" {(exozodi_life / exozodi_inlife).min():.4f}"
        f" – {(exozodi_life / exozodi_inlife).max():.4f}"
    )

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------

    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        height_ratios=(2.2, 1.0),
    )

    ax.plot(
        wavelength_um,
        exozodi_ft,
        color=COLORS["LIFEsimFT"],
        lw=2.0,
        label="LIFEsimFT",
    )

    ax.plot(
        wavelength_um,
        exozodi_inlife,
        color=COLORS["InLIFEsim"],
        lw=2.0,
        label="InLIFEsim",
    )

    ax.plot(
        wavelength_um,
        exozodi_life,
        color=COLORS["LIFEsim"],
        lw=2.0,
        label="LIFEsim",
        ls=":",
    )

    ax.set_yscale("log")

    ax.set_ylabel("Exozodi photon rate per output [ph s$^{-1}$]")

    ax.set_title("Exozodiacal-light comparison")

    ax.legend()
    ax.grid(alpha=0.25)

    ax_ratio.plot(
        wavelength_um,
        ratio_ft_life,
        color=COLORS["LIFEsimFT"],
        lw=2.0,
        label="LIFEsimFT / LIFEsim",
    )

    ax_ratio.plot(
        wavelength_um,
        ratio_inlife_life,
        color=COLORS["InLIFEsim"],
        lw=2.0,
        label="InLIFEsim / LIFEsim",
    )

    ax_ratio.axhline(
        1.0,
        color=COLORS["LIFEsim"],
        lw=1.5,
        ls=":",
        label="LIFEsim",
    )

    ax_ratio.set_xlabel("Wavelength [µm]")

    ax_ratio.set_ylabel("Ratio to LIFEsim")

    ax_ratio.grid(alpha=0.25)
    ax_ratio.legend(fontsize=9)

    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent.parent / "outputs"

    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "exozodi_three_code_comparison.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(f"Saved plot to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
