"""Planet signal over one array rotation for an Earth twin at 10 pc (cf. Fig. 1)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lifesimft import reference as ref
from lifesimft.geometry import baselines, rotate_positions, rotation_angles
from lifesimft.signal import planet_photon_rate

EARTH_AT_10_PC_RAD = 4.848e-7
N_SAMPLES = 720


def main() -> None:
    angles = rotation_angles(n_rotations=1, n_samples=N_SAMPLES)
    b = baselines(rotate_positions(ref.COLLECTOR_POSITIONS_M, angles))
    planet_position = np.array([EARTH_AT_10_PC_RAD, 0.0])

    common = dict(
        flux_density=1.0,
        amplitudes=np.ones(4),
        baselines=b,
        planet_position=planet_position,
        wavelength=ref.WAVELENGTH_M,
        bandwidth=1.0,
    )

    n_left = planet_photon_rate(phases=ref.PHASE_LEFT_RAD, **common)
    n_right = planet_photon_rate(phases=ref.PHASE_RIGHT_RAD, **common)
    n_diff = n_left - n_right

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    angle_deg = np.degrees(angles)
    ax_top.plot(angle_deg, n_left, label="left output")
    ax_top.plot(angle_deg, n_right, label="right output")
    ax_top.set_ylabel("photon rate (normalized)")
    ax_top.legend()
    ax_bottom.plot(angle_deg, n_diff, color="black")
    ax_bottom.set_xlabel("array rotation angle (deg)")
    ax_bottom.set_ylabel("differential L − R")
    fig.tight_layout()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "planet_signal.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()