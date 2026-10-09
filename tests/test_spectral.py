import numpy as np
import pytest

from lifesimft import reference as ref
from lifesimft.spectral import wavelength_bins

CENTERS, WIDTHS = wavelength_bins(
    ref.WAVELENGTH_MIN_M, ref.WAVELENGTH_MAX_M, ref.SPECTRAL_RESOLUTION
)


def test_band_is_covered():
    assert CENTERS[0] - WIDTHS[0] / 2 == pytest.approx(ref.WAVELENGTH_MIN_M)
    assert CENTERS[-1] + WIDTHS[-1] / 2 >= ref.WAVELENGTH_MAX_M


def test_bins_are_contiguous():
    upper = CENTERS[:-1] + WIDTHS[:-1] / 2
    lower = CENTERS[1:] - WIDTHS[1:] / 2
    np.testing.assert_allclose(upper, lower)


def test_resolution_is_constant():
    np.testing.assert_allclose(CENTERS / WIDTHS, ref.SPECTRAL_RESOLUTION + 0.5)


def test_reference_band_has_52_bins():
    assert CENTERS.size == 52


@pytest.mark.parametrize(
    ("wavelength_min", "wavelength_max", "resolution"),
    [(4e-6, 18.5e-6, 20.0), (3e-6, 20e-6, 50.0), (6e-6, 17e-6, 100.0)],
)
def test_band_is_covered_with_minimal_bins(wavelength_min, wavelength_max, resolution):
    centers, widths = wavelength_bins(wavelength_min, wavelength_max, resolution)
    upper = centers + widths / 2
    assert upper[-1] >= wavelength_max
    assert upper[-2] < wavelength_max
