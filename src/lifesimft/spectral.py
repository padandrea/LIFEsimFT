"""Spectral binning of the LIFE wavelength band."""

import numpy as np


def wavelength_bins(
    wavelength_min: float, wavelength_max: float, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength bins of constant spectral resolution R = lambda / delta lambda.

    Bin edges are geometric, e_(k+1) = e_k (1 + 1 / R), starting at wavelength_min;
    the last edge is the first one at or beyond wavelength_max.

    :param wavelength_min: lower edge of the first bin in m
    :param wavelength_max: upper end of the band in m
    :param resolution: spectral resolution R, dimensionless
    :return: bin centres in m and bin widths in m, each shape (n_bins,)
    """
    n_bins = int(
        np.ceil(np.log(wavelength_max / wavelength_min) / np.log(1 + 1 / resolution))
    )
    edges = wavelength_min * (1 + 1 / resolution) ** np.arange(n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    return centers, widths
