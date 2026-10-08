"""Matched-filter test statistic with photon noise only, Dannert et al. (2025), Sec. 3."""

import numpy as np


def planet_template(differential_signal: np.ndarray) -> np.ndarray:
    """Planet template normalized to unit rms (Eq. B23).

    :param differential_signal: planet photon rate n_L - n_R in ph s^-1, shape (n_t,)
    :return: template, dimensionless, shape (n_t,)
    """
    rms = np.sqrt(np.mean(differential_signal**2))
    return differential_signal / rms


def test_statistic(
    differential_signal: np.ndarray,
    mean_rate_left: float,
    mean_rate_right: float,
    total_time: float,
) -> float:
    """Test statistic T for a planet at a known position, photon noise only.

    T = sqrt(t) * rms(n_p,c) / sqrt(mean_rate_left + mean_rate_right)

    :param differential_signal: planet photon rate n_L - n_R in ph s^-1, shape (n_t,)
    :param mean_rate_left: mean total photon rate in the left output in ph s^-1
    :param mean_rate_right: mean total photon rate in the right output in ph s^-1
    :param total_time: total integration time in s
    :return: T, dimensionless
    """
    return (
        np.sqrt(total_time)
        * np.sqrt(np.mean(differential_signal**2))
        / np.sqrt(mean_rate_left + mean_rate_right)
    )
