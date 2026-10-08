# Modeling decisions

Reference: Dannert et al. (2025), arXiv:2506.20653. Units are SI throughout
(metres, radians, seconds).

## Scope

Static model: no phase, amplitude or polarization perturbations (δβ = 0, so
cos Δϑ_jk = 1). Noise will be photon noise only.

## Geometry

- Collector positions, baselines and observation timing from Table 1
  (`reference.py`).
- Collector numbering follows Table 1; Python index 0 is collector 1.
  Nulling baseline: collectors 1–3 (14.5 m). Imaging baseline: collectors 1–2 (87 m).
- Rotation is counter-clockwise for positive angles, about the array centre.
- Each detector integration is treated as a snapshot at the array angle at its
  start; rotation during one exposure (360°/155 ≈ 2.3° for the reference case)
  is neglected.
- Angles exclude the endpoint 2π·N_rot, so the time series is periodic.
- Baselines are x_jk = x_j − x_k.

## Planet signal

- Point source, constant over the wavelength bin:
  n(t) = Δλ F_p Σ_jk A_j A_k cos(Δφ_jk + 2π/λ · x_jk(t)·θ_p), from Eq. B12 with
  Eqs. B13–B14 and B20.
- Sign: Eq. B22 corresponds to cos(Δφ_jk − 2π/λ · x_jk·θ_p), i.e. θ_p → −θ_p
  relative to B12. This package follows B12. Unresolved: which convention
  InLIFEsim uses; check before comparing templates.
- Unresolved: amplitude normalization. The paper gives A_j = √(A_col η / N_col)
  (below Eq. B7); not yet checked against InLIFEsim `observatory.py`.
