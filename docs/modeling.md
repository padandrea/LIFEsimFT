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
  relative to B12. This package follows B12. InLIFEsim uses x_k − x_j and starts the planet 
  at (−θ, 0); both flips cancel, and planet_response matches planet_photon_rate to 5·10⁻¹⁴ 
  for the reference case (InLIFEsim commit 9505676)
- Unresolved: amplitude normalization. The paper gives A_j = √(A_col η / N_col)
  (below Eq. B7); not yet checked against InLIFEsim `observatory.py`.

  ## Sources

- the planet is a 254 K blackbody (equilibrium temperature), as in the InLIFEsim paper
  notebooks (`temp_p = 254`); still a stand-in for the Alei et al. (2024) Earth spectrum
  used in Dannert et al. (2025).
- the star is a uniform-disk blackbody (Eq. B24).

## Local zodiacal light

- Radiance from the empirical model of Dannert et al. (2022), as implemented in
  InLIFEsim `create_localzodi`; agrees to 7·10⁻⁶.
- Ecliptic coordinates λ_rel = 135°, β = 45° from Dannert et al. (2025) Table 1
  ("star position"), identical to Dannert et al. (2022) Table 1.
- Uniform over the single-mode field of view Ω = π(λ/2D)², so only the j = k terms
  of Eq. B19 survive: n = Δλ I Ω Σ_j A_j². Constant in time and equal in both
  outputs; cancels in n_L − n_R and enters T only as photon noise.
- Reference case at 10 µm: 10.08 ph s⁻¹ per output; T drops from 18.22 to 13.75 (×1.33).


## Exozodiacal dust

- Face-on disk following Kennedy et al. (2015): T(r) = 278.3 K L^¼ (r/au)^−½,
  Σ(r) = z Σ₀ (r/r₀)^−0.34 with Σ₀ = 7.12·10⁻⁸ and r₀ = √L au; zero inside the
  1500 K sublimation radius and outside the 88 K radius.
- Zodi level z = 1, following Dannert et al. (2025) Table 1 (Dannert et al. 2022 uses z = 3).
- Integrated out to the field of view λ/2D, as in InLIFEsim.
- Radially symmetric, so its Fourier transform is the Hankel transform
  Ĩ(q) = ∫ I(θ) J₀(2π q θ) 2πθ dθ with q = |x_jk|/λ. It depends only on baseline
  lengths, so the leakage is constant under rotation and equal in both outputs.
- InLIFEsim `create_exozodi` normalizes Σ to the inner radius, (r/r_in)^−α instead
  of (r/r₀)^−α, which makes its disk fainter by (r_in/r₀)^0.34 ≈ 1/3.14. After
  correcting for this factor, LIFEsimFT agrees with InLIFEsim to 0.1 % in the
  leakage rate. To be clarified with Felix Dannert whether this is intentional.
- Reference case at 10 µm, z = 1: 6.40 ph s⁻¹ per output (InLIFEsim: 2.04).


## Spectral bins and total test statistic

- Band 4–18.5 µm with constant spectral resolution R = λ/Δλ = 33.3, taken from
  Table 1 (0.3 µm at 10 µm); geometric bin edges e_(k+1) = e_k (1 + 1/R), 52 bins.
  The paper does not state the binning behind Fig. 8; to be confirmed.
- Total test statistic as root sum of squares over bins, T = √(Σ T_i²), as in
  Dannert et al. (2025) Fig. 8.
- Noise curves agree with InLIFEsim in all 52 bins up to constants (star 1.0033,
  local zodi 1.0000, planet flux 1.0024, exozodi 3.13–3.14 from its normalization).
- Reference case with a 254 K blackbody planet: S/N = 6.85 at 10.15 µm, total 42.2;
  with InLIFEsim's exozodi normalization 7.42 and 45.1 (paper: 46.6 with the
  Alei et al. 2024 spectrum).