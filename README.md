# LIFEsimFT

A static, Fourier-transform-based model of the LIFE nulling interferometer,
written as a learning and validation project for an MSc Physics thesis. It follows
the instrument model of Dannert et al. (2025, arXiv:2506.20653) with instrument
perturbations switched off: photon noise only, no instability noise.

**Status:** array geometry and the planet photon rate are implemented and tested.

## Setup

Requires Python 3.11 or newer. In a fresh environment, from the repository root:

```bash
python -m pip install -e '.[dev]'
ruff format . && ruff check .
python -m pytest
```

## Layout

```text
src/lifesimft/reference.py   Reference case, Dannert et al. (2025) Table 1, SI units
src/lifesimft/geometry.py    Array rotation and baseline vectors
src/lifesimft/signal.py      Planet photon rate in one interferometer output (Eq. B12)
tests/                       Physics checks for each module
docs/modeling.md             Conventions and modeling decisions
```

LIFEsim and InLIFEsim are reference projects for comparison; this package does not
depend on them.
