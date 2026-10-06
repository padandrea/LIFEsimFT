# LIFEsimFT

LIFEsimFT is a standalone scientific Python package developed as part of an MSc
Physics thesis on LIFE nulling interferometry. Its immediate goal is a static
Fourier-transform-based instrument model for learning, understanding the response,
and validation against existing LIFE simulation tools.

**Status:** package scaffold only. No interferometer model is implemented yet.

## Development

Requires Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
python -m build
```

Run `python -m pytest` once executable tests have been added. At this stage there
are no physics tests; pytest's “no tests collected” result is not validation.
The initial scaffold has no runtime dependencies. Add scientific libraries when
an implemented model needs them, and declare them in `pyproject.toml`.

## Layout

```text
src/lifesimft/       Importable Python package; future instrument model
tests/              Future analytic and numerical validation tests
docs/               Physics conventions, assumptions, and validation records
examples/           Future small, reproducible examples using the package API
AGENTS.md           Scope and contribution instructions for coding agents
pyproject.toml      Package metadata and development-tool configuration
```

Read [AGENTS.md](AGENTS.md) before making changes and
[the modeling notes](docs/modeling.md) before implementing physics. LIFEsim,
InLIFEsim, and LIFEsimMC are reference and comparison projects; this package does
not currently depend on, wrap, or claim numerical equivalence with them.
