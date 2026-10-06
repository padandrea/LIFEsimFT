# LIFEsimFT project instructions

## Scope

LIFEsimFT is a standalone scientific Python package for MSc Physics thesis work
on LIFE nulling interferometry. The immediate objective is an explicit, static
Fourier-transform-based instrument-response model for learning and validation.
The repository currently contains a scaffold, not a working simulator.

Implement physics incrementally when requested. Do not expand a scaffold or a
focused modeling task into a full simulator. Time-dependent instrument behavior,
noise budgets, detector models, observing strategies, population synthesis, and
retrievals require separate scope decisions. Do not introduce assumed LIFE design
parameters merely to obtain a runnable example.

## Relationship to existing tools

- **LIFEsim:** an external reference for LIFE simulation workflows and compatible
  instrument-response comparisons. LIFEsimFT remains independently installable.
- **InLIFEsim:** an additional external instrument-model comparison point. Verify
  the relevant implementation and conventions before translating any equations.
- **LIFEsimMC:** an additional external reference for cross-validation where its
  modeled quantities and assumptions overlap this static model. Do not infer its
  method or feature set from its name.

No particular repository URL, version, reference dataset, or adapter has been
selected for these comparisons. Verify and record these before using a tool as a
benchmark. Do not claim agreement from similar names, plots, or output shapes.
Compare the same physical observable with matching geometry, wavelength, units,
phase and Fourier conventions, normalization, and included effects. Document
remaining differences. Do not copy external code without checking its license
and preserving required attribution; cite sources for adopted equations.

## Physics and traceability

- Keep geometry, complex field propagation, beam combination, and observables
  conceptually separate. Introduce modules as these responsibilities are needed.
- State the mathematical model and its assumptions before optimizing it. Keep
  instrument parameters explicit in arguments or named configuration objects.
- Use SI units internally by default: lengths and wavelengths in metres, angles
  in radians, and times in seconds. Document conversions at input/output
  boundaries. Document the units of every physical input and output, including
  dimensionless quantities. Do not silently mix raw arrays and unit quantities.
- Document array shapes and axis meanings in numerical APIs. Specify coordinate
  handedness, baseline direction, angular origin, and wavelength ordering.
- Define Fourier signs, factors of 2π, frequency coordinates, normalization,
  sampling, and shifts explicitly. Never treat an FFT convention as an unstated
  physical choice. Distinguish optical path difference from phase.
- Distinguish complex amplitude, intensity transmission, flux, and detected
  counts. Record throughput and beam-combiner normalization. Do not imply
  absolute photometry when only normalized response is modeled.
- Record each adopted approximation, parameter source, equation reference, and
  validity range in `docs/modeling.md` or linked focused notes. Unresolved choices
  must remain visibly unresolved. Tie implemented assumptions to tests and code.

## Python and repository conventions

- Use the existing checkout; cloud tasks are already isolated. Do not create Git
  worktrees unless explicitly requested.
- Use the `src/lifesimft` package layout, Python 3.11+, type annotations for public
  APIs, and NumPy-style docstrings with units and shapes for numerical functions.
- Prefer small, readable functions and explicit configuration. Avoid global
  mutable state, import-time computation, and unnecessary abstractions.
- Keep reusable physics in the package. Examples and notebooks should call that
  API rather than duplicate equations. Seed stochastic examples if introduced.
- Declare dependencies in `pyproject.toml` when needed; avoid speculative runtime
  dependencies. Do not edit generated packaging outputs or commit local virtual
  environments, credentials, large datasets, or generated figures by default.
- Follow Ruff formatting and lint checks. Preserve unrelated user changes.

## Validation and development commands

From the repository root, create/activate `.venv` and install with
`python -m pip install -e '.[dev]'`. Use `ruff check .`,
`ruff format --check .`, and `python -m build` for packaging work.

When physics is implemented, add pytest tests under `tests/` and run
`python -m pytest`. Prioritize analytic limits, symmetry, expected nulls,
normalization or conservation where applicable, and sampling convergence. Use
independent analytic or direct-summation references where possible. State the
physical or numerical reason for tolerances, including absolute tolerances near
nulls. An empty test run, successful import, or visually plausible plot does not
validate instrument physics.

External comparison tests must identify the reference tool/version and input
configuration, and distinguish unavailable optional references from failed
comparisons. Report passed, failed, skipped, and unrun checks accurately. Never
change a reference result or weaken an assertion solely to make a test pass.
