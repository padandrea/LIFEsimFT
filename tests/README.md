# Validation tests

There are no executable tests yet because no model is implemented. Add tests
alongside the first physics implementation and run them with `python -m pytest`
from the repository root after installing `.[dev]`.

Use analytic limits and independent numerical references for instrument physics.
Document units, shapes, assumptions, and tolerance choices in fixtures and tests.
Keep external-tool comparisons separate from local unit tests so their additional
requirements and any skips are explicit. A zero-test run is not a passing suite.
