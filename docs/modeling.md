# Modeling conventions and assumption record

## Current scope

The intended first model is a static Fourier-transform description of the LIFE
interferometer response. No physics has been implemented. “Static” sets the
initial scope; it does not select an array layout, beam-combiner matrix, source
model, or approximation by itself.

## Conventions to establish before implementation

SI units are the project default. Numerical functions must document physical
units, array shapes, and coordinate definitions. The following choices remain
open and must be resolved explicitly in the task that introduces them:

| Topic | Required record |
| --- | --- |
| Array geometry | Aperture positions, baseline definitions, reference frame, units |
| Sky model | Angular coordinates, brightness units, coherence assumptions, field of view |
| Propagation | Governing equations, approximation regime, OPD and phase sign |
| Fourier transform | Transform pair, 2π factors, spatial frequencies, integration measure |
| Discretization | Grid spacing, ordering, FFT shifts, padding, normalization, convergence |
| Beam combination | Complex coefficients, phase shifts, output ordering, throughput |
| Observable | Amplitude versus intensity, normalization, spectral integration, units |
| Instrument inputs | Parameter provenance, selected values, uncertainty or limitations |

Do not silently assume a far-field or small-angle approximation, monochromatic
illumination, point apertures, ideal nulls, or a specific LIFE baseline ratio.
These may be useful choices, but each needs a stated domain of validity.

## Assumption entry template

Add an entry when a modeling decision is adopted; the template is not a claim
that the decision has been made.

```text
ID and title:
Status: proposed / adopted / superseded
Decision and equation (including units and conventions):
Motivation:
Source: citation, equation/page, or repository URL and commit
Validity range and excluded effects:
Implementation: module/function
Validation: analytic limit, independent reference, or convergence test
Known limitations and superseding entry (if any):
```

## Cross-validation record

LIFEsim, InLIFEsim, and LIFEsimMC are external comparison candidates. Their
authoritative locations, versions, and applicable capabilities have not yet been
verified in this repository. For each comparison, record the tool and commit or
release, exact inputs, matching assumptions, physical observable, unit and
normalization conversions, tolerances and their rationale, result, and known
differences. Archive small reproducible fixtures only when their license and
provenance permit it.
