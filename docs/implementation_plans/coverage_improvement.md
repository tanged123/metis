# Implementation Plan - Code Coverage Improvements

## Goal
Improve code coverage for `include/metis/math/IntegrateDiscrete.hpp` and `include/metis/math/Calculus.hpp` by identifying missing test cases and implementing them.

## User Review Required
- None currently.

## Proposed Changes

### `tests/math/`
#### [MODIFY] `tests/math/test_integrate_discrete.cpp`
- Add `test_integrate_simpson_combined`: Test `method="simpson"` explicitly.
- Add `test_integrate_backward_simpson`: Test `method="backward_simpson"`.
- Add `test_integrate_endpoints_ignore`: Test `method_endpoints="ignore"` and verify output size.
- Add `test_squared_curvature_hybrid`: Test `method="hybrid_simpson_cubic"`.
- Add `test_invalid_inputs`: Verify exceptions for unknown methods.

#### [MODIFY] `tests/math/test_calculus.cpp`
- Add `test_gradient_periodic`: Call `gradient_periodic`.
- Add `test_gradient_errors`: Verify exceptions for invalid `n`, `edge_order`, and `dx` dimensions.

## Verification Plan
### Automated Tests
- Run `scripts/test.sh`.
- Run coverage generation (if available locally) or rely on creating comprehensive tests covering all code paths found during analysis.
