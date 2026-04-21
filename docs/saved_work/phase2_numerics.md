# Phase 2: Math & Numerics Layer - Completion Report

**Status**: Complete & Verified
**Date**: 2025-12-12
**Tests**: `tests/test_math.cpp` (Passed 100%)

## Implemented Components

The following headers have been created in `include/metis/math/` to provide a dual-backend (Numeric `double` / Symbolic `casadi::MX`) mathematical core.

### 1. Core Arithmetic & Trigonometry
*   **Files**: `Arithmetic.hpp`, `Trig.hpp`
*   **Features**: `abs`, `sqrt`, `pow`, `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`.
*   **Implementation**: Uses `if constexpr` to dispatch to `std::` or `casadi::` (via Argument Dependent Lookup for `fabs` etc).

### 2. Logic & Control Flow
*   **Files**: `Logic.hpp`
*   **Features**:
    *   `metis::where(cond, if_true, if_false)`: Dispatches to ternary operator (numeric) or `casadi::if_else` (symbolic).
    *   `metis::sigmoid_blend`: Smooth blending function for optimization stability.

### 3. Differential Operators
*   **Files**: `DiffOps.hpp`
*   **Features**:
    *   `diff`: Adjacent difference on Eigen vectors.
    *   `trapz`: Trapezoidal integration.
    *   `gradient_1d`: Central difference gradient estimation.
*   **Note**: Fully vectorized using Eigen expressions.

### 4. Linear Algebra
*   **Files**: `Linalg.hpp`
*   **Features**:
    *   `solve(A, b)`: Uses `A.colPivHouseholderQr().solve(b).eval()` for numeric stability, `casadi::MX::solve` for symbolic.
    *   `norm(x)`: L2 norm.
    *   `outer(x, y)`: Outer product.
    *   Helpers: `to_mx` and `to_eigen` for conversion.

### 5. Interpolation
*   **Files**: `Interpolate.hpp`
*   **Features**: `MetisInterpolator` class.
    *   **Numeric**: STL `upper_bound` + linear interpolation.
    *   **Symbolic**: Wraps `casadi::interpolant`.

### 6. Utilities
*   **Files**: `Spacing.hpp`, `Rotations.hpp`
*   **Features**: `linspace`, `cosine_spacing`, `rotation_matrix_2d`, `rotation_matrix_3d`.

## Verification

A comprehensive test suite was added in `tests/test_math.cpp`.
To run tests:
```bash
./scripts/ci.sh
```
Or manually:
```bash
./scripts/build.sh
./scripts/test.sh
```

## Critical Design Notes for Next Phase
1.  **Eigen Evaluation**: When using `Ax=b` solvers in Eigen with CasADi types, the result expression from `solve` relies on temporary decomposition objects. Always call `.eval()` or assign immediately to a concrete matrix to avoid dangling references (Fixed in `Linalg.hpp`).
2.  **ADL**: CasADi functions like `fabs` are often hidden friends. Call them without `casadi::` prefix or use `using` declarations if inside the `metis` namespace.
3.  **Template Constraints**: Continue using `MetisScalar` concept to enforce valid types.

## File Manifest
*   `include/metis/math/Arithmetic.hpp`
*   `include/metis/math/Trig.hpp`
*   `include/metis/math/Logic.hpp`
*   `include/metis/math/DiffOps.hpp`
*   `include/metis/math/Linalg.hpp`
*   `include/metis/math/Interpolate.hpp`
*   `include/metis/math/Spacing.hpp`
*   `include/metis/math/Rotations.hpp`
*   `tests/test_math.cpp`
*   `scripts/ci.sh`
*   `scripts/build.sh` (updated)
*   `scripts/test.sh` (updated)
