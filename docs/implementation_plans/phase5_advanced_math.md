# Metis Phase 5: Advanced Math Utilities Implementation Plan

**Goal**: Extend Metis with advanced numerical methods for N-dimensional interpolation, root finding, and B-spline support.
**Status**: Planning Draft
**Created**: 2025-12-15

---

## Executive Summary

Phase 5 builds upon the solid foundation of Metis beta 1.0 by adding advanced mathematical utilities commonly needed in engineering and scientific computing. This phase focuses on:

1. **N-Dimensional Interpolation** — Extend beyond 1D with `interpn` for gridded data
2. **B-Spline Interpolation** — Smooth, differentiable interpolation via CasADi
3. **Root Finding** — Implicit equation solving with `rootfinder` wrapper
4. **Enhanced 1D Interpolation** — Additional methods (cubic, nearest) for existing interpolator

> [!IMPORTANT]
> These features extend Metis's numerical toolkit significantly. N-D interpolation and root finding are essential for lookup tables and implicit constraint handling in trajectory optimization.

---

## Gap Analysis: Phase 4 → Phase 5

### ✅ Completed in Phase 4

| Module | Status | Features |
|--------|--------|----------|
| `Interpolate.hpp` | ✅ Basic | 1D linear interpolation with `MetisInterpolator` |
| `Integrate.hpp` | ✅ Complete | `quad`, `solve_ivp` with symbolic support |
| `Quaternion.hpp` | ✅ Complete | Full quaternion algebra with `slerp` |
| `MetisIO.hpp` | ✅ Complete | Graph visualization with DOT export |
| Code Coverage | ✅ Complete | >95% line coverage achieved |

### ❌ Phase 5 Scope

| Priority | Module | Features | Notes |
|----------|--------|----------|-------|
| **P0** | `Interpolate.hpp` | `interpn` (N-D gridded interpolation) | CasADi interpolant with linear/bspline |
| **P0** | `Interpolate.hpp` | B-spline method for 1D/N-D | Smooth, differentiable for optimization |
| **P1** | `RootFinding.hpp` | `rootfinder` wrapper | CasADi Newton-based implicit solver |
| **P1** | `Interpolate.hpp` | `InterpolationMethod` enum | linear/bspline/nearest selection |
| **P2** | `Interpolate.hpp` | Cubic spline (natural) | Higher accuracy for smooth data |

---

## Proposed Implementation Structure

```
include/metis/math/
├── Interpolate.hpp       # [EXTEND] Add interpn, InterpolationMethod enum
├── RootFinding.hpp       # [NEW] Root finding wrapper for CasADi rootfinder
├── MetisMath.hpp         # [EXTEND] Include RootFinding.hpp
└── ...

tests/math/
├── test_interpolate.cpp  # [EXTEND] Add interpn, bspline, cubic tests
├── test_rootfinding.cpp  # [NEW] Root finding tests
└── ...

examples/
├── nd_interpolation_demo.cpp    # [NEW] N-D lookup table example
├── rootfinding_demo.cpp         # [NEW] Implicit equation solving example
└── ...
```

---

## Detailed Implementation Specifications

### Component 1: N-Dimensional Interpolation (`interpn`) — **P0**

**Source Reference**: [interpolate.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/interpolate.py)

#### [EXTEND] `include/metis/math/Interpolate.hpp`

Implement `interpn` for N-dimensional gridded data interpolation, mirroring scipy's `interpolate.interpn`.

```cpp
/**
 * @brief Supported interpolation methods
 */
enum class InterpolationMethod {
    Linear,   // Piecewise linear (C0 continuous)
    Hermite,  // Piecewise cubic Hermite (C1 continuous, monotonicity-preserving)
    BSpline,  // Cubic B-spline (C2 continuous, good for optimization)
    Nearest   // Nearest neighbor (non-differentiable)
};

/**
 * @brief N-dimensional interpolation on regular grids
 *
 * @param points Tuple of 1D coordinate arrays for each dimension (m1, m2, ..., mn)
 * @param values N-dimensional array of values at grid points, shape (m1, m2, ..., mn)
 * @param xi Query points, shape (n_points, n_dimensions)
 * @param method Interpolation method (linear, bspline, nearest)
 * @param fill_value Value for out-of-bounds queries (NaN by default)
 * @return Interpolated values at query points
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> interpn(
    const std::vector<Eigen::VectorXd>& points,
    const Eigen::Tensor<double, Eigen::Dynamic>& values,  // or use flattened vector
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& xi,
    InterpolationMethod method = InterpolationMethod::Linear,
    std::optional<Scalar> fill_value = std::nullopt
);
```

**Implementation Strategy**:

1. **Numeric Backend**: Delegate to CasADi `interpolant` which supports numeric evaluation
2. **Symbolic Backend**: Use `casadi::interpolant` with the same method string
3. **Method Mapping**:
   - `Linear` → `"linear"`
   - `Hermite` → Custom C1 cubic via Catmull-Rom or Akima slopes
   - `BSpline` → `"bspline"`
   - `Nearest` → Implement via rounding to nearest grid point

> [!WARNING]
> CasADi requires `values` to be flattened in Fortran order (`order='F'`). The implementation must handle this correctly.

**Design Decision**: Use `std::vector<Eigen::VectorXd>` for `points` instead of a tuple to maintain C++ simplicity.

---

### Component 2: B-Spline Support for 1D Interpolation — **P0**

#### [EXTEND] `include/metis/math/Interpolate.hpp`

Extend `MetisInterpolator` to support multiple interpolation methods:

```cpp
class MetisInterpolator {
  public:
    /**
     * @brief Construct interpolator with method selection
     *
     * @param x Grid points (must be sorted)
     * @param y Function values
     * @param method Interpolation method (Linear or BSpline)
     */
    MetisInterpolator(
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        InterpolationMethod method = InterpolationMethod::Linear
    );
    
    // ... existing operator() methods unchanged
    
  private:
    InterpolationMethod m_method;
    // Updated CasADi interpolant creation based on method
};
```

**Benefits of B-Spline**:
- C2 continuous (smooth second derivatives)
- Suitable for gradient-based optimization
- Natural curvature behavior

---

### Component 3: Root Finding (`RootFinding.hpp`) — **P1**

#### [NEW] `include/metis/math/RootFinding.hpp`

Provide a Metis root-finding layer for solving implicit equations `F(x) = 0`.

```cpp
#pragma once
#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisFunction.hpp"
#include <casadi/casadi.hpp>

namespace metis {

/**
 * @brief Options for root finding algorithms
 */
struct RootFinderOptions {
    double abstol = 1e-10;          // Absolute tolerance on residual
    double abstolStep = 1e-10;      // Tolerance on step size
    int max_iter = 50;              // Maximum Newton iterations
    bool line_search = true;        // Use line search for globalization
};

/**
 * @brief Result of a root finding operation
 */
template <typename Scalar>
struct RootResult {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x;  // Solution
    int iterations;                               // Number of iterations used
    bool converged;                               // Whether solution converged
};

/**
 * @brief Solve F(x) = 0 for x given an initial guess
 *
 * Uses Newton's method with optional line search. The function F must
 * return a vector of the same dimension as x.
 *
 * @param F Function mapping x -> residual (must be metis::Function)
 * @param x0 Initial guess
 * @param opts Solver options
 * @return RootResult containing solution and diagnostics
 */
template <typename Scalar>
RootResult<Scalar> rootfinder(
    const metis::Function& F,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x0,
    const RootFinderOptions& opts = {}
);

/**
 * @brief Solve implicit function G(x, p) = 0 for x, parametrized by p
 *
 * Creates a function that takes parameters p and returns the solution x(p).
 * Useful for embedding implicit constraints in optimization problems.
 *
 * @param G Implicit function G(x, p) where first input is unknown
 * @param x_guess Initial guess for x
 * @param opts Solver options
 * @return metis::Function mapping p -> x(p)
 */
metis::Function create_implicit_function(
    const metis::Function& G,
    const Eigen::VectorXd& x_guess,
    const RootFinderOptions& opts = {}
);

} // namespace metis
```

**Implementation Notes**:
- Numeric backend: Use Metis-compiled residual/Jacobian kernels with a globalization stack
  (trust-region Newton, line-search Newton, Broyden, pseudo-transient continuation)
- Symbolic backend: Embed `casadi::rootfinder` in the expression graph
- Differentiable implicit solves: CasADi provides exact implicit differentiation

> [!IMPORTANT]
> `create_implicit_function` is particularly powerful for optimization—it allows embedding implicit constraints (like steady-state equations) while maintaining full differentiability.

---

### Component 4: Enhanced 1D Methods — **P2**

#### [EXTEND] `include/metis/math/Interpolate.hpp`

Add natural cubic spline for higher accuracy:

```cpp
/**
 * @brief Natural cubic spline interpolator
 *
 * Implements cubic spline with natural boundary conditions (zero second
 * derivative at endpoints). Higher accuracy than linear for smooth data.
 */
class CubicSplineInterpolator {
  public:
    CubicSplineInterpolator(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    
    template <MetisScalar T>
    T operator()(const T& query) const;
    
  private:
    // Spline coefficients computed at construction
    Eigen::VectorXd m_a, m_b, m_c, m_d;  // Polynomial coefficients per segment
    std::vector<double> m_x;
};
```

---

### Component 5: Gridded C1 Interpolant (Hermite) — **P1**

#### [EXTEND] `include/metis/math/Interpolate.hpp`

Implement a C1-continuous gridded interpolant using piecewise Hermite cubic polynomials. This provides:
- **Continuous first derivatives** (essential for gradient-based optimization)
- **Lower computational cost** than B-splines
- **Monotonicity-preserving** option to avoid overshoots

```cpp
/**
 * @brief C1-continuous gridded interpolation using Hermite cubics
 *
 * Provides smooth first derivatives across cell boundaries, making it
 * suitable for optimization where gradient continuity matters but C2
 * smoothness is not required.
 *
 * Slope estimation methods:
 * - Catmull-Rom: Uses neighboring points for slope estimation
 * - Akima: Weighted average reducing oscillation near outliers
 * - Monotone (PCHIP): Preserves monotonicity, prevents overshoots
 */
enum class HermiteSlopeMethod {
    CatmullRom,  // Standard cubic Hermite with Catmull-Rom slopes
    Akima,       // Akima's slope estimation (reduces oscillation)
    Monotone     // Monotonicity-preserving (PCHIP-style)
};

/**
 * @brief N-dimensional C1 Hermite interpolation on regular grids
 *
 * @param points Coordinate arrays for each dimension
 * @param values N-dimensional array of values
 * @param xi Query points
 * @param slope_method Method for estimating slopes at grid points
 * @param fill_value Value for out-of-bounds (optional extrapolation)
 * @return Interpolated values with C1 continuity
 */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> interpn_hermite(
    const std::vector<Eigen::VectorXd>& points,
    const Eigen::VectorXd& values_flat,  // Fortran-order flattened
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& xi,
    HermiteSlopeMethod slope_method = HermiteSlopeMethod::CatmullRom,
    std::optional<Scalar> fill_value = std::nullopt
);
```

**Implementation Strategy**:

1. **1D Case**: Compute slopes at each grid point using chosen method, then evaluate cubic Hermite basis
2. **N-D Case**: Use tensor-product extension (interpolate along each dimension sequentially)
3. **Symbolic Support**: Express Hermite basis functions using `metis::where` for cell selection

> [!TIP]
> For trajectory optimization, C1 continuity (Hermite) is often sufficient—it ensures gradients are continuous without the additional smoothness constraints of C2 (B-spline) that can introduce oscillations.

---

## Task Breakdown & Milestones

### Milestone 1: N-Dimensional Interpolation (Week 1)
- [x] **Task 1.1**: Add `InterpolationMethod` enum to `Interpolate.hpp`
- [x] **Task 1.2**: Implement `interpn` numeric backend (CasADi interpolant)
- [x] **Task 1.3**: Implement `interpn` symbolic backend
- [x] **Task 1.4**: Add bounds checking and `fill_value` handling
- [x] **Task 1.5**: Write tests for 2D, 3D interpolation cases
- [x] **Task 1.6**: Test linear vs bspline accuracy comparison

### Milestone 2: B-Spline for 1D Interpolation (Week 1)
- [x] **Task 2.1**: Extend `MetisInterpolator` constructor with `method` parameter
- [x] **Task 2.2**: Update CasADi interpolant creation for bspline method
- [x] **Task 2.3**: Update numeric `eval_numeric` for bspline (delegate to CasADi)
- [x] **Task 2.4**: Write tests comparing linear vs bspline smoothness
- [x] **Task 2.5**: Test symbolic differentiation through bspline

### Milestone 3: Root Finding (Week 2)
- [x] **Task 3.1**: Create `RootFinding.hpp` with `RootFinderOptions`, `RootResult`
- [x] **Task 3.2**: Implement `rootfinder` function (numeric & symbolic)
- [x] **Task 3.3**: Implement `create_implicit_function`
- [x] **Task 3.4**: Implement `NewtonSolver` wrapper class for fine control
- [x] **Task 3.5**: Write tests for multi-dimensional root finding
- [x] **Task 3.6**: Test implicit function differentiation

### Milestone 4: Enhanced 1D Natural Cubic Spline (Week 2)
- [ ] **Task 5.1**: Implement `CubicSplineInterpolator` class
- [ ] **Task 5.2**: Compute spline coefficients via tridiagonal solve
- [ ] **Task 5.3**: Implement symbolic evaluation (or numeric-only if complex)
- [ ] **Task 5.4**: Write accuracy tests against known functions

### Milestone 5: Documentation & Polish (Week 3)
- [x] **Task 6.1**: Create `nd_interpolation_demo.cpp` example
- [x] **Task 6.2**: Create `rootfinding_demo.cpp` example
- [x] **Task 6.3**: Update `MetisMath.hpp` includes
- [ ] **Task 6.4**: Update `docs/design_overview.md` with Phase 5 summary
- [ ] **Task 6.5**: Update README with new feature highlights
- [ ] **Task 6.6**: Run full test suite and coverage check

---

## Verification Plan

### Automated Tests

All tests run via existing infrastructure:

```bash
# Build and run all tests
./scripts/ci.sh

# Run specific test files
cd build && ctest -R test_interpolate --output-on-failure
cd build && ctest -R test_rootfinding --output-on-failure
```

### New Test Files

| Test File | Coverage |
|-----------|----------|
| `test_interpolate.cpp` (extended) | `interpn` 2D/3D, bspline, hermite, cubic spline |
| `test_rootfinding.cpp` (new) | `rootfinder`, `create_implicit_function` |

### Specific Test Cases

#### N-Dimensional Interpolation
1. **2D Bilinear**: Interpolate on 2D grid, verify against known function
2. **3D Trilinear**: 3D temperature field lookup table
3. **Method Comparison**: Compare linear vs bspline on sin(x)*cos(y)
4. **Extrapolation**: Out-of-bounds with fill_value
5. **Symbolic**: Verify gradient computation through interpn

#### Root Finding
1. **Quadratic**: Solve x² - 4 = 0, expect x = ±2
2. **Transcendental**: Solve exp(x) - 2 = 0, expect x ≈ 0.693
3. **Multi-dimensional**: Solve [x² + y - 11, x + y² - 7] = 0 (known solution)
4. **Implicit Function**: Create f(p) that solves x³ - p = 0 for x, verify df/dp

#### B-Spline
1. **Smoothness**: Verify second derivative continuity
2. **Optimization**: Use bspline in symbolic expression, compute gradient

#### C1 Hermite Interpolation
1. **Derivative Continuity**: Verify first derivative matches at cell boundaries
2. **Slope Methods**: Compare Catmull-Rom, Akima, Monotone on oscillatory data
3. **Monotonicity**: PCHIP method should not overshoot on monotonic data
4. **N-D Extension**: 2D bicubic Hermite on smooth 2D function

### Dual-Backend Testing Pattern

```cpp
template <typename Scalar>
void test_interpn_2d() {
    // Grid: x = [0, 1], y = [0, 1]
    std::vector<Eigen::VectorXd> points = {...};
    // Values: z = x + y at grid points
    Eigen::MatrixXd values = ...;
    
    // Query at (0.5, 0.5) should give 1.0
    Eigen::Matrix<Scalar, 2, 1> xi;
    xi << 0.5, 0.5;
    
    auto result = metis::interpn(points, values, xi);
    
    if constexpr (std::is_floating_point_v<Scalar>) {
        EXPECT_NEAR(result(0), 1.0, 1e-10);
    } else {
        EXPECT_NEAR(metis::eval(result(0)), 1.0, 1e-9);
    }
}

TEST(InterpnTests, 2D_Numeric) { test_interpn_2d<double>(); }
TEST(InterpnTests, 2D_Symbolic) { test_interpn_2d<casadi::MX>(); }
```

### Manual Verification

1. **Visual Inspection**: Plot bspline vs linear interpolation for smooth function
2. **Gradient Check**: Finite-difference vs AD gradient for interpn
3. **Performance**: Benchmark N-D lookup table performance

---

## Dependencies & Prerequisites

- **Existing**: Eigen ≥ 3.4, CasADi ≥ 3.6, C++20
- **CasADi Features Used**:
  - `casadi::interpolant` with `"linear"`, `"bspline"` methods
  - `casadi::rootfinder` with `"newton"` solver

> [!NOTE]
> No new external dependencies required. All functionality builds on existing CasADi capabilities.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| CasADi bspline edge cases | Medium | Extensive testing, handle zero-values bug |
| N-D tensor flattening order | High | Strict Fortran-order enforcement, clear docs |
| Root finder convergence | Medium | Configurable tolerances, line search option |
| Symbolic cubic spline complexity | Low | Fall back to bspline if too complex |

---

## Success Criteria

1. ☐ `interpn` working for 2D and 3D grids with linear, hermite, and bspline methods
2. ☐ `MetisInterpolator` supports bspline and hermite method selection
3. ☐ C1 Hermite interpolant with Catmull-Rom, Akima, and Monotone slope methods
4. ☐ `rootfinder` solves simple and multi-dimensional equations
5. ☐ `create_implicit_function` produces differentiable implicit constraints
6. ☐ All tests pass for both numeric and symbolic backends
7. ☐ Examples demonstrating N-D interpolation and root finding
8. ☐ CI pipeline (`./scripts/verify.sh`) passes
9. ☐ Code coverage maintained at >90%

---

## Future Considerations (Phase 6+)

| Feature | Notes |
|---------|-------|
| Optimization Solvers | NLP with IPOPT/SNOPT (Phase 6 candidate) |
| Sparse Interpolation | Scattered data interpolation (RBF) |
| Adaptive Mesh | Automatic grid refinement for accuracy |
| Parallel Evaluation | Batch interpolation optimization |

---

*Generated by Metis Dev Team - Phase 5 Planning*
