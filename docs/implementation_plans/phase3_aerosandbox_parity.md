# Metis Phase 3: AeroSandbox Numerical Parity Implementation Plan

**Goal**: Complete 1:1 capability match with [AeroSandbox numpy module](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/__init__.py).
**Status**: Planning Draft
**Created**: 2025-12-14

---

## Executive Summary

Phase 3 completes the Metis framework by implementing all remaining functions from AeroSandbox's `numpy` module. This establishes full capability parity, enabling any AeroSandbox physics model to be ported to Metis C++ with minimal refactoring.

> [!IMPORTANT]
> **Architectural Constraint**: All implementations MUST be templated on `Scalar` and dispatch correctly to both:
> - **Numeric Mode**: `double` / `Eigen::MatrixXd` (standard execution)
> - **Symbolic Mode**: `casadi::MX` / `Eigen::Matrix<casadi::MX>` (graph generation)

---

## Gap Analysis: Current State vs. AeroSandbox

### ✅ Already Implemented (Phase 2 Complete)

| Module | Metis Header | Functions |
|--------|-------------|-----------|
| `arithmetic_*.py` | [Arithmetic.hpp](file:///home/tanged/sources/metis/include/metis/math/Arithmetic.hpp) | `abs`, `sqrt`, `pow`, `exp`, `log`, `floor`, `ceil`, `fmod` |
| `trig.py` | [Trig.hpp](file:///home/tanged/sources/metis/include/metis/math/Trig.hpp) | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `hypot`, `sinh`, `cosh`, `tanh` |
| `conditionals.py` | [Logic.hpp](file:///home/tanged/sources/metis/include/metis/math/Logic.hpp) | `where`, `min`, `max`, `clamp` |
| `linalg*.py` (partial) | [Linalg.hpp](file:///home/tanged/sources/metis/include/metis/math/Linalg.hpp) | `solve`, `norm`, `outer`, `dot`, `cross`, `inv`, `det` |
| `calculus.py` (partial) | [DiffOps.hpp](file:///home/tanged/sources/metis/include/metis/math/DiffOps.hpp) | `diff`, `trapz`, `gradient_1d`, `jacobian` |
| `interpolate.py` | [Interpolate.hpp](file:///home/tanged/sources/metis/include/metis/math/Interpolate.hpp) | `MetisInterpolator` class |
| `spacing.py` | [Spacing.hpp](file:///home/tanged/sources/metis/include/metis/math/Spacing.hpp) | `linspace`, `cosine_spacing` |
| `rotations.py` (partial) | [Rotations.hpp](file:///home/tanged/sources/metis/include/metis/math/Rotations.hpp) | `rotation_matrix_2d`, `rotation_matrix_3d` |

---

### ❌ Not Yet Implemented (Phase 3 Scope)

> [!NOTE]
> **Design Decision (Approved)**: Array operations use **Eigen directly** via the existing `metis::MetisMatrix<Scalar>` type alias. Only targeted free functions (e.g., `metis::roll`) are added where Eigen lacks native support. No `Array.hpp` wrapper needed.

| Priority | AeroSandbox Module | New Metis Header | Functions to Implement |
|----------|-------------------|------------------|------------------------|
| **P0** | `array.py` | *(Eigen native)* | Most handled by Eigen; add `metis::roll` to `Logic.hpp` if needed |
| **P0** | `logicals.py` | (extend `Logic.hpp`) | `logical_and`, `logical_or`, `logical_not`, `clip`, `all`, `any` |
| **P1** | `calculus.py` (full) | (extend `DiffOps.hpp`) | `gradient` (edge_order, n-th derivative, period support) |
| **P1** | `surrogate_model_tools.py` | `SurrogateModel.hpp` | `softmax`, `softmin`, `softplus`, `sigmoid`, `swish`, `blend` |
| **P1** | `linalg.py` (full) | (extend `Linalg.hpp`) | `pinv`, `inner` |
| **P2** | `finite_difference_operators.py` | `FiniteDifference.hpp` | `finite_difference_coefficients` |
| **P2** | `rotations.py` (full) | (extend `Rotations.hpp`) | `rotation_matrix_from_euler_angles`, `is_valid_rotation_matrix` |
| **P2** | `integrate_discrete.py` | `IntegrateDiscrete.hpp` | `integrate_discrete_intervals`, `integrate_discrete_squared_curvature` |
| **P3** | `integrate.py` | `Integrate.hpp` | `quad`, `solve_ivp` (ODE integration with CasADi CVODES) |
| **P3** | `arithmetic_dyadic.py` | (extend `Arithmetic.hpp`) | `mod`, `centered_mod`, broadcasting helpers |

> [!NOTE]
> **Priority Legend**:
> - **P0**: Core utilities needed by almost all downstream code
> - **P1**: Common numerical patterns in aerospace/physics models
> - **P2**: Specialized utilities for advanced models
> - **P3**: ODE/integration layer (complex, may defer to Phase 4)

---

## Proposed Implementation Structure

```
include/metis/math/
├── Arithmetic.hpp      # [EXTEND] Add mod, centered_mod
├── Calculus.hpp        # [NEW] Full gradient with edge_order
├── DiffOps.hpp         # [EXTEND] Link to Calculus, keep diff/trapz
├── FiniteDifference.hpp# [NEW] FD coefficient generation
├── Integrate.hpp       # [NEW] ODE integration (quad, solve_ivp)
├── IntegrateDiscrete.hpp # [NEW] Discrete interval integration
├── Interpolate.hpp     # [EXISTS]
├── MetisMath.hpp       # [EXTEND] Include new headers
├── Linalg.hpp          # [EXTEND] Add pinv, inner, roll (if needed)
├── Logic.hpp           # [EXTEND] Add logical_and/or/not, all, any
├── Rotations.hpp       # [EXTEND] Add euler_angles, validation
├── Spacing.hpp         # [EXISTS]
├── SurrogateModel.hpp  # [NEW] Optimization-friendly smooth functions
└── Trig.hpp            # [EXISTS]
```

> Array operations: Use Eigen directly (`<<`, `.reshaped()`, `.replicate()`, etc.)

---

## Detailed Implementation Specifications

### Component 1: Array Operations - **Use Eigen Directly**

**Design Decision (Approved)**: No `Array.hpp` needed. Use Eigen's native operations:

| Operation | Eigen Equivalent |
|-----------|-----------------|
| Concatenate | `<<` operator, `block()` |
| Stack | Block operations |
| Reshape | `.reshaped()` (Eigen 3.4+) |
| Diagonal | `.diagonal()`, `.asDiagonal()` |
| Zeros/Ones | `MatrixXd::Zero(rows, cols)`, `MatrixXd::Ones(...)` |
| Tile | `.replicate(n, m)` |
| Length | `.size()` |
| Max/Min | `.maxCoeff()`, `.minCoeff()`, `.colwise().maxCoeff()` |

**Only add if frequently needed**: `metis::roll(matrix, shift, axis)` as a free function in `Linalg.hpp`.

---

### Component 2: Extended Logic (`Logic.hpp`) - **P0**

**Source Reference**: [logicals.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/logicals.py)

#### [MODIFY] `include/metis/math/Logic.hpp`

Add the following functions:

| Function | Signature | Numeric Backend | Symbolic Backend |
|----------|-----------|-----------------|------------------|
| `logical_and` | `auto logical_and(x1, x2)` | `x1 && x2` | `casadi::logic_and` |
| `logical_or` | `auto logical_or(x1, x2)` | `x1 \|\| x2` | `casadi::logic_or` |
| `logical_not` | `auto logical_not(x)` | `!x` | `casadi::logic_not` |
| `all` | `auto all(a)` | Eigen `.all()` | `casadi::logic_all` |
| `any` | `auto any(a)` | Eigen `.any()` | `casadi::logic_any` |
| `clip` | `auto clip(x, lo, hi)` | `fmin(fmax(x, lo), hi)` | Use `min`/`max` |

---

### Component 3: Enhanced Calculus (`Calculus.hpp`) - **P1**

**Source Reference**: [calculus.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/calculus.py)

#### [NEW] `include/metis/math/Calculus.hpp`

The existing `gradient_1d` is a simplified version. The full `gradient` function needs:

```cpp
/**
 * @brief Full gradient computation with edge_order and n-th derivative support
 * @param f Function values
 * @param varargs Spacing (uniform or array)
 * @param axis Axis along which to compute gradient
 * @param edge_order 1 or 2 for boundary accuracy
 * @param n Derivative order (1 or 2)
 * @param period Optional periodicity for wrapped data
 */
template <typename Derived, typename... Args>
auto gradient(const Eigen::MatrixBase<Derived>& f, Args&&... args);
```

Key implementation details:
- Second-order accurate central differences in interior
- First or second order one-sided differences at boundaries
- Support for non-uniform spacing via `varargs`
- `n=2` for second derivative with reduced error

---

### Component 4: Surrogate Model Tools (`SurrogateModel.hpp`) - **P1**

**Source Reference**: [surrogate_model_tools.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/surrogate_model_tools.py)

#### [NEW] `include/metis/math/SurrogateModel.hpp`

These are essential for optimization-friendly physics models:

| Function | Purpose | Formula |
|----------|---------|---------|
| `softmax` | Differentiable max | `softness * log(sum(exp(x_i/softness)))` |
| `softmin` | Differentiable min | `-softmax(-args)` |
| `softplus` | Smooth ReLU | `(1/beta) * log(1 + exp(beta * x))` |
| `sigmoid` | S-curve activation | `tanh`, `logistic`, `arctan`, or `polynomial` |
| `swish` | Smooth activation | `x / (1 + exp(-beta * x))` |
| `blend` | Smooth transition | `sigmoid(switch) * high + (1-sigmoid(switch)) * low` |

> [!TIP]
> The existing `sigmoid_blend` in `Logic.hpp` covers part of this. Consider moving to `SurrogateModel.hpp` for organization.

---

### Component 5: Extended Linear Algebra (`Linalg.hpp`) - **P1**

**Source Reference**: [linalg.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/linalg.py)

#### [MODIFY] `include/metis/math/Linalg.hpp`

| Function | Signature | Numeric Backend | Symbolic Backend |
|----------|-----------|-----------------|------------------|
| `pinv` | `auto pinv(A)` | `Eigen::completeOrthogonalDecomposition().pseudoInverse()` | `casadi::pinv` |
| `inner` | `auto inner(x, y)` | `x.dot(y)` | `casadi::dot` |
| `norm` (extend) | `auto norm(x, ord, axis, keepdims)` | Handle `ord=1`, `ord=inf`, `ord='fro'` | CasADi equivalents |

---

### Component 6: Finite Difference Operators (`FiniteDifference.hpp`) - **P2**

**Source Reference**: [finite_difference_operators.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/finite_difference_operators.py)

#### [NEW] `include/metis/math/FiniteDifference.hpp`

```cpp
/**
 * @brief Computes finite difference coefficients for arbitrary grids
 * 
 * Based on Fornberg 1988: "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids"
 * 
 * @param x Grid points
 * @param x0 Evaluation point
 * @param derivative_degree Order of derivative
 * @return Coefficient vector
 */
template <typename Derived>
Eigen::VectorXd finite_difference_coefficients(
    const Eigen::MatrixBase<Derived>& x,
    double x0 = 0.0,
    int derivative_degree = 1
);
```

---

### Component 7: Extended Rotations (`Rotations.hpp`) - **P2**

**Source Reference**: [rotations.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/rotations.py)

#### [MODIFY] `include/metis/math/Rotations.hpp`

| Function | Signature | Notes |
|----------|-----------|-------|
| `rotation_matrix_from_euler_angles` | `auto rotation_matrix_from_euler_angles(roll, pitch, yaw)` | Standard yaw-pitch-roll sequence |
| `is_valid_rotation_matrix` | `bool is_valid_rotation_matrix(A, tol)` | Check det=1 and orthogonality |

---

### Component 8: Discrete Integration (`IntegrateDiscrete.hpp`) - **P2**

**Source Reference**: [integrate_discrete.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/integrate_discrete.py)

#### [NEW] `include/metis/math/IntegrateDiscrete.hpp`

```cpp
/**
 * @brief Integrates discrete samples using reconstruction methods
 * 
 * @param f Function values
 * @param x Grid points (optional, defaults to indices)
 * @param multiply_by_dx If true, returns interval integrals; if false, returns average values
 * @param method "forward_euler", "backward_euler", "trapezoidal", "forward_simpson", "backward_simpson", "cubic"
 * @param method_endpoints "lower_order", "ignore", or "periodic"
 */
template <typename DerivedF, typename DerivedX>
auto integrate_discrete_intervals(
    const Eigen::MatrixBase<DerivedF>& f,
    const Eigen::MatrixBase<DerivedX>& x,
    bool multiply_by_dx = true,
    const std::string& method = "trapezoidal",
    const std::string& method_endpoints = "lower_order"
);
```

---

### Component 9: ODE Integration (`Integrate.hpp`) - **P3**

**Source Reference**: [integrate.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/integrate.py)

> [!CAUTION]
> This is the most complex component. Consider deferring to Phase 4 if timeline is tight.

#### [NEW] `include/metis/math/Integrate.hpp`

| Function | Numeric Backend | Symbolic Backend |
|----------|-----------------|------------------|
| `quad` | Boost.Math or custom | `casadi::integrator` with CVODES |
| `solve_ivp` | Eigen-based RK45 or external lib | `casadi::integrator` |

The symbolic backend wraps CasADi's CVODES integrator to enable differentiation through ODEs.

---

## Task Breakdown & Milestones

### Milestone 1: Logic Extensions (Week 1)
- [x] **Task 1.1**: Extend `Logic.hpp` with `logical_and`, `logical_or`, `logical_not`
- [x] **Task 1.2**: Add `all`, `any` to `Logic.hpp`
- [x] **Task 1.3**: Verify `clip` is working (may already exist as `clamp`)
- [ ] **Task 1.4**: Add `metis::roll` free function to `Linalg.hpp` (if needed)
- [x] **Task 1.5**: Write tests for extended `Logic.hpp`

### Milestone 2: Calculus & Surrogate Models (Week 2)
- [x] **Task 2.1**: Create `Calculus.hpp` with full `gradient` implementation
- [x] **Task 2.2**: Create `SurrogateModel.hpp` with `softmax`, `softmin`, `softplus`
- [x] **Task 2.3**: Add `sigmoid`, `swish`, `blend` to `SurrogateModel.hpp`
- [x] **Task 2.4**: Extend `Linalg.hpp` with `pinv`, `inner`, extended `norm`
- [x] **Task 2.5**: Write tests for Calculus and SurrogateModel modules (Surrogate done)

### Milestone 3: Advanced Numerics (Week 3)
- [x] **Task 3.1**: Create `FiniteDifference.hpp`
- [x] **Task 3.2**: Extend `Rotations.hpp` with euler angles and validation
- [x] **Task 3.3**: Create `IntegrateDiscrete.hpp` (core methods)
- [x] **Task 3.4**: Add Simpson's and cubic methods to `IntegrateDiscrete.hpp`
- [x] **Task 3.5**: Write comprehensive tests for all new modules

### Milestone 4: ODE Integration (Week 4 / Phase 4)
- [ ] **Task 4.1**: Create `Integrate.hpp` with `quad` (numeric only first)
- [ ] **Task 4.2**: Add symbolic `quad` with CasADi CVODES
- [ ] **Task 4.3**: Implement `solve_ivp` numeric backend
- [ ] **Task 4.4**: Add symbolic `solve_ivp` with CasADi
- [ ] **Task 4.5**: Integration tests with known ODE solutions

### Milestone 5: Polish & Documentation
- [ ] **Task 5.1**: Update `MetisMath.hpp` to include all new headers
- [ ] **Task 5.2**: Run full test suite and fix any issues
- [ ] **Task 5.3**: Update `docs/design_overview.md` with Phase 3 summary
- [ ] **Task 5.4**: Create example showing AeroSandbox model port

---

## Verification Plan

### Automated Tests

All tests are in `tests/math/` and run via:

```bash
# Build and run all tests
./scripts/ci.sh

# Or run specific test file
cd build && ctest -R test_array --output-on-failure
```

**Existing Test Structure** (from [tests/math/](file:///home/tanged/sources/metis/tests/math)):
- `test_arithmetic.cpp` - Arithmetic operations
- `test_diffops.cpp` - Differential operators
- `test_geometry.cpp` - Geometry utilities
- `test_interpolate.cpp` - Interpolation
- `test_linalg.cpp` - Linear algebra
- `test_logic.cpp` - Logic and branching
- `test_trig.cpp` - Trigonometry

**New Test Files to Create**:
| Test File | Coverage |
|-----------|----------|
| `test_calculus.cpp` | Full `gradient` with edge_order, n-th derivative |
| `test_surrogate.cpp` | `softmax`, `softmin`, `softplus`, `sigmoid`, `swish`, `blend` |
| `test_finite_diff.cpp` | `finite_difference_coefficients` |
| `test_integrate_discrete.cpp` | `integrate_discrete_intervals` with all methods |
| `test_integrate.cpp` | `quad`, `solve_ivp` (when implemented) |

**Dual-Backend Testing Pattern**:
```cpp
template <typename Scalar>
void test_function() {
    if constexpr (std::is_same_v<Scalar, double>) {
        // Numeric: Check exact values
        EXPECT_DOUBLE_EQ(result, expected);
    } else {
        // Symbolic: Evaluate and check
        auto eval = metis::eval_scalar(result, {}, {});
        EXPECT_NEAR(eval, expected, 1e-9);
    }
}

TEST(ModuleTests, Numeric) { test_function<double>(); }
TEST(ModuleTests, Symbolic) { test_function<casadi::MX>(); }
```

### Full CI Pipeline

```bash
# Complete verification (build + test + examples)
./scripts/verify.sh
```

Logs are saved to `logs/verify.log`.

---

## Dependencies & Prerequisites

- **Eigen ≥ 3.4**: Required for `.reshaped()` in `Array.hpp`
- **CasADi ≥ 3.6**: Required for CVODES in `Integrate.hpp`
- **C++20**: Required for concepts and `if constexpr`

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Eigen 3.4 `reshaped()` not available | Medium | Fallback to `Map` with explicit strides |
| CasADi CVODES complexity | High | Defer `Integrate.hpp` to Phase 4 if needed |
| Template explosion in compile times | Medium | Use explicit instantiations for common types |
| Breaking existing API | High | Add new functions, don't modify existing signatures |

---

## Success Criteria

1. ✅ All P0/P1 functions implemented with dual-backend support
2. ✅ All tests pass for both numeric and symbolic modes
3. ✅ CI pipeline (`./scripts/verify.sh`) passes
4. ✅ At least one AeroSandbox example model ported as demo
5. ✅ Documentation updated with Phase 3 summary

---

## Appendix: Reference Module Mapping

| AeroSandbox Python Module | Metis C++ Header | Status |
|--------------------------|------------------|--------|
| `__init__.py` | `metis.hpp` | ✅ Complete |
| `arithmetic_monadic.py` | `Arithmetic.hpp` | ✅ Complete |
| `arithmetic_dyadic.py` | `Arithmetic.hpp` | ⏳ Need `mod`, `centered_mod` |
| `array.py` | *(Eigen native)* | ✅ Use Eigen APIs directly |
| `calculus.py` | `Calculus.hpp` + `DiffOps.hpp` | ⏳ Need full `gradient` |
| `conditionals.py` | `Logic.hpp` | ✅ Complete |
| `finite_difference_operators.py` | `FiniteDifference.hpp` | ❌ New |
| `integrate.py` | `Integrate.hpp` | ❌ New (P3) |
| `integrate_discrete.py` | `IntegrateDiscrete.hpp` | ❌ New |
| `interpolate.py` | `Interpolate.hpp` | ✅ Complete |
| `linalg.py` | `Linalg.hpp` | ⏳ Need `pinv`, `inner` |
| `linalg_top_level.py` | `Linalg.hpp` | ✅ Complete |
| `logicals.py` | `Logic.hpp` | ⏳ Need `logical_*`, `all`, `any` |
| `rotations.py` | `Rotations.hpp` | ⏳ Need euler, validation |
| `spacing.py` | `Spacing.hpp` | ✅ Complete |
| `surrogate_model_tools.py` | `SurrogateModel.hpp` | ❌ New |
| `trig.py` | `Trig.hpp` | ✅ Complete |
| `determine_type.py` | `MetisConcepts.hpp` | ✅ Complete |

---

*Generated by Metis Dev Team - Phase 3 Planning*
