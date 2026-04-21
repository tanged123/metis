# Metis Phase 4: Beta 1.0 Release Implementation Plan

**Goal**: Polish Metis for beta 1.0 release with ODE integration, expanded spacing functions, quaternion math, 100% code coverage, and graph visualization.
**Status**: Planning Draft
**Created**: 2025-12-14

---

## Executive Summary

Phase 4 completes the remaining Milestones from Phase 3 and adds key features for a polished beta release. This phase focuses on:

1. **ODE Integration** (deferred from Phase 3) — `quad`, `solve_ivp` with CasADi CVODES
2. **Spacing Extensions** — `sinspace`, `logspace`, `geomspace`
3. **Quaternion Math** — Native quaternion operations for rotation representations
4. **100% Code Coverage** — Expand test suite to cover all code paths
5. **Graph Visualization** — User-friendly tools to visualize CasADi computational graphs

> [!IMPORTANT]
> This phase marks the **beta 1.0 milestone**. Future phases will focus on optimization solvers, sparse matrices, and advanced interpolation.

---

## Gap Analysis: Phase 3 → Phase 4

### ✅ Completed in Phase 3

| Module | Status | Functions |
|--------|--------|-----------|
| `Logic.hpp` | ✅ Complete | `logical_and`, `logical_or`, `logical_not`, `all`, `any`, `clip` |
| `Calculus.hpp` | ✅ Complete | Full `gradient` with edge_order, n-th derivative |
| `SurrogateModel.hpp` | ✅ Complete | `softmax`, `softmin`, `softplus`, `sigmoid`, `swish`, `blend` |
| `FiniteDifference.hpp` | ✅ Complete | `finite_difference_coefficients` |
| `IntegrateDiscrete.hpp` | ✅ Complete | All methods including Simpson's and cubic |
| `Rotations.hpp` | ✅ Complete | `rotation_matrix_from_euler_angles`, `is_valid_rotation_matrix` |
| `Linalg.hpp` | ✅ Complete | `pinv`, `inner`, extended `norm` |

### ❌ Deferred/Missing (Phase 4 Scope)

| Priority | Module | Missing Functions | Notes |
|----------|--------|-------------------|-------|
| **P0** | `Integrate.hpp` | `quad`, `solve_ivp` | Deferred from Phase 3 Milestone 4 |
| **P0** | `Spacing.hpp` | `sinspace`, `logspace`, `geomspace` | User-requested, matches AeroSandbox |
| **P1** | `Quaternion.hpp` | Quaternion algebra | User-requested, common in robotics/aerospace |
| **P1** | Testing | Coverage gaps | Target 100% line coverage |
| **P2** | `MetisIO.hpp` | Graph visualization | User-requested for UX polish |

---

## Proposed Implementation Structure

```
include/metis/math/
├── Integrate.hpp       # [NEW] ODE integration (quad, solve_ivp)
├── Spacing.hpp         # [EXTEND] Add sinspace, logspace, geomspace
├── MetisMath.hpp       # [EXTEND] Include new headers
└── ...

include/metis/core/
├── Quaternion.hpp      # [NEW] Quaternion math operations
├── MetisIO.hpp         # [EXTEND] Add graph visualization utilities
└── ...

tests/math/
├── test_integrate.cpp        # [NEW] ODE integration tests
├── test_spacing.cpp          # [NEW] Extended spacing tests
└── ...

tests/core/
├── test_quaternion.cpp       # [NEW] Quaternion tests
└── ...
```

---

## Detailed Implementation Specifications

### Component 1: ODE Integration (`Integrate.hpp`) — **P0**

**Source Reference**: [integrate.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/integrate.py)

#### [NEW] `include/metis/math/Integrate.hpp`

| Function | Signature | Numeric Backend | Symbolic Backend |
|----------|-----------|-----------------|------------------|
| `quad` | `auto quad(func, a, b, ...)` | Custom adaptive quadrature or Boost integration | `casadi::integrator` with CVODES |
| `solve_ivp` | `auto solve_ivp(fun, t_span, y0, ...)` | Eigen-based RK45 | `casadi::integrator` with CVODES |

**Key Implementation Details**:

```cpp
/**
 * @brief Integrates a function over interval [a, b]
 * 
 * @param func Function to integrate (callable or symbolic expression)
 * @param a Lower bound
 * @param b Upper bound
 * @param variable_of_integration For symbolic: which variable to integrate over
 * @return Integral value and estimated error
 */
template <typename Func, typename T>
auto quad(Func&& func, T a, T b, ...);

/**
 * @brief Solves initial value problem for system of ODEs
 * 
 * @param fun Right-hand side dy/dt = fun(t, y)
 * @param t_span Integration interval (t0, tf)
 * @param y0 Initial state
 * @param t_eval Times at which to store solution (optional)
 * @return OdeResult with t, y arrays and solver info
 */
template <typename Func, typename Scalar>
OdeResult<Scalar> solve_ivp(Func&& fun, std::pair<Scalar, Scalar> t_span, 
                             const metis::MetisMatrix<Scalar>& y0, ...);
```

> [!CAUTION]
> The symbolic backend requires careful handling of CasADi's `integrator` interface. Time normalization and parameter extraction must match [integrate.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/integrate.py) exactly.

---

### Component 2: Extended Spacing (`Spacing.hpp`) — **P0**

**Source Reference**: [spacing.py](file:///home/tanged/sources/metis/reference/aerosandbox_numpy_reference/spacing.py)

#### [MODIFY] `include/metis/math/Spacing.hpp`

Add the following functions:

| Function | Purpose | Formula |
|----------|---------|---------|
| `sinspace` | Sine-spaced vector (denser at start) | `start + (stop - start) * (1 - cos(π*i/(2*(n-1))))` |
| `logspace` | Log-spaced vector | `10^linspace(start, stop, n)` |
| `geomspace` | Geometric progression | Endpoints directly specified |

```cpp
/**
 * @brief Generates sine-spaced vector (denser at start by default)
 * 
 * @param start Start value
 * @param stop End value
 * @param n Number of points
 * @param reverse_spacing If true, bunch near stop instead of start
 * @return Vector of n points
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> sinspace(
    const T& start, const T& stop, int n, bool reverse_spacing = false);

/**
 * @brief Returns numbers spaced evenly on a log scale
 * 
 * @param start Start exponent (result starts at 10^start)
 * @param stop End exponent (result ends at 10^stop)
 * @param n Number of points
 * @return Vector of n points
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> logspace(const T& start, const T& stop, int n);

/**
 * @brief Returns numbers spaced evenly on a log scale (endpoints specified directly)
 * 
 * @param start Start value (must be positive)
 * @param stop End value (must be positive)
 * @param n Number of points
 * @return Vector of n points
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> geomspace(const T& start, const T& stop, int n);
```

---

### Component 3: Quaternion Math (`Quaternion.hpp`) — **P1**

#### [NEW] `include/metis/core/Quaternion.hpp`

Quaternions are essential for efficient rotation representation in robotics and aerospace. We implement a minimal but complete quaternion algebra.

| Function | Purpose |
|----------|---------|
| `Quaternion<T>` | Quaternion class with (w, x, y, z) components |
| `quat_multiply` | Hamilton product of two quaternions |
| `quat_conjugate` | Quaternion conjugate (inverse for unit quaternions) |
| `quat_normalize` | Normalize to unit quaternion |
| `quat_from_euler` | Create quaternion from Euler angles |
| `quat_to_rotation_matrix` | Convert quaternion to 3x3 rotation matrix |
| `quat_rotate_vector` | Rotate a 3D vector by a quaternion |
| `quat_slerp` | Spherical linear interpolation |

```cpp
/**
 * @brief Quaternion class for rotation representation
 * 
 * Stores quaternion in (w, x, y, z) convention where w is the scalar part.
 * All operations support both numeric and symbolic types.
 */
template <typename Scalar>
class Quaternion {
public:
    Scalar w, x, y, z;  // w is scalar part
    
    // Constructors
    Quaternion();  // Identity quaternion (1, 0, 0, 0)
    Quaternion(Scalar w, Scalar x, Scalar y, Scalar z);
    
    // Operations
    Quaternion operator*(const Quaternion& other) const;  // Hamilton product
    Quaternion conjugate() const;
    Quaternion normalized() const;
    Scalar norm() const;
    
    // Conversions
    Eigen::Matrix<Scalar, 3, 3> to_rotation_matrix() const;
    static Quaternion from_euler(Scalar roll, Scalar pitch, Scalar yaw);
    static Quaternion from_axis_angle(const Eigen::Matrix<Scalar, 3, 1>& axis, Scalar angle);
    
    // Vector rotation
    Eigen::Matrix<Scalar, 3, 1> rotate(const Eigen::Matrix<Scalar, 3, 1>& v) const;
};

// Free functions
template <typename Scalar>
Quaternion<Scalar> slerp(const Quaternion<Scalar>& q0, const Quaternion<Scalar>& q1, Scalar t);
```

---

### Component 4: 100% Code Coverage — **P1**

**Current State**: Code coverage is enabled via `./scripts/coverage.sh` and Codecov integration.

**Strategy**: Identify uncovered lines and add targeted tests.

#### Step 4.1: Generate Coverage Report

```bash
./scripts/coverage.sh
# Open build/coverage/html/index.html to view report
```

#### Step 4.2: Identify Gaps

Review coverage report for:
- Uncovered branches in `where()` / `where_matrix()`
- Edge cases in interpolation
- Error handling paths
- Symbolic-specific code paths

#### Step 4.3: Add Targeted Tests

| Test File | Coverage Target |
|-----------|-----------------|
| `test_logic.cpp` | Edge cases for `where`, `all`, `any` |
| `test_arithmetic.cpp` | Edge cases: `fmod` with negative, `centered_mod` |
| `test_linalg.cpp` | Singular matrix handling, `pinv` accuracy |
| `test_interpolate.cpp` | Boundary extrapolation, single-point edge case |
| `test_rotations.cpp` | Symbolic validation path |

---

### Component 5: Graph Visualization (`MetisIO.hpp`) — **P2**

**Goal**: Allow users to visualize the CasADi computational graph for debugging and understanding.

#### [EXTEND] `include/metis/core/MetisIO.hpp`

CasADi provides built-in graph export via DOT format. We wrap this in a user-friendly API.

```cpp
namespace metis {

/**
 * @brief Exports a symbolic expression graph to DOT format
 * 
 * @param expr The symbolic expression to visualize
 * @param filename Output filename (without extension, creates .dot and .pdf)
 * @param name Optional graph name
 */
void export_graph_dot(const SymbolicScalar& expr, const std::string& filename,
                      const std::string& name = "graph");

/**
 * @brief Exports a metis::Function's graph to DOT format
 * 
 * @param func The function to visualize
 * @param filename Output filename
 */
void export_graph_dot(const Function& func, const std::string& filename);

/**
 * @brief Renders DOT to PDF/PNG using Graphviz (requires graphviz in PATH)
 * 
 * @param dot_file Input DOT file
 * @param output_file Output image file (.pdf, .png, .svg)
 */
void render_graph(const std::string& dot_file, const std::string& output_file);

/**
 * @brief Convenience: export + render in one call
 */
void visualize_graph(const SymbolicScalar& expr, const std::string& output_base);
void visualize_graph(const Function& func, const std::string& output_base);

} // namespace metis
```

**Implementation Notes**:
- Use `casadi::MX::print_graph()` or `casadi::Function::save()` for DOT export
- Call `dot -Tpdf input.dot -o output.pdf` via `std::system()` for rendering
- Graphviz is already in the Nix flake dependencies

---

## Task Breakdown & Milestones

### Milestone 1: ODE Integration (Week 1) ✅ COMPLETE
- [x] **Task 1.1**: Create `Integrate.hpp` with `quad` (numeric backend - Gauss-Kronrod G7K15)
- [x] **Task 1.2**: Add `quad` symbolic backend with CasADi CVODES
- [x] **Task 1.3**: Implement `solve_ivp` numeric backend (RK4 with substeps)
- [x] **Task 1.4**: Add `solve_ivp` symbolic backend with CasADi (`solve_ivp_symbolic`, `solve_ivp_expr`)
- [x] **Task 1.5**: Create `OdeResult` and `QuadResult` structs for return values
- [x] **Task 1.6**: Write comprehensive tests (`test_integrate.cpp` - 22 tests passing)

### Milestone 2: Extended Spacing (Week 1)
- [x] **Task 2.1**: Add `sinspace` to `Spacing.hpp`
- [x] **Task 2.2**: Add `logspace` to `Spacing.hpp`
- [x] **Task 2.3**: Add `geomspace` to `Spacing.hpp`
- [x] **Task 2.4**: Write tests in `test_spacing.cpp`

### Milestone 3: Quaternion Math (Week 2)
- [x] **Task 3.1**: Create `Quaternion.hpp` with `Quaternion<T>` class
- [x] **Task 3.2**: Implement Hamilton product, conjugate, and `inverse`
- [x] **Task 3.3**: Implement `from_euler`, `from_axis_angle`, `from_rotation_vector`
- [x] **Task 3.4**: Implement `to_rotation_matrix`, `from_rotation_matrix`, and `rotate`
- [x] **Task 3.5**: Implement `slerp` for interpolation
- [x] **Task 3.6**: Test against `rotation_matrix_from_euler_angles` for consistency (round-trip)
- [x] **Task 3.7**: Write comprehensive tests (`test_quaternion.cpp`)
- [x] **Task 3.8**: Implement `to_euler` for conversion recovery

### Milestone 4: Code Coverage (Week 2)
- [x] **Task 4.1**: Generate and analyze coverage report
- [x] **Task 4.2**: Add tests for uncovered Logic.hpp paths
- [x] **Task 4.3**: Add tests for uncovered Arithmetic.hpp paths
- [x] **Task 4.4**: Add tests for uncovered Linalg.hpp paths
- [x] **Task 4.5**: Add tests for uncovered Interpolate.hpp paths
- [x] **Task 4.6**: Verify 100% line coverage (or document intentional gaps)

### Milestone 5: Graph Visualization (Week 3) ✅ COMPLETE
- [x] **Task 5.1**: Extend `MetisIO.hpp` with `export_graph_dot`
- [x] **Task 5.2**: Implement `render_graph` (Graphviz wrapper)
- [x] **Task 5.3**: Add `visualize_graph` convenience function
- [x] **Task 5.4**: Create example demonstrating graph visualization
- [x] **Task 5.5**: Write tests for graph visualization (6 tests)


### Milestone 6: Beta 1.0 Polish (Week 3)
- [ ] **Task 6.1**: Update `MetisMath.hpp` to include all new headers
- [ ] **Task 6.2**: Run full test suite and fix any issues
- [ ] **Task 6.3**: Update `docs/design_overview.md` with Phase 4 summary
- [ ] **Task 6.4**: Update README with beta 1.0 feature summary
- [ ] **Task 6.5**: Create release notes for beta 1.0
- [ ] **Task 6.6**: Final verification with `./scripts/verify.sh`

---

## Verification Plan

### Automated Tests

All tests run via the existing infrastructure:

```bash
# Build and run all tests
./scripts/ci.sh

# Run specific test file
cd build && ctest -R test_integrate --output-on-failure
cd build && ctest -R test_spacing --output-on-failure
cd build && ctest -R test_quaternion --output-on-failure
```

### New Test Files

| Test File | Coverage |
|-----------|----------|
| `test_integrate.cpp` | `quad` accuracy, `solve_ivp` vs known ODE solutions |
| `test_spacing.cpp` | `sinspace`, `logspace`, `geomspace` values and endpoints |
| `test_quaternion.cpp` | All quaternion operations, consistency with rotation matrices |

### Dual-Backend Testing Pattern

All new tests follow the established pattern:

```cpp
template <typename Scalar>
void test_function() {
    // Test implementation
    if constexpr (std::is_floating_point_v<Scalar>) {
        // Numeric: Check exact values
        EXPECT_NEAR(result, expected, 1e-10);
    } else {
        // Symbolic: Evaluate and check
        auto eval = metis::eval_scalar(result, {symbolic_var}, {numeric_value});
        EXPECT_NEAR(eval, expected, 1e-9);
    }
}

TEST(ModuleTests, Numeric) { test_function<double>(); }
TEST(ModuleTests, Symbolic) { test_function<casadi::MX>(); }
```

### Integration Tests

| Test | Description |
|------|-------------|
| ODE Integration | Solve exponential decay, compare to analytic solution |
| ODE Integration | Solve Lotka-Volterra system, verify conservation properties |
| Quaternion | Round-trip: Euler → Quaternion → Rotation Matrix → Euler |
| Graph Viz | Export example graph, verify DOT file is valid |

### Code Coverage Verification

```bash
# Generate coverage report
./scripts/coverage.sh

# View HTML report
open build/coverage/html/index.html

# Check coverage percentage in CI via Codecov badge
```

**Target**: 100% line coverage (or documented intentional exclusions)

### Manual Verification

1. **Graph Visualization**: Generate graph for `examples/energy_intro.cpp`, visually inspect PDF output
2. **ODE Integration**: Run Lotka-Volterra example, plot results
3. **Quaternion**: Create rotating cube visualization (optional advanced demo)

---

## Dependencies & Prerequisites

- **Existing**: Eigen ≥ 3.4, CasADi ≥ 3.6, C++20
- **Graph Viz**: Graphviz (already in Nix flake)
- **ODE Integration**: CVODES via CasADi (already linked)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| CasADi CVODES complexity | High | Follow reference implementation closely; add extensive tests |
| Quaternion gimbal lock edge cases | Medium | Use quaternion-native operations; validate vs. rotation matrices |
| Coverage tool inconsistencies | Low | Use lcov consistently; document exclusions |
| Graph rendering depends on system Graphviz | Low | Graceful failure with helpful error message |

---

## Success Criteria

1. ✅ All P0/P1 functions implemented with dual-backend support
2. ✅ All tests pass for both numeric and symbolic modes
3. ✅ CI pipeline (`./scripts/verify.sh`) passes
4. ✅ Code coverage ≥ 95% (target 100%)
5. ✅ Graph visualization working with example output
6. ✅ Documentation updated with Phase 4 summary
7. ✅ README updated with beta 1.0 announcement

---

## Future Phases (Post-Beta)

| Phase | Focus | Key Features |
|-------|-------|--------------|
| Phase 5 | Optimization Solvers | IPOPT/SNOPT integration, NLP problem definition |
| Phase 6 | Advanced Interpolation | Multi-dimensional gridded interpolation, sparse table handling |
| Phase 7 | Sparse Matrices | Sparse Eigen types, CasADi sparse operations |
| Phase 8 | Engineering Applications | Full aircraft simulation, trajectory optimization examples |

---

*Generated by Metis Dev Team - Phase 4 Planning*
