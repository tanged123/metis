# Metis Phase 6: Optimization Framework Implementation Plan

**Goal**: Implement a high-level optimization interface (`metis::Opti`) that wraps CasADi's IPOPT backend, mirroring AeroSandbox's `asb.Opti` API in C++.

**Status**: Planning Draft
**Created**: 2025-12-16

---

## Executive Summary

Phase 6 introduces the **Optimization Engines & Strategies** layer to Metis, enabling users to formulate and solve nonlinear optimization problems using a clean, Metis-native C++ API. This builds upon the existing `metis::Function` infrastructure and symbolic tracing capabilities.

Key deliverables:
1. **`metis::Opti`** — Optimization environment class for variable, parameter, and constraint management
2. **`metis::OptiSol`** — Solution wrapper for extracting optimized values
3. **Derivative Helpers** — `derivative_of()` and `constrain_derivative()` for trajectory optimization
4. **NaN Propagation Sparsity** — Research feasibility (may defer to Phase 7)

> [!IMPORTANT]
> This phase unlocks trajectory optimization capabilities. The Brachistochrone example ([brachistochrone.cpp](file:///home/tanged/sources/metis/examples/brachistochrone.cpp)) currently demonstrates the ODE, but optimal control requires the optimization layer.

---

## Reference Analysis: AeroSandbox Opti

Source: [opti.py](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/opti.py)

### Key API Methods (Python → C++ Mapping)

| AeroSandbox Method | Metis Equivalent | Notes |
|--------------------|-----------------|-------|
| `opti.variable(init_guess, scale, bounds, freeze)` | `opti.variable(init, scale, lower, upper)` | Simplified, no freeze/category |
| `opti.parameter(value)` | `opti.parameter(value)` | Fixed values in optimization |
| `opti.subject_to(constraint)` | `opti.subject_to(constraint)` | Equality/inequality via CasADi |
| `opti.minimize(f)` / `maximize(f)` | Same | Objective setting |
| `opti.solve(options)` | `opti.solve(options)` | Returns `OptiSol` |
| `opti.derivative_of(var, wrt, init)` | `opti.derivative_of(var, wrt, init)` | Implicit derivative variable |
| `sol(x)` | `sol.value(x)` | Extract optimized value |

### Key Design Observations

1. **CasADi `cas.Opti` Inheritance**: AeroSandbox extends `casadi.Opti`. In C++, we will **wrap** `casadi::Opti` (composition over inheritance).
2. **Scaling**: Variables are internally scaled for numerical stability.
3. **IPOPT Configuration**: Default options like `mu_strategy=adaptive`, `fast_step_computation=yes` are applied.
4. **Constraint Tracking**: Declarations tracked for debugging (we may simplify this for v1).

---

## User Review Required

> [!IMPORTANT]
> **Scope Decision**: The AeroSandbox `Opti` class has many features (caching, freeze/categories, solve_sweep). For Metis v1, I propose a **minimal core API**. Please confirm this scope:
>
> **Included in Phase 6:**
> - `variable()`, `parameter()`, `subject_to()`, `minimize()`/`maximize()`, `solve()`
> - `derivative_of()`, `constrain_derivative()` (essential for trajectory optimization)
> - `OptiSol` with `value()` extraction
> - Basic IPOPT options (max_iter, verbose, tolerance)
>
> **Deferred to future phases:**
> - Variable freezing/categories
> - Solution caching to JSON
> - `solve_sweep()` for parametric studies
> - NaN propagation sparsity analysis (Phase 7 candidate)

> [!WARNING]
> **Dependency Consideration**: IPOPT is bundled with CasADi (already in `flake.nix`). No new dependencies required.

---

## Proposed Implementation Structure

```
include/metis/optimization/
├── Opti.hpp           # [NEW] Main optimization class
├── OptiSol.hpp        # [NEW] Solution wrapper
└── OptiOptions.hpp    # [NEW] Solver configuration

include/metis/
├── metis.hpp          # [EXTEND] Include optimization headers

tests/optimization/
├── test_opti.cpp      # [NEW] Core optimization tests

examples/
├── optimization_intro.cpp     # [NEW] Basic NLP example
├── rocket_trajectory.cpp      # [NEW] Trajectory optimal control

docs/implementation_plans/
├── phase6_optimization.md     # [NEW] This document
```

---

## Detailed Implementation Specifications

### Component 1: `metis::Opti` Class — **P0**

#### [NEW] `include/metis/optimization/Opti.hpp`

```cpp
#pragma once

#include "metis/core/MetisTypes.hpp"
#include <casadi/casadi.hpp>
#include <optional>
#include <string>

namespace metis {

// Forward declaration
class OptiSol;

/**
 * @brief Options for solving optimization problems
 */
struct OptiOptions {
    int max_iter = 1000;           // Maximum iterations
    double max_cpu_time = 1e20;    // Maximum solve time [seconds]
    double abstol = 1e-8;          // Constraint tolerance
    bool verbose = true;           // Print IPOPT progress
    bool jit = false;              // JIT compile expressions (experimental)
};

/**
 * @brief Main optimization environment class
 *
 * Wraps CasADi's Opti interface to provide Metis-native types
 * and a clean C++ API for nonlinear programming.
 *
 * Example:
 *   metis::Opti opti;
 *   auto x = opti.variable(0.0);  // scalar, init_guess=0
 *   auto y = opti.variable(0.0);
 *   opti.minimize((1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x));
 *   auto sol = opti.solve();
 *   double x_opt = sol.value(x);  // ~1.0
 */
class Opti {
  public:
    Opti();
    ~Opti() = default;

    // ---- Decision Variables ----
    
    /**
     * @brief Create a scalar decision variable
     * @param init_guess Initial guess value
     * @param scale Optional scale for numerical conditioning
     * @param lower_bound Optional lower bound
     * @param upper_bound Optional upper bound
     */
    SymbolicScalar variable(double init_guess = 0.0,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt);

    /**
     * @brief Create a vector of decision variables
     * @param n_vars Number of variables
     * @param init_guess Initial guess (scalar applied to all, or vector)
     */
    SymbolicVector variable(int n_vars,
                            double init_guess = 0.0,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt);

    SymbolicVector variable(const Eigen::VectorXd& init_guess,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt);

    // ---- Parameters ----
    
    /**
     * @brief Create a parameter (fixed value during optimization)
     * @param value Parameter value
     */
    SymbolicScalar parameter(double value);
    SymbolicVector parameter(const Eigen::VectorXd& value);

    // ---- Constraints ----
    
    /**
     * @brief Add constraint(s)
     * @param constraint Symbolic inequality/equality (e.g., x >= 0, x == 1)
     */
    void subject_to(const SymbolicScalar& constraint);
    void subject_to(const std::vector<SymbolicScalar>& constraints);

    // ---- Objective ----
    
    void minimize(const SymbolicScalar& objective);
    void maximize(const SymbolicScalar& objective);

    // ---- Solve ----
    
    OptiSol solve(const OptiOptions& options = {});

    // ---- Derivative Helpers (for trajectory optimization) ----
    
    /**
     * @brief Create a derivative variable constrained by integration
     *
     * Returns a new variable that is constrained to be the derivative of
     * `variable` with respect to `with_respect_to`.
     *
     * @param variable The quantity to differentiate
     * @param with_respect_to Independent variable (e.g., time array)
     * @param derivative_init_guess Initial guess for derivative values
     * @param method Integration method: "trapezoidal", "forward_euler", "backward_euler"
     */
    SymbolicVector derivative_of(const SymbolicVector& variable,
                                 const Eigen::VectorXd& with_respect_to,
                                 double derivative_init_guess,
                                 const std::string& method = "trapezoidal");

    /**
     * @brief Constrain an existing variable to be a derivative
     *
     * Adds constraints: d(variable)/d(with_respect_to) == derivative
     */
    void constrain_derivative(const SymbolicVector& derivative,
                              const SymbolicVector& variable,
                              const Eigen::VectorXd& with_respect_to,
                              const std::string& method = "trapezoidal");

    // ---- Access ----
    
    const casadi::Opti& casadi_opti() const { return opti_; }

  private:
    casadi::Opti opti_;
    
    // Internal scaling state
    struct VarInfo {
        casadi::MX raw_var;  // Scaled internal variable
        double scale;
    };
    std::vector<VarInfo> variables_;
};

} // namespace metis
```

**Implementation Notes**:
- `variable()` internally creates scaled variable: `var = scale * opti_.variable(n)`
- `set_initial()` called with `init_guess`
- Bounds applied via `subject_to(var >= lb)` normalized by scale
- Store mapping for later solution extraction

---

### Component 2: `metis::OptiSol` Class — **P0**

#### [NEW] `include/metis/optimization/OptiSol.hpp`

```cpp
#pragma once

#include "metis/core/MetisTypes.hpp"
#include <casadi/casadi.hpp>

namespace metis {

/**
 * @brief Solution wrapper for optimization results
 *
 * Provides type-safe extraction of optimized values.
 */
class OptiSol {
  public:
    OptiSol(casadi::OptiSol cas_sol);

    /**
     * @brief Extract scalar value at optimum
     */
    double value(const SymbolicScalar& var) const;

    /**
     * @brief Extract vector value at optimum
     */
    Eigen::VectorXd value(const SymbolicVector& var) const;

    /**
     * @brief Get solver statistics
     */
    casadi::Dict stats() const;

    /**
     * @brief Check if solve converged
     */
    bool converged() const;

  private:
    casadi::OptiSol cas_sol_;
};

} // namespace metis
```

---

### Component 3: Derivative Helpers — **P1**

The `derivative_of()` and `constrain_derivative()` methods are critical for trajectory optimization (Dymos-style collocation).

**Integration Methods**:

| Method | Order | Formula |
|--------|-------|---------|
| `forward_euler` | 1st | `x[i+1] - x[i] == xdot[i] * dt[i]` |
| `backward_euler` | 1st | `x[i+1] - x[i] == xdot[i+1] * dt[i]` |
| `trapezoidal` | 2nd | `x[i+1] - x[i] == 0.5 * (xdot[i] + xdot[i+1]) * dt[i]` |

**Example Usage** (from AeroSandbox test):
```cpp
metis::Opti opti;

Eigen::VectorXd time = metis::linspace(0.0, 1.0, 100);
auto position = opti.variable(100, 0.0);  // N-element vector
auto velocity = opti.derivative_of(position, time, 0.0);
auto accel = opti.derivative_of(velocity, time, 0.0);

// Boundary conditions
opti.subject_to(position(0) == 0);
opti.subject_to(velocity(0) == 0);
opti.subject_to(position(99) == 10);

opti.minimize(accel.cwiseAbs().maxCoeff());  // Minimize peak acceleration
auto sol = opti.solve();
```

---

### Component 4: NaN Propagation Sparsity — **P2 (Research)**

> [!NOTE]
> AeroSandbox's design mentions "NaN Propagation Sparsity" for efficient Jacobian computation. This is an advanced technique where NaN values propagate through computations to identify sparsity patterns.
>
> **Recommendation**: Defer to Phase 7. CasADi already has excellent sparsity detection via symbolic differentiation. This is primarily a Python/NumPy concern for hybrid operations.

---

## Task Breakdown & Milestones

### Milestone 1: Core Opti Class (3-4 days)

- [ ] Create directory structure `include/metis/optimization/`
- [ ] Implement `Opti.hpp` with `variable()`, `parameter()`, `subject_to()`, `minimize()`
- [ ] Implement `OptiSol.hpp` with `value()` extraction
- [ ] Add `OptiOptions.hpp` for solver configuration
- [ ] Update `metis.hpp` to include optimization headers
- [ ] Update `CMakeLists.txt` to include new headers

### Milestone 2: Derivative Helpers (2-3 days)

- [ ] Implement `derivative_of()` with trapezoidal method
- [ ] Implement `constrain_derivative()` with method selection
- [ ] Add forward/backward Euler methods

### Milestone 3: Tests (2-3 days)

- [ ] Create `tests/optimization/test_opti.cpp`
- [ ] Rosenbrock 2D unconstrained
- [ ] Rosenbrock 2D constrained (unit circle)
- [ ] N-dimensional Rosenbrock
- [ ] Rocket trajectory control (from AeroSandbox reference)
- [ ] Update `tests/CMakeLists.txt`

### Milestone 4: Examples & Documentation (1-2 days)

- [ ] Create `examples/optimization_intro.cpp` (simple NLP)
- [ ] Create `examples/rocket_trajectory.cpp` (trajectory optimization)
- [ ] Update `design_overview.md` with Phase 6 summary
- [ ] Update README with optimization feature highlights

---

## Verification Plan

### Automated Tests

All tests run via existing infrastructure:

```bash
# Build and run all tests
./scripts/ci.sh

# Run specific optimization tests
cd build && ctest -R test_opti --output-on-failure
```

### Test Cases

#### Rosenbrock Tests

Based on [test_opti_rosenbrock.py](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/test_optimization/test_opti_rosenbrock.py):

| Test | Expected Result |
|------|-----------------|
| 2D Unconstrained | `x ≈ 1, y ≈ 1` |
| 2D Constrained (unit circle) | `x ≈ 0.7864, y ≈ 0.6177` |
| N-D (N=10) | All `x[i] ≈ 1` |

#### Rocket Trajectory Test

Based on [test_opti_optimal_control_manual_integration.py](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/test_optimization/test_opti_optimal_control_manual_integration.py):

| Test | Expected Result |
|------|-----------------|
| Rocket control | `a_max ≈ 0.0218` |

### New Test File Structure

```cpp
// tests/optimization/test_opti.cpp

#include <gtest/gtest.h>
#include <metis/metis.hpp>

TEST(OptiTests, Rosenbrock2D_Unconstrained) {
    metis::Opti opti;
    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);
    
    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);
    
    auto sol = opti.solve();
    EXPECT_NEAR(sol.value(x), 1.0, 1e-4);
    EXPECT_NEAR(sol.value(y), 1.0, 1e-4);
}

TEST(OptiTests, Rosenbrock2D_Constrained) {
    metis::Opti opti;
    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);
    
    opti.subject_to(x * x + y * y <= 1);  // Unit circle
    
    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);
    
    auto sol = opti.solve();
    EXPECT_NEAR(sol.value(x), 0.7864, 1e-3);
    EXPECT_NEAR(sol.value(y), 0.6177, 1e-3);
}

TEST(OptiTests, RocketTrajectory) {
    // ... (trajectory optimization test)
}
```

### Manual Verification

1. **Run examples and visually inspect output**:
   ```bash
   ./build/examples/optimization_intro
   ./build/examples/rocket_trajectory
   ```

2. **Verify IPOPT solver status** in verbose output:
   - Check "Optimal Solution Found" message
   - Verify constraint satisfaction

---

## Dependencies & Prerequisites

- **Existing**: Eigen ≥ 3.4, CasADi ≥ 3.6 (includes IPOPT), C++20
- **No new dependencies**

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| IPOPT convergence issues | Medium | Start with simple problems, provide sensible defaults |
| Scaling sensitivity | High | Document scaling best practices, test with poorly-scaled problems |
| Constraint violation on frozen vars | Low | Defer freeze feature to later phase |
| Symbolic vector indexing | Medium | Carefully test CasADi slice semantics |

---

## Success Criteria

1. ☐ `metis::Opti` class implemented with core methods
2. ☐ `metis::OptiSol` provides correct value extraction
3. ☐ `derivative_of()` and `constrain_derivative()` work for trajectory problems
4. ☐ Rosenbrock tests pass (2D, constrained, N-D)
5. ☐ Rocket trajectory test passes (`a_max ≈ 0.0218`)
6. ☐ Examples compile and run successfully
7. ☐ CI pipeline (`./scripts/ci.sh`) passes
8. ☐ Code coverage maintained at >90%

---

## Example Blueprints

### Example 1: `optimization_intro.cpp` — Rosenbrock Benchmark

Based on [nd_rosenbrock/run_times.py](file:///home/tanged/sources/metis/reference/AeroSandbox/tutorial/01%20-%20Optimization%20and%20Math/01%20-%20Optimization%20Benchmark%20Problems/nd_rosenbrock/run_times.py)

```cpp
/**
 * @file optimization_intro.cpp
 * @brief Rosenbrock optimization benchmark
 * 
 * Demonstrates core metis::Opti API with the classic Rosenbrock problem:
 *   minimize sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
 * 
 * Global optimum: x = [1, 1, ..., 1], objective = 0
 */

#include <metis/metis.hpp>
#include <iostream>

template <typename Scalar>
Scalar rosenbrock_objective(const metis::MetisVector<Scalar>& x) {
    Scalar obj = 0.0;
    for (int i = 0; i < x.size() - 1; ++i) {
        Scalar term1 = 100 * metis::pow(x(i+1) - x(i) * x(i), 2);
        Scalar term2 = metis::pow(1 - x(i), 2);
        obj = obj + term1 + term2;
    }
    return obj;
}

int main() {
    std::cout << "=== Rosenbrock Optimization Benchmark ===" << std::endl;
    
    // 2D Rosenbrock (unconstrained)
    {
        metis::Opti opti;
        auto x = opti.variable(0.0);
        auto y = opti.variable(0.0);
        
        auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
        opti.minimize(f);
        
        auto sol = opti.solve();
        std::cout << "2D Unconstrained: x=" << sol.value(x) 
                  << ", y=" << sol.value(y) << std::endl;
    }
    
    // 2D Rosenbrock (constrained to unit circle)
    {
        metis::Opti opti;
        auto x = opti.variable(0.0);
        auto y = opti.variable(0.0);
        
        opti.subject_to(x * x + y * y <= 1);  // Unit circle constraint
        
        auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
        opti.minimize(f);
        
        auto sol = opti.solve();
        std::cout << "2D Constrained: x=" << sol.value(x) 
                  << ", y=" << sol.value(y) << std::endl;
    }
    
    // N-D Rosenbrock
    {
        constexpr int N = 10;
        metis::Opti opti;
        auto x = opti.variable(N, 0.0);  // N variables, init_guess=0
        
        opti.subject_to(x >= 0);  // Keep unimodal
        opti.minimize(rosenbrock_objective(x));
        
        auto sol = opti.solve();
        std::cout << "N-D (N=" << N << "): x = [" 
                  << sol.value(x).transpose() << "]" << std::endl;
    }
    
    return 0;
}
```

### Example 2: `beam_deflection.cpp` — Structural Analysis

Based on [gp_beam/demo_code.py](file:///home/tanged/sources/metis/reference/AeroSandbox/tutorial/01%20-%20Optimization%20and%20Math/01%20-%20Optimization%20Benchmark%20Problems/gp_beam/demo_code.py)

This elegantly demonstrates the `derivative_of()` chain for structural analysis:

```cpp
/**
 * @file beam_deflection.cpp
 * @brief Cantilever beam bending analysis using derivative chains
 * 
 * Solving Euler-Bernoulli beam equations via optimization:
 *   d²w/dx² = M/EI  (curvature = moment / stiffness)
 *   dM/dx = V       (shear)
 *   dV/dx = q       (distributed load)
 */

#include <metis/metis.hpp>
#include <iostream>

int main() {
    constexpr int N = 50;          // Discretization nodes
    constexpr double L = 6.0;      // Beam length [m]
    constexpr double EI = 1.1e4;   // Bending stiffness [N·m²]
    
    Eigen::VectorXd x = metis::linspace(0.0, L, N);
    Eigen::VectorXd q = Eigen::VectorXd::Constant(N, 110);  // Distributed load [N/m]
    
    metis::Opti opti;
    
    auto w = opti.variable(N, 0.0);   // Displacement [m]
    
    auto th = opti.derivative_of(w, x, 0.0);           // Slope [rad]
    auto M = opti.derivative_of(th * EI, x, 0.0);      // Moment [N·m]
    auto V = opti.derivative_of(M, x, 0.0);            // Shear [N]
    
    opti.constrain_derivative(V, x, q);  // dV/dx = q
    
    // Boundary conditions (fixed end at x=0)
    opti.subject_to({
        w(0) == 0,
        th(0) == 0,
        M(N-1) == 0,
        V(N-1) == 0
    });
    
    auto sol = opti.solve();
    
    std::cout << "Tip deflection: " << sol.value(w)(N-1) << " m" << std::endl;
    // Expected: ~1.62 m
    
    return 0;
}
```

---

## Future Considerations: Phase 7+ Roadmap

### Phase 7: Advanced Optimization Features

The following features are deferred from Phase 6 to maintain scope. Detailed implementation notes provided for future reference.

---

#### 7.1 Variable Freezing & Categories

**Purpose**: Enable partial optimization for design studies (e.g., freeze wingspan, optimize fuselage).

**API Design**:
```cpp
struct VariableOptions {
    std::string category = "Uncategorized";
    bool freeze = false;
};

class Opti {
    Opti(const std::vector<std::string>& categories_to_freeze = {});
    
    SymbolicScalar variable(double init_guess, 
                            const VariableOptions& opts = {});
};
```

**Implementation Notes**:
- Frozen variables become parameters internally
- Track variable categories in `std::map<std::string, std::vector<MX>>`
- Support `freeze_style`: "parameter" (symbolic) vs "value" (numeric substitution)

**Reference**: [opti.py L72-L366](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/opti.py#L72-L366)

---

#### 7.2 Solution Caching

**Purpose**: Warm-start optimization from previous solutions, persist solutions across sessions.

**API Design**:
```cpp
class Opti {
    void set_cache_file(const std::string& filename);
    void save_solution() const;
    void load_frozen_from_cache();
};

class OptiSol {
    void save(const std::string& filename) const;
    static OptiSol load(const std::string& filename);
};
```

**Implementation Notes**:
- Use JSON for portability (nlohmann/json or similar)
- Store: variable values, categories, solve stats
- Handle version mismatches gracefully

**Reference**: [opti.py L952-L997](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/opti.py#L952-L997)

---

#### 7.3 Parametric Studies (`solve_sweep`)

**Purpose**: Efficiently solve optimization for ranges of parameter values.

**API Design**:
```cpp
struct SweepResult {
    Eigen::MatrixXd parameter_values;  // P x N_runs
    std::vector<OptiSol> solutions;
};

SweepResult solve_sweep(
    const std::map<SymbolicScalar, Eigen::VectorXd>& parameter_mapping,
    bool update_initial_guesses = false
);
```

**Implementation Notes**:
- Reuse Opti problem structure across solves
- Optionally warm-start from previous solution
- Parallelize with OpenMP or similar

**Reference**: [opti.py L734-L837](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/opti.py#L734-L837)

---

#### 7.4 NaN Propagation Sparsity

**Purpose**: Efficient Jacobian computation by detecting sparsity patterns through NaN propagation.

**Background**: When computing Jacobians symbolically, many entries are structurally zero. NaN propagation can identify these without full symbolic differentiation.

**Implementation Considerations**:
- CasADi already has excellent sparsity detection
- This is more relevant for Python/NumPy hybrid operations
- May not be needed for pure CasADi workflows

**Recommendation**: Evaluate actual sparsity performance in Phase 6 before implementing.

---

#### 7.5 Alternative Solvers

**SNOPT Backend**:
- Sparse sequential quadratic programming
- Better for large-scale, sparse problems
- Requires SNOPT license

**KNITRO Backend**:
- Commercial alternative to IPOPT
- Active-set and interior-point methods

**Implementation**:
```cpp
enum class Solver { IPOPT, SNOPT, KNITRO };

struct OptiOptions {
    Solver solver = Solver::IPOPT;
    // Solver-specific options...
};
```

---

#### 7.6 Multiple Shooting / Direct Collocation

**Purpose**: Advanced transcription methods for trajectory optimization.

**Direct Collocation**:
- Defect constraints at collocation points
- Higher-order accuracy
- Larger but sparser NLP

**Multiple Shooting**:
- Segment-based integration
- Better for unstable dynamics
- Smaller NLP, denser

**Reference**: Consider Dymos (OpenMDAO) or GPOPS-II for design patterns.

---

### Phase 7 Estimated Scope

| Feature | Complexity | Priority |
|---------|------------|----------|
| Variable Freezing | Medium | High |
| Solution Caching | Low-Medium | High |
| `solve_sweep()` | Medium | Medium |
| NaN Propagation | High | Low |
| Alternative Solvers | High | Low |
| Multiple Shooting | Very High | Low |

---

*Generated by Metis Dev Team - Phase 6 Planning*
