# Metis Phase 7: Advanced Introspection & Optimization Extensions

**Goal**: Implement sparsity introspection, higher-order symbolic derivatives, and optimization extensions to complete the Metis core math library.

**Status**: In Progress
**Created**: 2025-12-16
**Last Updated**: 2025-12-16

---

## Executive Summary

Phase 7 adds advanced introspection and analysis capabilities:

1. **Sparsity Introspection** — Expose CasADi's sparsity machinery via `SparsityPattern` and `jacobian_sparsity`
2. **Higher-Order Symbolic Derivatives** — `sym_gradient`, `hessian`, `hessian_lagrangian`
3. **Variable Freezing & Categories** — Partial optimization support
4. **Solution Caching** — JSON persistence for warm-starting
5. **API Streamlining** — Cleaner syntax (breaking changes OK)
6. **Parametric Studies** — `solve_sweep` for batch optimization
7. **Sparse Matrix Types** — Eigen sparse integration

> [!NOTE]
> **Deferred to Phase 8**: NaN-Propagation for black-box sparsity detection, alternative solvers (SNOPT/KNITRO), multiple shooting/collocation.

---

## User Review Required

> [!IMPORTANT]
> **Breaking Changes Allowed**: Per user confirmation, no downstream repos depend on Metis yet. Breaking API changes are acceptable for Phase 7.

---

## Proposed Implementation Structure

```
include/metis/
├── core/
│   └── Sparsity.hpp           # [NEW] SparsityPattern class
├── math/
│   └── AutoDiff.hpp           # [MODIFY] Add sym_gradient, hessian
├── optimization/
│   ├── Opti.hpp               # [MODIFY] Add freezing, caching, solve_sweep
│   ├── OptiSol.hpp            # [MODIFY] Add save/load
│   └── OptiCache.hpp          # [NEW] JSON caching utilities

tests/
├── core/
│   └── test_sparsity.cpp      # [NEW] Sparsity tests
├── math/
│   └── test_autodiff.cpp      # [NEW] Hessian/gradient tests
├── optimization/
│   └── test_opti_advanced.cpp # [NEW] Freezing, caching, sweep tests

docs/implementation_plans/
└── phase7_introspection.md    # [NEW] This document
```

---

## Detailed Implementation Specifications

### Milestone 7.1: Sparsity Introspection — **P0**

#### [NEW] `include/metis/core/Sparsity.hpp`

```cpp
#pragma once
#include <casadi/casadi.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace metis {

/**
 * @brief Wrapper around CasADi Sparsity for pattern analysis
 *
 * Provides query interface for Jacobian/Hessian sparsity patterns
 * useful for debugging, optimization configuration, and visualization.
 */
class SparsityPattern {
  public:
    SparsityPattern() = default;
    explicit SparsityPattern(const casadi::Sparsity& sp);

    // === Query Interface ===
    int n_rows() const;
    int n_cols() const;
    int nnz() const;  // Number of structural non-zeros
    double density() const;  // nnz / (nrows * ncols)

    // === Element Access ===
    bool has_nz(int row, int col) const;
    std::vector<std::pair<int, int>> nonzeros() const;

    // === Export Formats ===
    std::tuple<std::vector<int>, std::vector<int>> get_triplet() const;
    std::tuple<std::vector<int>, std::vector<int>> get_crs() const;
    std::tuple<std::vector<int>, std::vector<int>> get_ccs() const;

    // === Visualization ===
    std::string to_string() const;  // ASCII spy plot
    void export_spy_dot(const std::string& filename) const;

    // === Underlying Access ===
    const casadi::Sparsity& casadi_sparsity() const { return sp_; }

  private:
    casadi::Sparsity sp_;
};

// === Sparsity Query Functions ===

/**
 * @brief Get Jacobian sparsity without computing the full Jacobian
 */
SparsityPattern jacobian_sparsity(const SymbolicScalar& expr, 
                                   const SymbolicScalar& vars);

/**
 * @brief Get Hessian sparsity
 */
SparsityPattern hessian_sparsity(const SymbolicScalar& expr, 
                                  const SymbolicScalar& vars);

/**
 * @brief Get sparsity of a metis::Function Jacobian
 */
SparsityPattern get_jacobian_sparsity(const Function& fn);

} // namespace metis
```

**Implementation Notes**:
- Wrap `casadi::Sparsity` class
- Use `casadi::MX::jacobian_sparsity()` for efficient pattern extraction
- ASCII spy plot: iterate nonzeros and print `*` vs `.`
- **COMPLETED**: Implemented `SparsityPattern` in `include/metis/core/Sparsity.hpp`. Refactored to use `metis::SymbolicScalar` and `metis::NumericMatrix` native types. Added `visualize_spy` for PDF generation using Graphviz HTML tables. Added `examples/intro/sparsity_intro.cpp` with 2D Laplacian example.

---

### Milestone 7.2: Higher-Order Symbolic Derivatives — **P0**

#### [MODIFY] `include/metis/math/AutoDiff.hpp`

Add the following functions:

```cpp
namespace metis {

/**
 * @brief Symbolic gradient (for scalar-output functions)
 *
 * Unlike the numerical gradient() in Calculus.hpp, this computes
 * the symbolic gradient via CasADi automatic differentiation.
 *
 * @param expr Scalar expression to differentiate
 * @param vars Variables to differentiate with respect to
 * @return Column vector of partial derivatives
 */
SymbolicVector sym_gradient(const SymbolicScalar& expr, 
                             const SymbolicScalar& vars);

template <typename... Vars>
SymbolicVector sym_gradient(const SymbolicScalar& expr, 
                             const Vars&... vars);

/**
 * @brief Hessian matrix (second-order derivatives)
 *
 * @param expr Scalar expression
 * @param vars Variables
 * @return Symmetric Hessian matrix
 */
SymbolicMatrix hessian(const SymbolicScalar& expr, 
                        const SymbolicScalar& vars);

/**
 * @brief Hessian of Lagrangian for constrained optimization
 *
 * Computes ∇²L where L = f(x) + λᵀg(x)
 *
 * @param objective Objective function f(x)
 * @param constraints Constraint functions g(x)
 * @param vars Decision variables x
 * @param multipliers Lagrange multipliers λ
 * @return Hessian of Lagrangian
 */
SymbolicMatrix hessian_lagrangian(
    const SymbolicScalar& objective,
    const SymbolicScalar& constraints,
    const SymbolicScalar& vars,
    const SymbolicScalar& multipliers);

} // namespace metis
```

**Implementation**:
```cpp
inline SymbolicVector sym_gradient(const SymbolicScalar& expr, 
                                    const SymbolicScalar& vars) {
    // CasADi gradient returns row vector, transpose to column
    return to_eigen(SymbolicScalar::gradient(expr, vars).T());
}

inline SymbolicMatrix hessian(const SymbolicScalar& expr, 
                               const SymbolicScalar& vars) {
    return to_eigen(SymbolicScalar::hessian(expr, vars));
}
```

---

### Milestone 7.3: Variable Freezing & Categories — **P1**

#### [MODIFY] `include/metis/optimization/Opti.hpp`

```cpp
namespace metis {

/**
 * @brief Variable creation options
 */
struct VariableOptions {
    std::string category = "Uncategorized";
    bool freeze = false;
};

class Opti {
  public:
    /**
     * @brief Construct optimizer with optional frozen categories
     * @param categories_to_freeze Variable categories to freeze
     */
    explicit Opti(const std::vector<std::string>& categories_to_freeze = {});

    /**
     * @brief Create a categorized scalar variable
     */
    SymbolicScalar variable(double init_guess,
                            const VariableOptions& opts,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt);

    /**
     * @brief Get all variables in a category
     */
    std::vector<SymbolicScalar> get_category(const std::string& category) const;

    /**
     * @brief Freeze a variable to its current initial guess
     */
    void freeze(const SymbolicScalar& var);

    /**
     * @brief Unfreeze a previously frozen variable
     */
    void unfreeze(const SymbolicScalar& var);

  private:
    std::map<std::string, std::vector<SymbolicScalar>> categories_;
    std::set<SymbolicScalar> frozen_vars_;
};

} // namespace metis
```

**Reference**: [opti.py L72-L366](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/opti.py#L72-L366)

---

### Milestone 7.4: Solution Caching — **P1**

#### [NEW] `include/metis/optimization/OptiCache.hpp`

```cpp
#pragma once
#include "OptiSol.hpp"
#include <fstream>
#include <nlohmann/json.hpp>  // or lightweight JSON alternative

namespace metis {

/**
 * @brief Save optimization solution to JSON file
 */
void save_solution(const OptiSol& sol, 
                   const std::string& filename,
                   const std::map<std::string, SymbolicScalar>& named_vars);

/**
 * @brief Load solution from JSON file
 * @return Map of variable names to values
 */
std::map<std::string, Eigen::VectorXd> load_solution(const std::string& filename);

} // namespace metis
```

#### [MODIFY] `include/metis/optimization/OptiSol.hpp`

```cpp
class OptiSol {
  public:
    // ... existing methods ...

    /**
     * @brief Save solution to file
     */
    void save(const std::string& filename) const;

    /**
     * @brief Check if a cache file exists and is valid
     */
    static bool cache_valid(const std::string& filename);
};
```

**Reference**: [opti.py L952-L997](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/opti.py#L952-L997)

---

### Milestone 7.5: API Streamlining — **P2**

#### A. ADL-Friendly Math Functions

Ensure `metis::pow`, `metis::sin`, etc. are found via ADL when called without namespace:

```cpp
// In Arithmetic.hpp - already works via ADL when operands are SymbolicScalar
// Just ensure documentation clarifies this
```

#### B. Simplified Symbol Creation

```cpp
namespace metis {

/**
 * @brief Create a named symbolic scalar (shorthand)
 */
inline SymbolicScalar sym(const std::string& name) {
    return SymbolicScalar::sym(name);
}

/**
 * @brief Create a named symbolic vector (shorthand)
 */
inline SymbolicVector sym(const std::string& name, int size) {
    return to_eigen(SymbolicScalar::sym(name, size, 1));
}

} // namespace metis
```

#### C. Cleaner Function Construction (Future)

```cpp
// Future API exploration - lambda-based function construction
// Defer to Phase 8 if complexity is high
```

---

### Milestone 7.6: Parametric Studies (`solve_sweep`) — **P2**

#### [MODIFY] `include/metis/optimization/Opti.hpp`

```cpp
namespace metis {

/**
 * @brief Result of a parametric sweep
 */
struct SweepResult {
    Eigen::MatrixXd parameter_values;  // P x N_runs
    std::vector<OptiSol> solutions;
    std::vector<bool> converged;
};

class Opti {
  public:
    // ... existing methods ...

    /**
     * @brief Solve optimization for multiple parameter values
     *
     * @param parameter_mapping Map of parameter -> values to sweep
     * @param update_initial_guesses Warm-start from previous solution
     * @return SweepResult containing all solutions
     */
    SweepResult solve_sweep(
        const std::map<SymbolicScalar, Eigen::VectorXd>& parameter_mapping,
        bool update_initial_guesses = false,
        const OptiOptions& options = {});
};

} // namespace metis
```

**Reference**: [opti.py L734-L837](file:///home/tanged/sources/metis/reference/AeroSandbox/aerosandbox/optimization/opti.py#L734-L837)

---

### Milestone 7.7: Sparse Matrix Types — **P3**

#### [MODIFY] `include/metis/core/MetisTypes.hpp`

```cpp
#include <Eigen/Sparse>

namespace metis {

// === Sparse Matrix Types ===
using SparseMatrix = Eigen::SparseMatrix<double>;
using SparseMatrixRowMajor = Eigen::SparseMatrix<double, Eigen::RowMajor>;

/**
 * @brief Convert dense matrix to sparse, dropping near-zero entries
 */
SparseMatrix to_sparse(const NumericMatrix& dense, double tol = 1e-14);

/**
 * @brief Construct sparse matrix from triplets
 */
SparseMatrix sparse_from_triplets(
    int rows, int cols,
    const std::vector<Eigen::Triplet<double>>& triplets);

} // namespace metis
```

---

## Task Breakdown & Milestones

### Milestone 7.1: Sparsity Introspection (Completed)

- [x] Create `include/metis/core/Sparsity.hpp`
- [x] Implement `SparsityPattern` class with query methods
- [x] Implement `jacobian_sparsity()`, `hessian_sparsity()`
- [x] Implement ASCII spy plot visualization
- [x] Create `tests/core/test_sparsity.cpp`
- [x] Create `examples/intro/sparsity_intro.cpp` (2D Laplacian example added)
- [x] Add PDF visualization support (`visualize_spy`)

### Milestone 7.2: Higher-Order Derivatives (Completed)

- [x] Add `sym_gradient()` to AutoDiff.hpp
- [x] Add `hessian()` to AutoDiff.hpp
- [x] Add `hessian_lagrangian()` to AutoDiff.hpp
- [x] Create `tests/math/test_autodiff.cpp` for Hessian tests
- [x] Document distinction from numerical `gradient()` in Calculus.hpp

### Milestone 7.3: Variable Freezing (Completed)
- [x] Add `VariableOptions` struct
- [x] Modify `Opti::variable()` to accept options
- [x] Implement category tracking with `std::map`
- [x] Implement freezing methods (via creation-time parameter substitution)
- [x] Add constructor with frozen categories
- [x] Test partial optimization scenarios

### Milestone 7.4: Solution Caching (Completed)

- [x] Create simple JSON read/write utility (`metis/utils/JsonUtils.hpp`)
- [x] Add `save(filename, vars)` to `OptiSol`
- [x] Create `OptiCache::load(filename)` helper
- [x] Verify roundtrip persistence with testsave()` method
- [x] Test round-trip save/load
- [x] Document cache file format

### Milestone 7.5: API Streamlining (Completed)

- [x] Add `sym_vector()` shorthand function
- [x] Create ADL documentation (`math_functions.md`)
- [x] Create convenience header (`using.hpp`)
- [x] Document ADL behavior for math functions

### Milestone 7.6: Parametric Studies (Completed)

- [x] Implement `SweepResult` struct
- [x] Implement `solve_sweep()` method
- [x] Support warm-starting between solves
- [x] Test with parameter sweep examples

### Milestone 7.7: Sparse Matrix Types (Completed)

- [x] Add sparse type aliases to MetisTypes.hpp
- [x] Add `to_sparse()`, `sparse_from_triplets()` to Linalg.hpp
- [x] Add `is_numeric_scalar_v` trait for compile-time checks

---

## Verification Plan

### Automated Tests

```bash
./scripts/ci.sh
```

### Test Cases

#### Sparsity Tests

| Test | Expected |
|------|----------|
| `jacobian_sparsity(x*y, {x,y})` | 1x2, nnz=2 |
| `hessian_sparsity(x*x*y, {x,y})` | 2x2, nnz=3 (symmetric) |
| ASCII spy plot | Visual verification |

#### Hessian Tests

| Test | Expected |
|------|----------|
| `hessian(x^2 + y^2, {x,y})` | `[[2,0],[0,2]]` |
| `hessian(x*y, {x,y})` | `[[0,1],[1,0]]` |
| Rosenbrock Hessian | Matches known form |

#### Freezing Tests

| Test | Expected |
|------|----------|
| Freeze x, optimize y | x unchanged, y optimal |
| Category freezing | All vars in category frozen |

---

## Dependencies & Prerequisites

- **Existing**: Eigen ≥ 3.4, CasADi ≥ 3.6, C++20
- **Optional**: nlohmann/json for solution caching (can use lightweight alternative)

---

## Success Criteria

1. [x] `SparsityPattern` class implemented with full query API
2. [x] `jacobian_sparsity()` and `hessian_sparsity()` working
3. [x] `sym_gradient()` and `hessian()` implemented
4. [x] Variable freezing functional for partial optimization
5. [x] Solution caching saves/loads correctly
6. [x] `solve_sweep()` runs parametric studies
7. [x] API streamlining reduces boilerplate in examples
8. [x] CI passes with >90% coverage maintained

---

## Phase 8 Preview

Features deferred from Phase 7:

| Feature | Notes |
|---------|-------|
| **NaN-Propagation Sparsity** | Black-box sparsity detection |
| **Alternative Solvers** | SNOPT, KNITRO backends |
| **Multiple Shooting** | Advanced trajectory transcription |
| **Lambda-style Function** | Syntactic sugar for function creation |

---

*Generated by Metis Dev Team - Phase 7 Planning*
