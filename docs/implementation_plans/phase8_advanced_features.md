# Metis Phase 8: Advanced Features & Solver Extensions

**Goal**: Implement advanced optimization features including NaN-propagation sparsity, alternative solvers, trajectory transcription methods (multiple shooting & collocation), and lambda-style function construction.

**Status**: Planning Draft
**Created**: 2025-12-17

---

## Executive Summary

Phase 8 introduces advanced features deferred from Phase 7:

1. **Lambda-Style Functions** — Syntactic sugar for intuitive function construction
2. **NaN-Propagation Sparsity** — Black-box Jacobian sparsity detection
3. **Alternative Solvers** — SNOPT and KNITRO backend support
4. **Trajectory Transcription** — Multiple shooting & direct collocation methods

> [!NOTE]
> These are **optional extensions**. Existing IPOPT optimization and `derivative_of()` remain the primary workflow.

---

## User Review Required

> [!IMPORTANT]
> **Feature Prioritization**:
>
> | Feature | Complexity | Priority |
> |---------|------------|----------|
> | Lambda-Style Functions | Low | P0 |
> | NaN-Propagation Sparsity | Medium | P1 |
> | Alternative Solvers | Medium-High | P2 |
> | Collocation Methods | High | P3 |
> | Multiple Shooting | Very High | P3 |

> [!WARNING]
> **Dependencies**: SNOPT/KNITRO require commercial licenses and CasADi built with their support.

---

## Proposed Structure

```
include/metis/
├── core/
│   ├── Function.hpp          # [MODIFY] Lambda-style construction
│   └── Sparsity.hpp          # [MODIFY] NaN-propagation sparsity
├── optimization/
│   ├── OptiOptions.hpp       # [MODIFY] Solver enum
│   ├── Collocation.hpp       # [NEW] Direct collocation
│   └── MultiShooting.hpp     # [NEW] Multiple shooting

tests/optimization/
├── test_opti_solvers.cpp     # [NEW]
├── test_collocation.cpp      # [NEW]
└── test_multishoot.cpp       # [NEW]

examples/optimization/
├── collocation_demo.cpp      # [NEW]
└── multiple_shooting.cpp     # [NEW]
```

---

## Milestone 8.1: Lambda-Style Functions — **P0**

### Motivation

Current API is verbose:
```cpp
auto x = SymbolicScalar::sym("x");
auto y = SymbolicScalar::sym("y");
metis::Function fn("f", {x, y}, {x*x + y*y});
```

Proposed API:
```cpp
auto fn = metis::make_function<2, 1>("f", [](auto x, auto y) {
    return x*x + y*y;
});
```

### API Specification

```cpp
namespace metis {

template <int NInputs, int NOutputs, typename Func>
Function make_function(const std::string& name, Func&& fn);

// Named inputs variant
template <typename Func>
Function make_function(const std::string& name,
                       const std::vector<std::string>& input_names,
                       Func&& fn);
} // namespace metis
```

---

## Milestone 8.2: NaN-Propagation Sparsity — **P1**

### Motivation

CasADi symbolic sparsity works for pure symbolic graphs. NaN-propagation enables black-box sparsity detection for hybrid/external functions.

### API Specification

```cpp
namespace metis {

struct NaNSparsityOptions {
    NumericVector reference_point;
    double perturbation = 1e-7;
};

template <typename Func>
SparsityPattern nan_propagation_sparsity(
    Func&& fn, int n_inputs, int n_outputs,
    const NaNSparsityOptions& opts = {});

} // namespace metis
```

---

## Milestone 8.3: Alternative Solvers — **P2**

### API Specification

```cpp
namespace metis {

enum class Solver { IPOPT, SNOPT, KNITRO, QPOASES };

struct SNOPTOptions {
    int major_iterations_limit = 1000;
    double major_optimality_tolerance = 1e-6;
};

struct KNITROOptions {
    int algorithm = 0;  // 0=auto, 1=interior, 2=active-set
    double opttol = 1e-6;
};

// Add to OptiOptions
struct OptiOptions {
    Solver solver = Solver::IPOPT;
    SNOPTOptions snopt_opts;
    KNITROOptions knitro_opts;
    // ... existing fields ...
};

bool solver_available(Solver solver);

} // namespace metis
```

---

## Milestone 8.4: Trajectory Transcription — **P3**

### 8.4a: Direct Collocation

Direct collocation discretizes state and control, enforcing dynamics via algebraic defect constraints at collocation points.

**Methods**:
| Method | Order | Collocation Points |
|--------|-------|-------------------|
| Trapezoidal | 2nd | Segment endpoints |
| Hermite-Simpson | 4th | Endpoints + midpoint |
| Gauss-Lobatto | 2N | Lobatto quadrature nodes |
| Radau | 2N-1 | Radau quadrature nodes |

```cpp
namespace metis {

enum class CollocationScheme {
    Trapezoidal,
    HermiteSimpson,
    LegendreGaussLobatto,
    LegendreGaussRadau
};

struct CollocationOptions {
    CollocationScheme scheme = CollocationScheme::HermiteSimpson;
    int n_segments = 10;
    int poly_order = 3;  // For LGL/LGR
};

class DirectCollocation {
  public:
    explicit DirectCollocation(Opti& opti);

    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf,
          const CollocationOptions& opts = {});

    template <typename Func>
    void set_dynamics(Func&& dynamics);

    void add_defect_constraints();
    void set_initial_state(const NumericVector& x0);
    void set_final_state(const NumericVector& xf);

  private:
    Opti& opti_;
    // ...
};

} // namespace metis
```

### 8.4b: Multiple Shooting

Multiple shooting integrates explicitly within segments, adding continuity constraints at boundaries.

```cpp
namespace metis {

struct ShootingSegment {
    int n_nodes;
    double t_start, t_end;
};

struct MultiShootingOptions {
    std::vector<ShootingSegment> segments;
    std::string integrator = "rk4";  // "rk4", "cvodes"
    double integrator_tol = 1e-8;
};

class MultipleShooting {
  public:
    explicit MultipleShooting(Opti& opti);

    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, const MultiShootingOptions& opts);

    template <typename Func>
    void set_dynamics(Func&& dynamics);

    void add_continuity_constraints();

  private:
    Opti& opti_;
    // ...
};

} // namespace metis
```

### Comparison

| Method | NLP Size | Sparsity | Stiff Systems | Accuracy |
|--------|----------|----------|---------------|----------|
| Direct (derivative_of) | Small | Dense | Poor | Low |
| Collocation | Large | Sparse | Good | High |
| Multiple Shooting | Medium | Block | Excellent | Variable |

---

## Task Breakdown

### Milestone 8.1: Lambda Functions — 2-3 days
- [x] Add `make_function<N,M>()` to Function.hpp
- [x] Implement tuple handling for multi-output
- [x] Add tests and example

### Milestone 8.2: NaN-Propagation — 2-3 days
- [x] Implement `nan_propagation_sparsity()`
- [x] Add comparison tests with symbolic detection

### Milestone 8.3: Alternative Solvers — 3-4 days
- [x] Add `Solver` enum and options
- [x] Implement solver dispatch
- [x] Add `solver_available()` utility

###- [x] **Trajectory Transcription** (Milestone 8.4)
  - [x] Implement `DirectCollocation` (Trapezoidal, Hermite-Simpson)
  - [x] Implement `MultipleShooting` with CasADi integrators
  - [x] Comparison examples
- [ ] Document trade-offs

---

## Verification Plan

```bash
./scripts/ci.sh
cd build && ctest -R "test_function|test_sparsity|test_opti" --output-on-failure
```

### Key Test Cases

| Feature | Test | Expected |
|---------|------|----------|
| Lambda | `make_function<2,1>` | Valid function |
| NaN Sparsity | Diagonal fn | Diagonal pattern |
| Solvers | `solver_available(IPOPT)` | `true` |
| Collocation | Double integrator | Matches analytic |
| Multi-shoot | Van der Pol | Converges |

---

## Success Criteria

1. [ ] Lambda functions work for various input/output counts
2. [ ] NaN sparsity matches symbolic for traceable functions
3. [ ] Solver selection with graceful fallback
4. [ ] Collocation solves trajectory problems
5. [ ] Multiple shooting handles stiff dynamics
6. [ ] All existing tests pass
7. [ ] Coverage >90%

---

## Phase 9+ Preview

| Feature | Notes |
|---------|-------|
| Parallel Optimization | Multi-start, parallel sweeps |
| MPC Framework | Real-time model predictive control |
| Code Generation | C/CUDA export for embedded |

---

*Generated by Metis Dev Team - Phase 8 Planning*
