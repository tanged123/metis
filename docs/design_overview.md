# Metis v2.0.0 Design Overview

## 1. Project Mission

Metis is a high-performance C++ numerical framework implementing the **Code Transformations** paradigm. It serves as a drop-in replacement for standard math libraries, enabling engineers to write physics models once and execute them in two distinct modes:

* **Fast Numeric Mode**: For simulation, debugging, and real-time control (using standard `double` and Eigen).
* **Symbolic Trace Mode**: For generating static computational graphs to enable automatic differentiation and gradient-based optimization (using CasADi).

## 2. Core Architectural Principles

### A. The "Template-First" Traceability Paradigm

The core innovation is **Traceability**: the ability to inspect and transform numerical code.

* **Constraint**: User physics models must be templated on a generic `Scalar` type.
* **Implementation**: Use C++ Templates (compile-time polymorphism) rather than `std::variant` (runtime polymorphism).
* **Goal**: Zero-cost abstraction. In Numeric Mode, the compiler generates assembly identical to raw Eigen/C++.

### B. The Dual-Backend Type System

The framework defines a unified type alias system that routes to specific backends via template specialization.

| Feature | Numeric Backend (Fast) | Symbolic Backend (Graph) |
|---|---|---|
| **Scalar** | `double` (or float) | `casadi::MX` |
| **Matrix** | `Eigen::MatrixXd` | `Eigen::Matrix<casadi::MX>` |
| **Mutation** | Direct Memory | CasADi Register Machine |

### C. The Dispatch Layer (The "Agnostic" Math Stack)

To maintain traceability, the framework intercepts all mathematical operators.

* **Mechanism**: A custom namespace (`metis::`) shadows `std::`.
* **Implementation**: Use C++20 Concepts to dispatch logic (e.g., `std::sin` vs `casadi::sin`) at compile time.

## 3. Type Handling & Control Flow Policies

To preserve the static computational graph structure, strict policies are enforced on C++ types.

### A. The "Red Line": Structural vs. Value Logic

* **Structural Logic (Allowed)**: Integers, Booleans, and logic determined at compile/trace time (e.g., `num_segments`, `use_viscous_model`). These define the shape of the graph.
* **Value Logic (Modified)**: Floating-point values dependent on optimization variables. These define the data flow through the graph.

### B. Branching Logic (`metis::where`)

Standard C++ `if/else` cannot branch on symbolic types because symbols do not evaluate to true/false during graph construction.

* **Solution**: `metis::where(condition, true_val, false_val)` and `metis::select()` for multi-way branching.
  * **Numeric Mode**: Compiles to `cond ? a : b`.
  * **Symbolic Mode**: Compiles to a switch node `casadi::if_else(cond, a, b)`.

See `docs/user_guides/math_functions.md` for the full dispatch table.

### C. Loop Handling

* **Standard Loops**: Standard `for` loops are supported. The symbolic backend unrolls these loops into a chain of graph operations.
* **Variable Bounds**: Loops with variable iteration counts (e.g., `i < optimization_var`) are banned as they break the static graph topology.

## 4. v2.0.0 Feature Scope

The following capabilities are implemented in v2.0.0.

### Core & Math

- Scalar concept and dual-backend math primitives (arithmetic, trig, logic)
- Linear algebra shim with **linear solve policies** (dense, sparse-direct, iterative Krylov) -- see `docs/user_guides/math_functions.md`
- Interpolation (1D/ND, linear/cubic/monotonic) -- see `docs/user_guides/interpolation.md`
- **Root finding** with a globalization stack (trust-region Newton, line-search, Broyden, pseudo-transient continuation) and differentiable implicit function wrappers -- see `docs/user_guides/root_finding.md`
- ODE integration (`solve_ivp`, `quad`, multiple steppers) -- see `docs/user_guides/integration.md`
- **Second-order integrators** (Stormer-Verlet, Runge-Kutta-Nystrom 4) for `q'' = a(t, q)` systems
- **Mass-matrix integrators** for `M(t,y) y' = f(t,y)` systems
- B-Splines, spacing functions, finite differences, quaternion algebra, rotations, surrogate models

### Uncertainty Quantification

- **Polynomial chaos expansions** (Askey-scheme bases, projection/regression fitting, symbolic moments) -- see `docs/user_guides/polynomial_chaos.md`
- **Stochastic quadrature** (probability-measure rules, tensor grids, Smolyak sparse grids) -- see `docs/user_guides/stochastic_quadrature.md`

### Automatic Differentiation & Sparsity

- Forward and adjoint Jacobian/Hessian, Hessian-vector products, Lagrangian second-order adjoints
- **Sparsity pipelines**: NaN-propagation sparsity detection, graph coloring, sparse Jacobian/Hessian kernels -- see `docs/user_guides/sparsity.md`

### Structural Analysis

- **Structural diagnostics**: observability and identifiability preflight checks -- see `docs/user_guides/structural_diagnostics.md`
- **Structural transforms**: alias elimination, BLT decomposition, tearing recommendations -- see `docs/user_guides/structural_transforms.md`

### Optimization & Trajectory

- Opti interface (IPOPT/SNOPT, variable freezing, categories, explicit scaling)
- **Scaling diagnostics** (`opti.analyze_scaling()`) for preflight detection of poorly scaled problems
- **Error handling** via typed exception hierarchy (`MetisError`, `InvalidArgument`, `RuntimeError`, `IntegrationError`, `InterpolationError`)
- Trajectory optimization: direct collocation, multiple shooting, pseudospectral (LG/LGR), Birkhoff pseudospectral (LGL/CGL) -- see `docs/user_guides/transcription_methods.md`
- Parametric sweeps, JIT compilation, solution save/load

### Tooling

- Graph visualization (`to_dot`, `graphviz`) -- see `docs/user_guides/graph_visualization.md`
- Function compilation and evaluation utilities

## 5. API Reference Example

```cpp
// User Physics Model
template <typename Scalar>
Scalar compute_drag(const Scalar& velocity, const Scalar& rho) {
    // 1. Procedural declarations allowed
    Scalar drag = 0.0;

    // 2. Logic using metis::where (not if/else)
    auto is_supersonic = (velocity > 343.0);
    Scalar cd = metis::where(is_supersonic, 0.5, 0.02);

    // 3. Metis Math Dispatch (not std::pow)
    drag = 0.5 * rho * metis::pow(velocity, 2) * cd;

    return drag;
}
```

## 6. Cross-References

| Topic | User Guide |
|-------|-----------|
| Numeric mode | `docs/user_guides/numeric_computing.md` |
| Symbolic mode | `docs/user_guides/symbolic_computing.md` |
| Math functions | `docs/user_guides/math_functions.md` |
| Interpolation | `docs/user_guides/interpolation.md` |
| Integration | `docs/user_guides/integration.md` |
| Root finding | `docs/user_guides/root_finding.md` |
| Polynomial chaos | `docs/user_guides/polynomial_chaos.md` |
| Stochastic quadrature | `docs/user_guides/stochastic_quadrature.md` |
| Sparsity | `docs/user_guides/sparsity.md` |
| Structural diagnostics | `docs/user_guides/structural_diagnostics.md` |
| Structural transforms | `docs/user_guides/structural_transforms.md` |
| Graph visualization | `docs/user_guides/graph_visualization.md` |
| Optimization | `docs/user_guides/optimization.md` |
| Collocation | `docs/user_guides/collocation.md` |
| Multiple shooting | `docs/user_guides/multiple_shooting.md` |
| Pseudospectral | `docs/user_guides/pseudospectral.md` |
| Birkhoff pseudospectral | `docs/user_guides/birkhoff_pseudospectral.md` |
| Transcription overview | `docs/user_guides/transcription_methods.md` |
