# Metis v2.0.0 Usage Guide

> **Purpose**: Map of the Metis library for AI agents and developers. Each section gives a brief overview and links to the detailed user guide. For the full API, see the [Doxygen docs](https://tanged123.github.io/metis/index.html).

---

## Table of Contents

1. [Best Practices](#best-practices)
2. [Type System](#type-system)
3. [Module Reference](#module-reference)
   - [Core Layer](#core-layer)
   - [Math Layer](#math-layer)
   - [Optimization Layer](#optimization-layer)
4. [User Guide Index](#user-guide-index)

---

## Best Practices

### 1. Template-First Design (MANDATORY)

All physics/math functions **must** be templated on `Scalar`:

```cpp
// Correct
template <typename Scalar>
Scalar my_function(const Scalar& x) { ... }

// Wrong -- breaks symbolic mode
double my_function(double x) { ... }
```

### 2. Math Dispatch -- Use `metis::` Namespace

**Always** use Metis math functions instead of `std::`:

```cpp
metis::sin(x);  metis::pow(x, 2);  metis::sqrt(x);  metis::exp(x);
// Never: std::sin(x), std::pow(x, 2), etc.
```

### 3. Branching -- Use `metis::where()`, Never `if/else`

```cpp
Scalar result = metis::where(x > 0, x, -x);

// Multi-way branching
Scalar cd = metis::select(
    {mach < 0.3, mach < 0.8, mach < 1.2},
    {Scalar(0.02), Scalar(0.025), Scalar(0.05)},
    Scalar(0.03));  // default
```

### 4. Loops -- Structural Bounds Only

```cpp
// Correct -- structural bound (known at trace time)
for (int i = 0; i < N; ++i) { ... }

// Wrong -- dynamic bound breaks symbolic mode
while (error > tolerance) { ... }
```

### 5. Type Aliases -- Use Metis Native Types

```cpp
#include <metis/core/MetisTypes.hpp>

metis::Vec3<Scalar>   // 3D vector
metis::Mat3<Scalar>   // 3x3 matrix
metis::VecX<Scalar>   // Dynamic vector
metis::MatX<Scalar>   // Dynamic matrix
```

### 6. Include Convention

```cpp
#include <metis/metis.hpp>    // Everything (recommended for applications)
#include <metis/using.hpp>    // Convenience header -- brings common symbols into scope
```

---

## Type System

### Backend Types

| Type | Numeric Mode | Symbolic Mode |
|------|--------------|---------------|
| **Scalar** | `double` | `casadi::MX` |
| **Matrix** | `Eigen::MatrixXd` | `Eigen::Matrix<casadi::MX>` |
| **Vector** | `Eigen::VectorXd` | `Eigen::Matrix<casadi::MX, Dynamic, 1>` |

### Metis Type Aliases (`metis/core/MetisTypes.hpp`)

```cpp
// Symbolic types (for graph building)
metis::SymbolicScalar   // casadi::MX
metis::SymbolicMatrix   // Eigen::Matrix<casadi::MX, Dynamic, Dynamic>
metis::SymbolicVector   // Eigen::Matrix<casadi::MX, Dynamic, 1>

// Numeric types (for evaluation)
metis::NumericMatrix    // Eigen::MatrixXd
metis::NumericVector    // Eigen::VectorXd

// Fixed-size templated types
metis::Vec2<T>, metis::Vec3<T>, metis::Vec4<T>
metis::Mat2<T>, metis::Mat3<T>, metis::Mat4<T>
metis::VecX<T>, metis::MatX<T>, metis::RowVecX<T>

// Sparse types (numeric only)
metis::SparseMatrix     // Eigen::SparseMatrix<double>
metis::SparseTriplet    // Eigen::Triplet<double>
```

### Symbolic Variable Creation

```cpp
auto x = metis::sym("x");                          // Scalar
auto M = metis::sym("M", rows, cols);              // Matrix (MX)
auto v = metis::sym_vector("v", size);             // SymbolicVector (Eigen)
auto [vec, mx] = metis::sym_vec_pair("state", 3);  // Both representations
```

### Conversion Utilities

```cpp
metis::to_mx(eigen_matrix)     // Eigen -> CasADi MX
metis::to_eigen(casadi_mx)     // CasADi MX -> Eigen
metis::as_mx(symbolic_vector)  // SymbolicVector -> single MX
metis::as_vector(casadi_mx)    // MX -> SymbolicVector
```

---

## Module Reference

### Core Layer

| File | Description |
|------|-------------|
| `MetisTypes.hpp` | Type system, aliases, `metis::sym()`, `metis::to_mx()`, `metis::to_eigen()` |
| `MetisConcepts.hpp` | C++20 concepts: `ScalarType`, `NumericScalar`, `SymbolicScalar` |
| `MetisError.hpp` | Exception hierarchy: `MetisError`, `InvalidArgument`, `RuntimeError`, `IntegrationError`, `InterpolationError` |
| `MetisIO.hpp` | `metis::eval()`, `metis::print()`, `metis::to_dot()`, `metis::graphviz()` |
| `Function.hpp` | `metis::Function` -- compiled symbolic function wrapper |
| `Sparsity.hpp` | Sparsity patterns, graph coloring, `metis::sparse_jacobian()`, `metis::sparse_hessian()` |
| `Diagnostics.hpp` | Structural observability/identifiability: `metis::analyze_structural_observability()`, `metis::analyze_structural_identifiability()` |
| `StructuralTransforms.hpp` | `metis::alias_eliminate()`, `metis::block_triangularize()`, `metis::structural_analyze()` |

See `docs/user_guides/sparsity.md`, `docs/user_guides/structural_diagnostics.md`, `docs/user_guides/structural_transforms.md`.

---

### Math Layer

#### Arithmetic & Trigonometry

Standard math dispatch (`metis::sin`, `metis::pow`, `metis::exp`, etc.) with scalar and matrix overloads. See `docs/user_guides/math_functions.md` for the full function table.

#### Logic & Branching

`metis::where()`, `metis::select()`, `metis::min()`, `metis::max()`, `metis::clamp()`, element-wise comparisons, smooth blending (`metis::sigmoid_blend`, `metis::blend`). See `docs/user_guides/math_functions.md`.

#### Calculus & Autodiff

Gradient, Jacobian, Hessian, Hessian-vector products, Lagrangian second-order adjoints, sensitivity regime selection. See `docs/user_guides/symbolic_computing.md`.

```cpp
auto J = metis::jacobian(f, x);
auto H = metis::hessian(f, x);
auto Hv = metis::hessian_vector_product(f, x, direction);
```

#### Linear Algebra

`metis::solve(A, b)` with optional `LinearSolvePolicy` for backend selection:

```cpp
metis::LinearSolvePolicy policy;
policy.backend = metis::LinearSolveBackend::SparseDirect;
policy.sparse_direct_solver = metis::SparseDirectLinearSolver::SparseLU;
auto x = metis::solve(A, b, policy);
```

Available backends: `Dense` (ColPivHouseholderQR, PartialPivLU, FullPivLU, LLT, LDLT), `SparseDirect` (SparseLU, SparseQR, SimplicialLLT, SimplicialLDLT), `IterativeKrylov` (BiCGSTAB, GMRES with preconditioner hooks). Also includes `metis::dot`, `metis::cross`, `metis::norm`, `metis::inv`, `metis::det`, `metis::eye`, `metis::zeros`, `metis::ones`, `metis::block_diag`, and more.

See `docs/user_guides/math_functions.md`.

#### Interpolation

1D and N-dimensional interpolation with `"linear"`, `"cubic"`, and `"monotonic"` methods. See `docs/user_guides/interpolation.md`.

```cpp
auto y = metis::interp1(x, xp, fp);
auto y = metis::interp_nd(point, table);
```

#### Polynomial Chaos Expansions (PCE)

Askey-scheme orthogonal polynomial bases (Hermite, Legendre, Jacobi, Laguerre), total-order and tensor-product truncation, coefficient fitting via projection or regression, symbolic mean/variance extraction.

```cpp
metis::PolynomialChaosBasis basis(dimensions, order, options);
auto coeffs = metis::pce_projection_coefficients(basis, grid, values);
auto mu = metis::pce_mean(coeffs);
auto var = metis::pce_variance(basis, coeffs);
```

See `docs/user_guides/polynomial_chaos.md`.

#### Stochastic Quadrature

Probability-measure quadrature rules, tensor-product grids, and Smolyak sparse grids for high-dimensional integration and PCE projection.

```cpp
auto rule = metis::stochastic_quadrature_rule(dim, order, family);
auto grid = metis::tensor_product_quadrature(rules);
auto sparse = metis::smolyak_sparse_grid(dimensions, level, options);
```

See `docs/user_guides/stochastic_quadrature.md`.

#### Root Finding

Nonlinear solve for `F(x) = 0` with a numeric globalization stack (trust-region Newton, line-search Newton, Broyden, pseudo-transient continuation) and differentiable implicit function wrappers for embedding solves inside symbolic graphs.

```cpp
auto result = metis::rootfinder(function, x0, opts);

// Differentiable implicit solve for use inside optimization
auto implicit_fn = metis::create_implicit_function(function, x_guess, opts, implicit_opts);
```

See `docs/user_guides/root_finding.md`.

#### ODE Integration

IVP solvers with multiple steppers (`RK4`, `CVODES`, `BDF1`, `RosenbrockEuler`), definite integration via Gauss-Kronrod quadrature. See `docs/user_guides/integration.md`.

```cpp
auto result = metis::solve_ivp(dynamics, x0, t_span);
double I = metis::quad(f, a, b);
```

#### Second-Order Integrators

Dedicated solvers for systems of the form `q'' = a(t, q)`:

```cpp
auto result = metis::solve_second_order_ivp(accel, q0, v0, t_span);
// Single steps:
metis::stormer_verlet_step(accel, q, v, t, dt);  // Symplectic
metis::rkn4_step(accel, q, v, t, dt);            // 4th-order RKN
```

#### Mass-Matrix Integrators

Native support for stiff systems `M(t,y) y' = f(t,y)`:

```cpp
auto result = metis::solve_ivp_mass_matrix(rhs, M, x0, t_span);         // Numeric
auto result = metis::solve_ivp_mass_matrix_expr(rhs, M, t, y, x0, t_span); // Symbolic (IDAS)
```

#### Sparsity Pipelines

NaN-propagation sparsity detection, graph coloring, and compiled sparse derivative kernels that avoid materializing dense Jacobian/Hessian matrices.

```cpp
auto J = metis::sparse_jacobian(result, x);
auto H = metis::sparse_hessian(objective, vars);
auto nz = J.values(x_val);  // Evaluate only nonzero entries
```

See `docs/user_guides/sparsity.md`.

#### Structural Diagnostics

Preflight checks for structural observability and identifiability before committing to an optimization solve.

```cpp
auto obs = metis::analyze_structural_observability(measurement_fn, 0);
auto id  = metis::analyze_structural_identifiability(measurement_fn, 1);
auto all = metis::analyze_structural_diagnostics(system_fn, options);
```

See `docs/user_guides/structural_diagnostics.md`.

#### Structural Transforms

Alias elimination, BLT (block lower-triangular) decomposition, and tearing recommendations for large-scale equation systems.

```cpp
auto alias   = metis::alias_eliminate(residual_fn);
auto blt     = metis::block_triangularize(residual_fn);
auto analysis = metis::structural_analyze(residual_fn);
```

See `docs/user_guides/structural_transforms.md`.

#### Other Math Modules

- **Spacing**: `metis::linspace`, `metis::cosspace`, `metis::sinspace`, `metis::logspace`, `metis::geomspace`
- **Discrete integration**: rectangular, trapezoidal, Simpson, cubic, squared-curvature methods
- **Finite differences**: `metis::finite_difference_coefficients()`
- **Rotations**: `metis::rotation_x/y/z()`, `metis::rotation_2d()`
- **Quaternions**: `metis::Quaternion<Scalar>` with full algebra, conversions, `metis::slerp()`
- **Surrogate models**: `metis::softmax`, `metis::softmin`, `metis::softabs`, `metis::sigmoid`, `metis::tanh_blend`

---

### Optimization Layer

#### Opti Interface (`Opti.hpp`)

```cpp
metis::Opti opti;

// Variables
auto x = opti.variable(1.0);                    // Scalar
auto v = opti.variable(3, 0.0);                 // Vector
auto x = opti.variable(1.0, {.category = "Wing", .freeze = true});

// Parameters (fixed between solves)
auto p = opti.parameter(5.0);

// Objective
opti.minimize(cost_function);
opti.minimize(cost_function, 1e6);   // Explicit objective scaling
opti.maximize(profit_function);

// Constraints
opti.subject_to(x >= 0);
opti.subject_to(g == 0, 1e3);        // Explicit constraint scaling
opti.subject_to(x * x + y * y <= 1);
opti.subject_to_bounds(x, lower, upper);  // Box constraints
```

See `docs/user_guides/optimization.md`.

#### Solving and Options (`OptiOptions.hpp`)

Options are passed to `solve()` via `OptiOptions`:

```cpp
auto sol = opti.solve();                                           // Defaults
auto sol = opti.solve({.max_iter = 500, .verbose = false});        // Designated initializers
auto sol = opti.solve(metis::OptiOptions{}.set_tol(1e-10));       // Builder pattern

// Solver selection
auto sol = opti.solve({.solver = metis::Solver::Ipopt});           // Default
auto sol = opti.solve({.solver = metis::Solver::Snopt});           // Requires SNOPT license

if (metis::solver_available(metis::Solver::Snopt)) { ... }
```

#### Solution Extraction (`OptiSol.hpp`)

```cpp
auto sol = opti.solve();

double x_opt = sol.value(x);                  // Scalar
metis::NumericVector v_opt = sol.value(v);    // Vector
metis::NumericMatrix M_opt = sol.value(M);    // Matrix
auto stats = sol.stats();                     // Solver statistics

// Save / load
sol.save("result.json", {{"x", x}, {"y", y}});
auto data = metis::OptiSol::load("result.json");
```

#### Scaling Diagnostics

Preflight analysis of variable, constraint, and objective scaling before solving:

```cpp
auto report = opti.analyze_scaling();
// Report contains ScalingIssue entries with severity, suggested scales, etc.
```

#### Parametric Sweep (`OptiSweep.hpp`)

```cpp
metis::OptiSweep sweep(opti);
auto results = sweep.run(parameter, values);
```

#### Trajectory Optimization

Four transcription methods are available, all sharing a common `TranscriptionBase` interface. See `docs/user_guides/transcription_methods.md` for comparison.

- **Direct collocation** -- `docs/user_guides/collocation.md`
- **Multiple shooting** -- `docs/user_guides/multiple_shooting.md`
- **Pseudospectral** (LGL/CGL) -- `docs/user_guides/pseudospectral.md`
- **Birkhoff pseudospectral** (LGL/CGL) -- `docs/user_guides/birkhoff_pseudospectral.md`

```cpp
metis::Opti opti;
metis::DirectCollocation colloc(opti);
auto [X, U, tau] = colloc.setup(n_states, n_controls, t0, tf);

colloc.set_dynamics([](const auto& x, const auto& u, const auto& t) {
    return dynamics(x, u, t);
});
colloc.add_dynamics_constraints();
colloc.set_initial_state(x0);
colloc.set_final_state(xf);

opti.minimize(objective);
auto sol = opti.solve();
```

---

## User Guide Index

| Guide | File | Topics |
|-------|------|--------|
| Numeric Computing | `docs/user_guides/numeric_computing.md` | Numeric mode, evaluation |
| Symbolic Computing | `docs/user_guides/symbolic_computing.md` | Symbolic mode, graph building |
| Math Functions | `docs/user_guides/math_functions.md` | Full metis:: math dispatch table |
| Interpolation | `docs/user_guides/interpolation.md` | 1D/ND interpolation |
| Integration | `docs/user_guides/integration.md` | ODE solvers, quadrature |
| Root Finding | `docs/user_guides/root_finding.md` | Nonlinear solves, implicit functions |
| Polynomial Chaos | `docs/user_guides/polynomial_chaos.md` | PCE bases, fitting, moments |
| Stochastic Quadrature | `docs/user_guides/stochastic_quadrature.md` | Quadrature rules, sparse grids |
| Sparsity | `docs/user_guides/sparsity.md` | Sparsity, coloring, sparse derivatives |
| Structural Diagnostics | `docs/user_guides/structural_diagnostics.md` | Observability, identifiability |
| Structural Transforms | `docs/user_guides/structural_transforms.md` | Alias elimination, BLT, tearing |
| Graph Visualization | `docs/user_guides/graph_visualization.md` | DOT export, Graphviz |
| Optimization | `docs/user_guides/optimization.md` | Opti interface, constraints |
| Collocation | `docs/user_guides/collocation.md` | Direct collocation |
| Multiple Shooting | `docs/user_guides/multiple_shooting.md` | Multiple shooting |
| Pseudospectral | `docs/user_guides/pseudospectral.md` | LGL/CGL pseudospectral |
| Birkhoff Pseudospectral | `docs/user_guides/birkhoff_pseudospectral.md` | LGL/CGL Birkhoff |
| Transcription Methods | `docs/user_guides/transcription_methods.md` | Comparison of methods |

---

## Summary for Agents

### DO NOT Reimplement

The following functionality already exists in Metis -- check the relevant user guide before building anything new:

- All basic math (`sin`, `cos`, `pow`, `exp`, `log`, `sqrt`, etc.)
- Linear algebra (`dot`, `cross`, `norm`, `inv`, `det`, `solve` with policies)
- Quaternion algebra and rotations
- Interpolation (1D, ND, multiple methods)
- Root finding (globalization stack, implicit function wrappers)
- ODE integration (`solve_ivp`, `quad`, second-order, mass-matrix)
- Polynomial chaos and stochastic quadrature
- Sparsity analysis and sparse derivative kernels
- Structural diagnostics and transforms
- Discrete integration (trapz, Simpson, etc.)
- Branching logic (`where`, `select`, `clamp`, `min`, `max`)
- Function compilation
- Optimization (`Opti`, IPOPT, SNOPT)
- Trajectory optimization (collocation, multiple shooting, pseudospectral, Birkhoff)
- Scaling diagnostics

### When Building on Metis

1. **Import via** `#include <metis/metis.hpp>` (includes everything)
2. **Use Metis types** (`metis::Vec3<Scalar>`, `metis::SymbolicScalar`)
3. **Use Metis math** (`metis::sin`, not `std::sin`)
4. **Use Metis branching** (`metis::where`, not `if/else`)
5. **Template everything** on `Scalar`
6. **Test both modes** (numeric AND symbolic)
