# Metis Moonshot Implementation Plan

Source: `docs/architecture/moonshot_roadmap.md` — Metis TODO items J-1 through J-23.

This plan is intentionally high-level. Each item describes what to build and why. Implementation details (file placement, API signatures, test structure) should follow existing Metis conventions — see `CLAUDE.md`, existing headers in `include/metis/`, and the test patterns in `tests/`.

---

## Core Fixes and Gaps (J-1 through J-12)

### J-1. Softplus C-infinity fix

**Problem**: `softplus` in `SurrogateModel.hpp` uses a hard threshold switch (`if beta*x > threshold: return x`) creating a C^0 kink in the symbolic graph.

**Goal**: Replace with a C-infinity formulation. Options:
- Numerically stable LogSumExp identity without the threshold branch
- Blend the linear regime smoothly via a sigmoid transition
- Must remain numerically stable for large `beta*x` values

**Where**: `include/metis/math/SurrogateModel.hpp`
**Tests**: Verify C-infinity by checking first and second derivatives are continuous across the transition region. Test both numeric and symbolic modes.

---

### J-2. `gradient_periodic` implementation

**Problem**: Currently delegates to standard `gradient` with a TODO comment. No periodic boundary handling.

**Goal**: Implement wrap-around finite differences at domain edges using the periodicity assumption. The last point wraps to the first.

**Where**: `include/metis/math/Calculus.hpp`
**Use cases**: Angle-of-attack derivatives, orbital true anomaly gradients, any signal defined on a periodic domain.

---

### J-3. Hermite interpolation symbolic support

**Problem**: Hermite/Catmull-Rom is numeric-only because interval finding uses value comparisons that break the symbolic trace. Fallback is silent.

**Goal**: Either:
- (a) Implement symbolic-compatible interval finder using `metis::where` cascades for fixed breakpoint counts, or
- (b) Accept numeric-only Hermite, ensure BSpline (symbolic-compatible, C^2) covers optimization use cases, and document the fallback explicitly with a clear warning/error when symbolic mode is attempted

**Where**: `include/metis/math/Interpolate.hpp`

---

### J-4. Smooth approximation suite

**Problem**: Missing smooth approximations for common discontinuous operations that break NLP solver convergence.

**Goal**: Add to `SurrogateModel.hpp` or a new `SmoothApproximations.hpp`:
- `smooth_abs(x, hardness)` — e.g., `x * tanh(hardness * x)` or `sqrt(x^2 + eps)`
- `smooth_max(a, b, hardness)` — pairwise smooth maximum
- `smooth_min(a, b, hardness)` — pairwise smooth minimum
- `smooth_clamp(x, low, high, hardness)` — composed smooth_min/smooth_max
- `ks_max(values, rho)` — Kreisselmeier-Steinhauser constraint aggregation

**Where**: `include/metis/math/SurrogateModel.hpp` or new header
**Tests**: Verify each converges to the non-smooth version as hardness -> infinity. Test gradients exist everywhere. Numeric + symbolic.

---

### J-5. Cumulative trapezoidal integration (`cumtrapz`)

**Problem**: Not implemented. Needed for running integrals (cumulative delta-V, accumulated impulse, energy integrals).

**Goal**: Implement `cumtrapz(y, x)` returning a vector of partial sums. Support both numeric and symbolic modes.

**Where**: `include/metis/math/Calculus.hpp` or `IntegrateDiscrete.hpp`

---

### J-6. Eigendecomposition

**Problem**: No eigenvalue/eigenvector solver. Needed for stability analysis, inertia tensor principal axes, covariance analysis, active subspace discovery (I-5).

**Goal**:
- Numeric: delegate to Eigen's `SelfAdjointEigenSolver` (symmetric case) and `EigenSolver` (general case)
- Symbolic: investigate CasADi's `eig_symbolic`, or implement 3x3 closed-form for the common inertia tensor case
- Return eigenvalues (sorted) and eigenvectors

**Where**: `include/metis/math/Linalg.hpp`

---

### J-7. Code generation API

**Problem**: Experimental JIT only. No way to generate standalone C code from a `metis::Function`.

**Goal**: First-class API: `metis::Function` -> standalone C source file with no CasADi runtime dependency. Enable embedded deployment and CI validation of generated code.

**Where**: `include/metis/core/CodeGen.hpp` (new)
**Consideration**: May also need a Vulcan-level variant for full engineering models.

---

### J-8. Implicit function theorem / CasADi rootfinder sensitivities

**Problem**: Newton solver works but doesn't propagate sensitivities through implicit solutions.

**Goal**: Given `F(x, p) = 0`, solve for `x(p)` and propagate `dx/dp` through the NLP via the implicit function theorem. Essential for trim-in-the-loop optimization.

**Where**: `include/metis/math/RootFinding.hpp`

---

### J-9. Parameterized interpolation tables

**Problem**: N-D interpolation treats breakpoints and values as fixed numeric data. Can't optimize over table contents.

**Goal**: Rework `interpn` to accept `Scalar`-typed value arrays so table values (and potentially breakpoints) can be symbolic decision variables. Enables optimization over aero coefficient tables, engine maps, control schedules.

**Where**: `include/metis/math/Interpolate.hpp`

---

### J-10. Symbolic constraint semantics

**Problem**: `any()` and `all()` on symbolic expressions evaluate to boolean instead of producing constraint-like expressions.

**Goal**: Return constraint expressions (conjunctions/disjunctions) for the optimizer. Enable natural feasibility conditions in the optimization interface.

**Where**: `include/metis/math/Logic.hpp`

---

### J-11. Verify `inv_symmetric_3x3_explicit` tuple order

**Problem**: AeroSandbox-derived explicit 3x3 symmetric inverse returns elements in unverified order.

**Goal**: Verify tuple order matches Eigen column-major storage convention. Add a doc comment specifying the convention. Add a test comparing against `metis::inv()` for several symmetric matrices.

**Where**: `include/metis/math/Linalg.hpp`

---

### J-12. CasADi `map` for batch parallelization

**Problem**: No batch evaluation support. Each function call processes one input at a time.

**Goal**: Expose CasADi's `map` function for SIMD-like parallel evaluation over batches of inputs. Target use cases: batch interpolation queries, Monte Carlo derivative evaluation, parallel sensitivity analysis.

**Where**: `include/metis/core/Function.hpp`

---

## SciML-Inspired Enhancements (J-13 through J-23)

Reference: `metis/docs/saved_work/sciml_comparison_for_metis.md`

These are algorithmic ideas from Julia's SciML ecosystem. Math-level improvements only — Metis stays a dual-headed math library, not an acausal modeling framework.

### P0 — High impact, unblocks optimization quality

#### J-13. Sparsity-first derivative pipeline

**Problem**: `SparsityPattern` exists for inspection but no end-to-end pipeline exploits sparsity.

**Goal**: Detect Jacobian/Hessian sparsity from symbolic trace -> compile sparse value evaluators (only compute nonzero entries) -> cache and reuse structure across solves. Surface CasADi's internal graph coloring through Metis. Downstream gets `O(nnz)` derivative cost instead of dense `O(n^2)`.

**Where**: `include/metis/core/Sparsity.hpp` (extend), possibly new `SparseDerivatives.hpp`

---

#### J-14. Sensitivity regime switching

**Problem**: User manually chooses forward or adjoint sensitivity.

**Goal**: Automatic selection based on parameter count vs output count:
- Forward mode: few parameters (<~20), many outputs
- Adjoint mode: many parameters (hundreds+), few outputs
- Checkpointed adjoint for long-horizon trajectories: choose backsolve / interpolating / quadrature-based checkpointing based on horizon length and stiffness

**Where**: `include/metis/math/AutoDiff.hpp` (extend)

---

#### J-15. Nonlinear solver globalization

**Problem**: `NewtonSolver` has basic line search only. Brittle for aggressive flight conditions.

**Goal**: Robust fallback stack:
1. Newton with trust-region (Levenberg-Marquardt) — primary
2. Line-search Newton — secondary
3. Quasi-Newton (Broyden/BFGS) — for expensive Jacobian cases
4. Pseudo-transient continuation — last resort for highly nonlinear problems

Configurable via solver options with sane defaults for 6DOF trim.

**Where**: `include/metis/math/RootFinding.hpp` (extend)

---

#### J-16. Structural simplification passes

**Problem**: Symbolic graphs are handed to the NLP solver without structural reduction.

**Goal**: Pass pipeline on `metis::Function`:
1. Alias elimination — remove trivially equal variables
2. BLT decomposition — identify independent subsystems
3. Tearing — select minimal iteration variables in algebraic loops
4. Codegen — generate simplified residual/Jacobian evaluators

**Where**: New `include/metis/core/StructuralTransforms.hpp`
**Design question**: Pass ordering matters — alias elimination first to simplify dependency graph before BLT.

---

### P1 — Important, builds on P0

#### J-17. Hessian-vector products and second-order adjoints

**Problem**: No way to compute `H * v` without forming the full Hessian.

**Goal**: Implement via CasADi's forward-over-reverse AD. Also provide second-order adjoint pathways for `d²L/dx²` in Lagrangian-based optimization. Enables scaling to problems where the Hessian is too large to form densely.

**Where**: `include/metis/math/AutoDiff.hpp` (extend)

---

#### J-18. Linear solve backend policy

**Problem**: `metis::solve` uses QR (numeric) or CasADi default (symbolic). No configurability.

**Goal**: Swappable backends via policy objects:
- Dense (current default)
- Sparse direct (for block-sparse Jacobians from J-13)
- Iterative Krylov (GMRES/BiCGSTAB for very large systems)
- Preconditioner hooks

**Where**: `include/metis/math/Linalg.hpp` (extend) or new `LinearSolvePolicy.hpp`

---

#### J-19. Structure-preserving integrators

**Problem**: RK4/RK45 + CVODES don't preserve energy/momentum invariants for long-horizon problems.

**Goal**: Add alongside existing integrators (not replacements):
- Symplectic integrators (Stormer-Verlet, RKN) for orbital propagation
- Rosenbrock/BDF with mass matrix support for stiff constrained systems

**Where**: `include/metis/math/Integrate.hpp` and `IntegratorStep.hpp` (extend)

---

### P2 — Nice to have, advanced

#### J-20. Automatic scaling and nondimensionalization

**Problem**: Poor NLP scaling causes convergence failure when signals span orders of magnitude (1e-6 fuel flow to 1e6 position).

**Goal**: Diagnostic tool and/or automatic scaling transforms:
- Analyze symbolic magnitudes and sparsity statistics
- Suggest or apply variable/constraint scaling to Opti problem
- Warn about badly scaled variables

**Where**: `include/metis/optimization/Opti.hpp` (extend) or new `Scaling.hpp`

---

#### J-21. Structural identifiability and observability checks

**Problem**: No automated way to check if parameters are identifiable or states are observable.

**Goal**: Symbolic analysis pass on the Jacobian structure:
- Report rank deficiencies
- Identify unobservable states and unidentifiable parameters
- Suggest fixes (add sensors, constrain parameters)

**Where**: New `include/metis/core/Diagnostics.hpp`

---

#### J-22. Polynomial chaos expansion (PCE) basis construction

**Problem**: No spectral UQ capability. Monte Carlo is the only uncertainty quantification path.

**Goal**: Provide PCE primitives:
- Basis polynomials matching the Askey scheme (Hermite/Gaussian, Legendre/uniform, Jacobi/beta, Laguerre/exponential)
- Multi-dimensional basis via tensor product or total-order truncation
- Projection/regression methods for computing expansion coefficients from collocation samples
- Key payoff: PCE coefficients through CasADi MX give gradients of statistical moments w.r.t. design variables — optimization-under-uncertainty without nested Monte Carlo

**Where**: New `include/metis/math/PolynomialChaos.hpp`

---

#### J-23. Stochastic collocation quadrature nodes

**Problem**: No structured quadrature for spectral projection. Only Gauss-Kronrod for definite integrals.

**Goal**: Quadrature rules for PCE evaluation:
- Univariate: Gauss-Hermite, Gauss-Legendre, Clenshaw-Curtis for each Askey-scheme distribution
- Smolyak sparse grids for high-dimensional problems (avoids curse of dimensionality)
- Nested rule variants (Gauss-Patterson, Clenshaw-Curtis) for incremental refinement

**Where**: `include/metis/math/Integrate.hpp` (extend) or new `Quadrature.hpp`, plus `Spacing.hpp` for node generation
