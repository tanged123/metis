# SciML-to-Metis Comparison (6DOF + Real-Time Gradients)

**Date**: 2026-03-03  
**Purpose**: Keep a compact reference for future brainstorming sessions on Metis architecture direction.

## Scope

Target downstream use case:
- Engineering utility libraries consume Metis.
- 6DOF flight simulation uses Metis symbolic/numeric dual mode.
- Real-time gradient-based optimization (AeroSandbox-like workflows).

## Architecture Boundary (Current Decision)

Metis should stay dual-headed math infrastructure, not a domain hierarchy host.

Metis owns:
- Symbolic/numeric core and transforms.
- Derivatives, sparsity, and linear algebra backends.
- Solver-facing callable contracts and generated-code/cache plumbing.

Downstream repos own:
- Domain object hierarchies (`ODEProblem`, `OCP`, mission phases, vehicle systems).
- Composition and orchestration logic for application-specific workflows.
- Runtime policy for scheduling, mode management, and closed-loop control integration.

## Local Julia Reference Snapshot

Cloned under `reference/Julia/SciML`:

| Repo | Snapshot |
|---|---|
| `ModelingToolkit.jl` | `79d6cf5` |
| `Symbolics.jl` | `8063ea8` |
| `SciMLBase.jl` | `1822262` |
| `DifferentialEquations.jl` | `bfd3144` |
| `OrdinaryDiffEq.jl` | `011191c` |
| `SciMLSensitivity.jl` | `7a31d30` |
| `NonlinearSolve.jl` | `3256777` |
| `Optimization.jl` | `1fb6e4f` |
| `LinearSolve.jl` | `f83d323` |

## Math Ideas To Borrow From SciML

Focus here is math and numerics inside Metis, not simulator/runtime orchestration.

| Idea (Julia source) | What to take | Why it matters for 6DOF gradients |
|---|---|---|
| Structural simplification (`ModelingToolkit`) | Alias elimination + BLT decomposition + tearing before numerical solve | Shrinks coupled algebraic systems and exposes independent blocks/parallel work |
| DAE index handling (`ModelingToolkit`) | Pantelides-style index reduction and consistent initialization strategy | Better conditioning and fewer startup failures on constrained dynamics |
| Sparsity programming (`Symbolics`) | Automatic Jacobian/Hessian sparsity detection + sparse value kernels (`sparsejacobian_vals`, `sparsehessian_vals`) | Real-time derivatives at `O(nnz)` cost instead of dense cost |
| Derivative structure traits (`Symbolics` + ADTypes) | Linearity/affinity/structure flags attached to expressions | Lets transcriptions choose cheaper linearized or partially condensed math paths |
| Sensitivity regime switching (`SciMLSensitivity`) | Rules for forward vs adjoint vs mixed methods based on parameter count/stiffness | Stable low-latency gradients across tiny and large parameter sets |
| Checkpointed adjoints (`SciMLSensitivity`) | Memory-stability trade between backsolve/interpolating/gauss/quadrature styles | Practical gradients for long horizons without exploding memory |
| Manual VJP injection (`SciMLSensitivity`) | Allow hand-tuned VJPs for difficult kernels | Control over hot-path derivative performance and robustness |
| Globalization stack (`NonlinearSolve`) | Trust-region + line-search + quasi-Newton + pseudo-transient fallbacks | Reduces nonlinear solve brittleness in aggressive maneuvers |
| Linear solve policy (`LinearSolve`) | Swap dense/sparse/iterative solves via small policy objects | Better conditioning/performance portability without rewriting math code |
| Structure-aware integration (`OrdinaryDiffEq`) | Symplectic/RKN for mechanics, Rosenbrock/BDF for stiff mass-matrix forms | Preserves invariants where needed; handles stiff/constrained regimes where needed |

## What To Copy First (Math Priority)

### P0
- Structural transform pipeline: alias elimination -> BLT -> tearing -> simplified residual/Jacobian generation.
- Sparsity-first derivative pipeline: detect sparsity once, compile sparse Jacobian/Hessian value evaluators, reuse structure.
- Sensitivity policy engine: forward for small parameter dimension, adjoint for large, with checkpoint heuristics for long horizons.
- Nonlinear globalization defaults: Newton with trust-region and line-search safeguards, plus quasi-Newton fallback.

### P1
- DAE index reduction and consistent-initialization math for constrained systems.
- Hessian-vector and second-order adjoint pathways for Newton-Krylov/SQP-style methods.
- Linear solve backend policy with matrix-free Krylov and preconditioner hooks.
- Structure-preserving integrator options for long-horizon energy behavior.

### P2
- Automatic scaling/nondimensionalization helpers from symbolic magnitudes and sparsity statistics.
- Structural identifiability/observability checks as symbolic preflight diagnostics.

## Out Of Scope For Metis (Icarus-Owned)

- Problem hierarchy classes and mission/system orchestration.
- Runtime event scheduling and simulator loop semantics.
- High-level solve workflow composition across scenario phases.

## Immediate Design Questions To Revisit In Future Sessions

1. What transform ordering gives the best tradeoff for Metis: alias -> BLT -> tearing -> CSE/codegen?
2. What parameter-count and stiffness thresholds should switch sensitivity modes?
3. Which nonlinear fallback sequence is most robust for flight trim and aggressive transients?
4. Which invariants should be explicitly preserved in 6DOF integration (energy, momentum, constraints)?
5. How do we keep sparse patterns stable across mode changes so compiled kernels remain reusable?

## Suggested Session Starters

- "Design Metis structural transform passes (alias, BLT, tearing) and define pass ordering."
- "Prototype sparse Jacobian/Hessian value kernels with fixed sparsity structure reuse."
- "Draft sensitivity mode switching heuristics for 6DOF OCP/NMPC workloads."
- "Evaluate trust-region + line-search globalization defaults for Metis nonlinear solves."
