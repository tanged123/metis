# Metis 🎭

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tanged123.github.io/metis/) [![Metis CI](https://github.com/tanged123/metis/actions/workflows/ci.yml/badge.svg)](https://github.com/tanged123/metis/actions/workflows/ci.yml) [![Clang-Format Check](https://github.com/tanged123/metis/actions/workflows/format.yml/badge.svg)](https://github.com/tanged123/metis/actions/workflows/format.yml) [![codecov](https://codecov.io/github/tanged123/metis/graph/badge.svg?token=0DSF7KK8W7)](https://codecov.io/github/tanged123/metis)

Metis is a C++20 header-only numerical framework that implements a dual-mode code-transformations paradigm inspired by [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox). You write physics models once using generic templates; the same code compiles to optimized native arithmetic (Eigen) for simulation **and** builds a symbolic computational graph (CasADi) for automatic differentiation and gradient-based optimization. No code duplication, no runtime dispatch overhead.

## Building & Development

### Prerequisites

**Recommended:** [Nix](https://nixos.org/) with flakes enabled -- the `flake.nix` pins every dependency.

**Manual:** CMake 3.20+, a C++20 compiler (Clang recommended), Eigen 3.4+, CasADi, GoogleTest, Ninja.

### Dev environment

```bash
# Enter the Nix dev shell (sets up compiler, Eigen, CasADi, gtest, ccache, doxygen, lcov, etc.)
nix develop

# Or use the convenience wrapper
./scripts/dev.sh
```

### Build, test, run

```bash
./scripts/build.sh              # CMake + Ninja debug build
./scripts/build.sh --release    # Release build
./scripts/test.sh               # Build (if needed) + ctest
./scripts/coverage.sh           # Build with coverage, run tests, generate lcov report
./scripts/run_examples.sh       # Build and run all 30 example programs
./scripts/verify.sh             # Full pre-push check: build + test + examples
```

Formatting is enforced via `treefmt` (clang-format, cmake-format, nixfmt):

```bash
nix fmt                         # Format everything
./scripts/install-hooks.sh      # Install pre-commit formatting hook
```

## Project Structure

```plaintext
metis/
├── docs/                        # Documentation
│   ├── user_guides/             # 18 topic guides (see Documentation below)
│   ├── patterns/                # Reusable patterns (branching, loops, hybrid optimization)
│   └── design_overview.md       # Architecture deep-dive
├── examples/                    # 30 runnable demos
│   ├── intro/                   # Getting started (numeric, energy, sparsity, printing)
│   ├── math/                    # Branching, graphs, loops, sensitivity, PCE, structural analysis
│   ├── interpolation/           # N-D tables, scattered interpolation, root finding, file-based tables
│   ├── simulation/              # ODE integration, drag, brachistochrone, hybrid, attitudes
│   └── optimization/            # Rosenbrock, drag, beam, brachistochrone, sweeps, transcription comparison
├── include/
│   └── metis/
│       ├── core/                # Type system, concepts, diagnostics, sparsity, structural analysis, IO
│       ├── math/                # Trig, calculus, autodiff, interpolation, integration, root finding,
│       │                        #   quadrature, PCE, linalg, quaternions, rotations, surrogates
│       ├── optimization/        # Opti interface, scaling, collocation, multiple shooting,
│       │                        #   pseudospectral, Birkhoff pseudospectral
│       ├── utils/               # Utility helpers (JSON)
│       ├── metis.hpp            # Main umbrella include
│       └── using.hpp            # Convenience aliases/imports
├── scripts/                     # Build, test, CI, coverage, formatting, doc generation
├── tests/                       # GoogleTest suites (core/, math/, optimization/)
├── CMakeLists.txt
└── flake.nix                    # Nix dev environment and package definition
```

## Architecture Overview

Metis is built on a **template-first traceability** paradigm. User models are templated on a generic scalar type; the framework provides two backends that satisfy the same concept constraints. The **numeric backend** maps to `double` and `Eigen::MatrixXd` -- the compiler generates assembly identical to hand-written C++. The **symbolic backend** maps to `casadi::MX` and `Eigen::Matrix<casadi::MX>` -- the same code constructs a static computational graph suitable for automatic differentiation, sparsity detection, and NLP solvers.

A **dispatch layer** in the `metis::` namespace shadows `std::` math functions and uses C++20 concepts to route calls (`metis::sin`, `metis::pow`, etc.) to the correct backend at compile time. The framework draws a strict line between **structural logic** (integers, booleans, loop bounds -- these shape the graph) and **value logic** (floating-point quantities that flow through the graph). Standard `if/else` cannot branch on symbolic values; `metis::where(condition, true_val, false_val)` compiles to a ternary in numeric mode and a `casadi::if_else` switch node in symbolic mode.

For the full design rationale, see [docs/design_overview.md](docs/design_overview.md).

## Documentation

### Organization

- **`docs/user_guides/`** -- 18 standalone guides, each covering one subsystem end-to-end.
- **`docs/patterns/`** -- Reusable coding patterns (branching, loops, hybrid optimization).
- **`docs/design_overview.md`** -- Architecture principles and type system.
- **Doxygen API docs** -- Generated from source comments.

### Generating docs

```bash
./scripts/generate_docs.sh      # Or: doxygen Doxyfile
```

Hosted on [GitHub Pages](https://tanged123.github.io/metis/).

### User guides

| Guide | Description |
|---|---|
| [numeric_computing](docs/user_guides/numeric_computing.md) | Numeric backend basics -- templates, Eigen, optimized machine code |
| [symbolic_computing](docs/user_guides/symbolic_computing.md) | Symbolic backend -- CasADi graph construction, derivatives, code generation |
| [math_functions](docs/user_guides/math_functions.md) | Dispatch layer and ADL for dual-mode math (`metis::sin`, `metis::pow`, etc.) |
| [interpolation](docs/user_guides/interpolation.md) | N-dimensional gridded interpolation in numeric and symbolic modes |
| [integration](docs/user_guides/integration.md) | ODE solvers -- RK4, RK45, Stormer-Verlet, mass-matrix, second-order systems |
| [root_finding](docs/user_guides/root_finding.md) | Newton-Raphson and bracketing solvers with globalization |
| [optimization](docs/user_guides/optimization.md) | `metis::Opti` interface for constrained NLP (IPOPT/SNOPT/QPOASES) |
| [collocation](docs/user_guides/collocation.md) | Direct collocation transcription for optimal control |
| [multiple_shooting](docs/user_guides/multiple_shooting.md) | Multiple shooting transcription via CasADi integrators (CVODES/IDAS) |
| [pseudospectral](docs/user_guides/pseudospectral.md) | Global polynomial pseudospectral transcription |
| [birkhoff_pseudospectral](docs/user_guides/birkhoff_pseudospectral.md) | Birkhoff-form pseudospectral with derivative collocation |
| [transcription_methods](docs/user_guides/transcription_methods.md) | Comparison and selection guide for all transcription methods |
| [graph_visualization](docs/user_guides/graph_visualization.md) | Visualizing computational graphs for debugging |
| [polynomial_chaos](docs/user_guides/polynomial_chaos.md) | Polynomial chaos expansion for uncertainty quantification |
| [stochastic_quadrature](docs/user_guides/stochastic_quadrature.md) | Probability-measure quadrature rules for PCE workflows |
| [sparsity](docs/user_guides/sparsity.md) | Jacobian/Hessian sparsity detection and exploitation |
| [structural_transforms](docs/user_guides/structural_transforms.md) | Alias elimination and BLT decomposition for residual systems |
| [structural_diagnostics](docs/user_guides/structural_diagnostics.md) | Structural observability analysis for state estimation |

## Features

- 🎭 **Dual-Mode Physics**: Write once, run as Numeric (Fast C++) or Symbolic (CasADi Graph).
- 🔢 **Unified Math**: Std/CasADi agnostic math functions (`metis::sin`, `metis::pow`, `metis::where`).
- ⚡ **Linear Algebra**: Eigen-based matrix operations compatible with symbolic types, policy-driven linear solvers.
- 📉 **Optimization**: High-level `Opti` interface for NLP solvers (IPOPT/SNOPT/QPOASES) with Direct Collocation, Multiple Shooting, Pseudospectral, and Birkhoff Pseudospectral transcriptions. Variable/constraint scaling diagnostics, parametric sweeps.
- 🔁 **Differentiation**: Automatic differentiation (Forward/Reverse) via CasADi, sparse Jacobian/Hessian pipelines, sensitivity regime selection, matrix-free HVPs.
- ⏱️ **Integration**: ODE solvers (`solve_ivp`), Runge-Kutta methods, Stormer-Verlet (symplectic), second-order and mass-matrix integrators, discrete integration.
- 📈 **Interpolation**: 1D, 2D, Sparse, and N-D table lookups with B-spline support.
- 📐 **Geometry**: Quaternions, Rotation Matrices, Euler Angles.
- 🔍 **Root Finding**: Multi-method globalization stack (trust-region, line-search, Broyden, pseudo-transient), implicit function builders.
- 🧠 **Surrogates**: Differentiable approximations (sigmoid, softmax, smooth abs/max/min/clamp) for discontinuous functions.
- 📊 **Stochastic Analysis**: Polynomial chaos expansion, stochastic quadrature (Gauss, Clenshaw-Curtis, Smolyak sparse grids).
- 🔬 **Structural Analysis**: Sparsity detection, alias elimination, BLT decomposition, structural observability/identifiability analysis.
- 💾 **IO**: Graph visualization, serialization, HTML export.

## Examples

The `examples/` directory contains 30 runnable demos organized by topic:

- **intro/** (4) -- Numeric basics, energy model, sparsity intro, printing
- **math/** (10) -- Branching logic, graph visualization, loops, sensitivity, PCE, linear solve policies, structural analysis
- **interpolation/** (4) -- N-D tables, scattered data, root finding, file-based table loading
- **simulation/** (5) -- Drag model, brachistochrone, hybrid systems, aircraft attitudes, smooth trajectories
- **optimization/** (6) -- Rosenbrock, drag optimization, beam deflection, brachistochrone, parametric sweeps, transcription comparison
- **integration** (1) -- RK4, RK45, Stormer-Verlet, mass-matrix integrators

Build and run all examples:

```bash
./scripts/run_examples.sh
```

## Quick Look

Write a model once, evaluate it numerically and symbolically:

```cpp
#include <metis/metis.hpp>

// Generic model -- works with double or CasADi symbolic types
auto drag(auto rho, auto v, auto S, auto Cd) {
    return 0.5 * rho * metis::pow(v, 2) * S * Cd;
}

int main() {
    // Numeric mode -- compiles to native arithmetic
    double D = drag(1.225, 50.0, 10.0, 0.02);

    // Symbolic mode -- same function builds a CasADi graph
    metis::Opti opti;
    auto v = opti.variable(50.0);
    auto D_sym = drag(1.225, v, 10.0, 0.02);

    opti.minimize(D_sym);
    opti.subject_to_bounds(v, 10, 100);
    auto sol = opti.solve();  // IPOPT under the hood
}
```

## Contributing

**Language and style:** C++20, header-only, heavily templated. Formatting is enforced by `nix fmt` (clang-format). Run `./scripts/install-hooks.sh` to auto-format on commit.

**Adding tests:** Test files live in `tests/` and mirror the `include/metis/` directory layout (`tests/core/`, `tests/math/`, `tests/optimization/`). Use GoogleTest. Run `./scripts/test.sh` to verify.

**Adding examples:** Drop a `.cpp` file in the appropriate `examples/` subdirectory and add it to `examples/CMakeLists.txt`. Run `./scripts/run_examples.sh` to verify.

**PR workflow:** Fork, branch, make changes, ensure `./scripts/verify.sh` passes (build + test + examples), open a PR against `main`.

## Inspiration & Credits

Metis is heavily inspired by **AeroSandbox**, Peter Sharpe's Python-based design optimization framework. Metis serves as a C++ implementation and extension of the "Code Transformations" paradigm pioneered by Sharpe.

- **Primary Inspiration**: [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) by Peter Sharpe.
- **Theoretical Foundation**: Sharpe, Peter D. *AeroSandbox: A Differentiable Framework for Aircraft Design Optimization*. PhD Thesis, MIT, 2024. [Read Thesis](https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/sharpe-pds-phd-AeroAstro-2024-thesis.pdf)

Metis is built upon the shoulders of giants:

- **[Eigen](https://eigen.tuxfamily.org/)**: For high-performance linear algebra and numeric storage.
- **[CasADi](https://web.casadi.org/)**: For symbolic graph generation, automatic differentiation, and optimization interfaces.
