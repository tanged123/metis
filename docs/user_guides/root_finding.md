# Root Finding

Metis provides nonlinear root-finding utilities in `RootFinding.hpp` for solving `F(x) = 0` systems. The API serves two distinct roles: numeric nonlinear solves with a globalization stack for difficult residual systems, and symbolic implicit solves that stay differentiable inside CasADi graphs. Both modes work with dense column-vector inputs and outputs of matching dimensions.

## Quick Start

```cpp
#include <metis/metis.hpp>

// Define a residual symbolically
auto x = metis::sym("x");
metis::Function F("f", {x}, {x * x - 2.0});

// Solve F(x) = 0 starting from x0 = 1
Eigen::VectorXd x0(1);
x0 << 1.0;

auto result = metis::rootfinder<double>(F, x0);
std::cout << "Root: " << result.x.transpose() << std::endl;  // ~1.4142
```

## Core API

*   **`metis::rootfinder<double>(F, x0, opts)`**: Numeric root-finding with automatic globalization fallback.
*   **`metis::NewtonSolver(F, opts)`**: Persistent solver for repeated solves of the same residual system.
*   **`metis::create_implicit_function(G, x_guess, opts)`**: Create a differentiable implicit function `x(p)` from `G(x, p) = 0`.
*   **`metis::rootfinder<metis::SymbolicScalar>()`**: CasADi-backed symbolic solve node.
*   **`metis::RootFinderOptions`**: Configuration for tolerance, max iterations, strategy, and pseudo-transient parameters.
*   **`metis::RootResult<double>`**: Result struct with `x`, `converged`, `method`, `iterations`, `residual_norm`, `step_norm`, and `message`.

## Usage Patterns

### Numeric Solves

`rootfinder<double>()` and `NewtonSolver::solve()` use Metis's own globalization stack. The default is `RootSolveStrategy::Auto`, which tries:

1. Trust-region Newton (Levenberg-Marquardt)
2. Line-search Newton
3. Quasi-Newton Broyden updates
4. Pseudo-transient continuation

This is designed to be more robust than a bare Newton step when the initial Jacobian is poor, singular, or badly scaled.

### Strategy Selection

```cpp
metis::RootFinderOptions opts;
opts.strategy = metis::RootSolveStrategy::Auto;
```

Available numeric strategies:

| Strategy | Use Case |
|----------|----------|
| `Auto` | Default. Best general-purpose option for trim and nonlinear residual solves. |
| `TrustRegionNewton` | Good first choice when Jacobians are reliable and you want fast local convergence. |
| `LineSearchNewton` | Useful when exact Newton steps are too aggressive but the Jacobian is still trustworthy. |
| `QuasiNewtonBroyden` | Useful when recomputing exact Jacobians is expensive and secant updates are acceptable. |
| `PseudoTransientContinuation` | Last-resort path for highly nonlinear or singular-start problems. |

### Result Diagnostics

Numeric solves return a `RootResult<double>` with method and convergence diagnostics:

```cpp
auto result = metis::rootfinder<double>(F, x0);

if (result.converged) {
    std::cout << result.x.transpose() << "\n";
    std::cout << result.iterations << "\n";
    std::cout << result.residual_norm << "\n";
}
```

Useful fields:

- `x`: final iterate
- `converged`: whether the residual tolerance was met
- `method`: which stage actually converged
- `iterations`: total iterations used across the stack
- `residual_norm`: final infinity norm of the residual
- `step_norm`: last accepted step infinity norm
- `message`: short status string

### Automatic Fallback

This example starts from a singular Jacobian, so `Auto` falls through to pseudo-transient continuation:

```cpp
auto x = metis::sym("x");
metis::Function F("singular_start", {x}, {x * x - 1.0});

Eigen::VectorXd x0(1);
x0 << 0.0;

metis::RootFinderOptions opts;
opts.max_iter = 60;
opts.pseudo_transient_dt0 = 0.1;

auto result = metis::rootfinder<double>(F, x0, opts);
```

### Persistent Solver Reuse

When you will solve the same residual system multiple times, use `NewtonSolver` so the residual and Jacobian kernels are compiled once:

```cpp
metis::RootFinderOptions opts;
opts.strategy = metis::RootSolveStrategy::LineSearchNewton;

metis::NewtonSolver solver(F, opts);

auto res1 = solver.solve(x0_a);
auto res2 = solver.solve(x0_b);
```

### Differentiable Implicit Solves

`create_implicit_function()` is the right tool when the solve itself must remain inside a symbolic graph:

```cpp
// G(x, p) = 0  ->  x(p)
auto x = metis::sym("x");
auto p = metis::sym("p");
metis::Function G("G", {x, p}, {x * x - p});

Eigen::VectorXd guess(1);
guess << 1.0;

auto x_of_p = metis::create_implicit_function(G, guess);
auto dxdp = metis::jacobian(x_of_p(p)[0](0), p);
```

Metis keeps this path on CasADi's differentiable `newton` rootfinder so exact implicit sensitivities are preserved.

### Non-Default Implicit Slots

If the unknown state is not the first input or the residual is not the first output, use `ImplicitFunctionOptions`:

```cpp
metis::ImplicitFunctionOptions implicit_opts;
implicit_opts.implicit_input_index = 1;
implicit_opts.implicit_output_index = 1;

auto implicit = metis::create_implicit_function(G, guess, {}, implicit_opts);
```

### Symbolic `rootfinder`

`rootfinder<metis::SymbolicScalar>()` also remains CasADi-rootfinder-backed. Use it when you want a symbolic solve node directly, but for optimization workflows the higher-level `create_implicit_function()` wrapper is usually cleaner.

## See Also

- [Symbolic Computing Guide](symbolic_computing.md) - Working with symbolic expressions and differentiation
- [Optimization Guide](optimization.md) - Nonlinear optimization with `metis::Opti`
- [`examples/interpolation/rootfinding_demo.cpp`](../../examples/interpolation/rootfinding_demo.cpp) - Runnable root-finding example
- [`include/metis/math/RootFinding.hpp`](../../include/metis/math/RootFinding.hpp) - Full API implementation
