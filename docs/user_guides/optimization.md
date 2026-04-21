# Optimization

Metis provides a high-level C++ interface to constrained nonlinear optimization through `metis::Opti`, wrapping CasADi/IPOPT. It allows you to define optimization variables, write objectives and constraints using standard C++ syntax, and reuse your physicist-written simulation models directly in the optimization loop. This works in symbolic mode, with seamless interop with numeric constants.

## Quick Start

```cpp
#include <metis/metis.hpp>

metis::Opti opti;

auto x = opti.variable(0.0);
auto y = opti.variable(0.0);

// Minimize the Rosenbrock function
auto f = metis::pow(1 - x, 2) + 100 * metis::pow(y - metis::pow(x, 2), 2);
opti.minimize(f);

auto sol = opti.solve({.max_iter = 500, .verbose = false});
std::cout << "x=" << sol.value(x) << ", y=" << sol.value(y) << std::endl;
```

## Core API

*   **`metis::Opti`**: The optimization problem builder.
*   **`opti.variable(initial_guess)`**: Create a scalar decision variable.
*   **`opti.variable(n, initial_guess)`**: Create a vector decision variable of length `n`.
*   **`opti.parameter(value)`**: Create a parameter that can be changed between solves.
*   **`opti.minimize(expr)`**: Set the objective function to minimize.
*   **`opti.subject_to(constraint)`**: Add an equality (`==`) or inequality (`<=`, `>=`) constraint.
*   **`opti.subject_to_bounds(var, lb, ub)`**: Set variable bounds (more efficient than general constraints).
*   **`opti.subject_to_lower(var, lb)`** / **`opti.subject_to_upper(var, ub)`**: One-sided bounds.
*   **`opti.solve(options)`**: Execute IPOPT and return a solution object.
*   **`sol.value(var)`**: Retrieve the optimal value of a variable from the solution.

## Usage Patterns

### The Rosenbrock Benchmark

The Rosenbrock function (or "banana function") is a classic test for optimization algorithms.
Code reference: [`examples/optimization/rosenbrock.cpp`](../../examples/optimization/rosenbrock.cpp)

```cpp
metis::Opti opti;

auto x = opti.variable(0.0);
auto y = opti.variable(0.0);

auto f = metis::pow(1 - x, 2) + 100 * metis::pow(y - metis::pow(x, 2), 2);
opti.minimize(f);

// Add constraints
opti.subject_to(metis::pow(x, 2) + metis::pow(y, 2) <= 2.0);

auto sol = opti.solve({.max_iter = 500, .verbose = false});
double x_star = sol.value(x);
double y_star = sol.value(y);
std::cout << "Optimal Solution: x=" << x_star << ", y=" << y_star << std::endl;
```

### "Write Once, Use Everywhere" (C++20 Style)

The true power of Metis is reusing your existing physics code. With C++20 **Abbreviated Function Templates** (`auto` parameters), this is seamless.

Code reference: [`examples/optimization/drag_optimization.cpp`](../../examples/optimization/drag_optimization.cpp)

The shared physics function works for `double`, `SymbolicScalar`, or a mix of both:

```cpp
auto compute_drag(auto rho, auto v, auto S,
                  auto Cd0, auto k, auto Cl, auto Cl0) {
    auto q = 0.5 * rho * metis::pow(v, 2.0);
    auto Cd = Cd0 + k * metis::pow(Cl - Cl0, 2.0);
    return q * S * Cd;
}
```

Using it in optimization requires no casting or explicit templates:

```cpp
metis::Opti opti;

auto V = opti.variable(50.0);
auto Cl = opti.variable(0.5);

const double rho = 1.225;
const double S = 16.0;

auto D = compute_drag(rho, V, S, 0.02, 0.04, Cl, 0.1);

opti.minimize(D);
opti.subject_to_bounds(V, 20.0, 150.0);
```

Why this matters:
*   **Zero Boilerplate**: No `template <typename T>` syntax required for users.
*   **Type Safety**: Still statically checked at compile time.
*   **Performance**: Compiles down to optimized machine code (Numeric mode) or efficient graph construction (Symbolic mode).

### Bounds Handling

Metis provides explicit helpers for variable bounds, which is more efficient than general constraints for the solver.

```cpp
// Scalar variable bounds
auto x = opti.variable(0.5);
opti.subject_to_bounds(x, 0.0, 1.0);  // 0 <= x <= 1
opti.subject_to_lower(x, 0.0);        // x >= 0
opti.subject_to_upper(x, 1.0);        // x <= 1

// Vector variable bounds (applies to all elements)
auto vec = opti.variable(10, 0.0);
opti.subject_to_bounds(vec, -5.0, 5.0);
```

### Parametric Sweeps

Run the same optimization across a range of parameter values with automatic warm-starting:

```cpp
metis::Opti opti;

auto rho = opti.parameter(1.225);
auto V = opti.variable(50.0);

const double S = 16.0;    // Reference area [m^2]
const double Cd0 = 0.02;  // Zero-lift drag coefficient

auto drag = 0.5 * rho * V * V * S * Cd0;
opti.minimize(drag);
opti.subject_to(V >= 10.0);

std::vector<double> rho_values = {1.2, 1.0, 0.8, 0.6};
auto result = opti.solve_sweep(rho, rho_values);

for (size_t i = 0; i < result.size(); ++i) {
    if (result.converged[i]) {
        std::cout << "rho=" << result.param_values[i]
                  << " V*=" << result.solutions[i]->value(V) << "\n";
    }
}
```

See the full example: [`examples/optimization/parametric_sweep.cpp`](../../examples/optimization/parametric_sweep.cpp)

## Advanced Usage

### Scaling Diagnostics and Nondimensionalization

Poor scaling is a common failure mode for nonlinear programs. `metis::Opti` exposes three
practical tools:

- Variable scales can still be supplied explicitly through `variable(..., scale, ...)`.
- If the initial guess is zero, finite bounds are now used to infer a more sensible default scale.
- Objectives and constraints can be scaled explicitly, and `analyze_scaling()` reports suspicious magnitudes before solve.

```cpp
metis::Opti opti;

auto x = opti.variable(0.0, std::nullopt, -1e6, 1e6);

opti.subject_to(x == 1e6, 1e6);
opti.minimize(metis::pow(x - 1e6, 2), 1e12);

auto report = opti.analyze_scaling();
if (report.has_issues()) {
    for (const auto& issue : report.issues) {
        std::cout << issue.label << ": " << issue.message << "\n";
    }
}
```

The report summarizes:

- variable block scales and normalized initial guesses
- constraint row magnitudes versus applied linear scales
- objective magnitude versus applied objective scale
- Jacobian sparsity density for the current NLP

This is intended as a pre-solve diagnostic pass, not a full automatic reformulation engine.

### Solution Persistence & Warm Starting

Metis allows you to save optimization results to JSON and use them to warm-start subsequent runs. This is crucial for complex problems where a good initial guess can significantly reduce solve time.

**Saving Results:**

```cpp
auto sol = opti.solve();

std::map<std::string, metis::SymbolicScalar> vars;
vars["x"] = x;
vars["y"] = y;
sol.save("solution.json", vars);
```

**Loading & Warm Starting:**

```cpp
try {
    auto cache = metis::OptiSol::load("solution.json");

    double x_init = cache.count("x") ? cache["x"][0] : 0.0;

    auto x = opti.variable(x_init);
} catch (...) {
    // Handle cold start (file not found, etc.)
}
```

## See Also

- [Symbolic Computing Guide](symbolic_computing.md) - Building symbolic expressions for optimization
- [Interpolation Guide](interpolation.md) - Using interpolation as surrogate models in optimization
- [`examples/optimization/rosenbrock.cpp`](../../examples/optimization/rosenbrock.cpp) - Rosenbrock benchmark
- [`examples/optimization/brachistochrone_opti.cpp`](../../examples/optimization/brachistochrone_opti.cpp) - Trajectory optimization example
- [`include/metis/optimization/Opti.hpp`](../../include/metis/optimization/Opti.hpp) - `metis::Opti` implementation
