# Pseudospectral

`metis::Pseudospectral` implements global polynomial optimal-control transcription using spectral differentiation matrices. It enforces dynamics globally via `D * X = (dt / 2) * F(X, U, t)`, where `D` is a differentiation matrix on Lobatto nodes. For smooth problems this gives spectral convergence, so high accuracy is often possible with fewer nodes than local collocation. This works in **symbolic mode** via the `metis::Opti` interface. The class lives in `<metis/optimization/Pseudospectral.hpp>`.

## Quick Start

```cpp
#include <metis/metis.hpp>

metis::Opti opti;
metis::Pseudospectral ps(opti);

metis::PseudospectralOptions opts;
opts.scheme = metis::PseudospectralScheme::LGL;
opts.n_nodes = 31;

auto [x, u, tau] = ps.setup(3, 1, 0.0, 2.0, opts);

ps.set_dynamics([](const metis::SymbolicVector& x,
                    const metis::SymbolicVector& u,
                    const metis::SymbolicScalar& t) {
    metis::SymbolicVector dxdt(3);
    dxdt(0) = x(2) * metis::sin(u(0));
    dxdt(1) = -x(2) * metis::cos(u(0));
    dxdt(2) = 9.81 * metis::cos(u(0));
    return dxdt;
});

ps.add_dynamics_constraints();
ps.set_initial_state(metis::NumericVector{{0.0, 10.0, 0.001}});
ps.set_final_state(0, 10.0);
ps.set_final_state(1, 5.0);

opti.minimize(/* objective */);
auto sol = opti.solve();
```

## Core API

| Method | Description |
|--------|-------------|
| `Pseudospectral(opti)` | Construct with a `metis::Opti` instance |
| `setup(n_states, n_controls, t0, tf, opts)` | Create decision variables and time grid |
| `set_dynamics(ode)` | Set the ODE function: `(x, u, t) -> dxdt` |
| `add_dynamics_constraints()` | Apply spectral differentiation matrix constraints |
| `add_defect_constraints()` | Alias for `add_dynamics_constraints()` |
| `set_initial_state(x0)` | Set initial boundary condition |
| `set_final_state(xf)` | Set final boundary condition |
| `n_nodes()` | Number of collocation nodes |
| `time_grid()` | Normalized time grid `[0, 1]` |
| `diff_matrix()` | Access the differentiation matrix `D` |
| `quadrature_weights()` | Access the quadrature weight vector |
| `quadrature(integrand)` | Compute weighted integral of an integrand vector |

**Supported schemes:**

```cpp
enum class PseudospectralScheme {
    LGL, // Legendre-Gauss-Lobatto
    CGL  // Chebyshev-Gauss-Lobatto
};
```

Both include endpoints, so boundary-condition setup matches `DirectCollocation`.

## Usage Patterns

### Free Final Time

```cpp
auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);
auto [x, u, tau] = ps.setup(3, 1, 0.0, T, opts);

ps.set_dynamics(my_ode);
ps.add_dynamics_constraints();
ps.set_initial_state(x0);
ps.set_final_state(0, xf_x);
ps.set_final_state(1, xf_y);

opti.minimize(T);
auto sol = opti.solve();
```

### Quadrature for Running Costs

Use `quadrature()` for Lagrange objectives:

```cpp
metis::SymbolicVector integrand(ps.n_nodes());
for (int k = 0; k < ps.n_nodes(); ++k) {
    integrand(k) = u(k, 0) * u(k, 0);
}
opti.minimize(ps.quadrature(integrand));
```

This computes `J = (dt / 2) * sum_i w_i * L_i` where `w_i` are the scheme quadrature weights.

### Unified Comparison Example

The file `examples/optimization/transcription_comparison_demo.cpp` runs Direct Collocation, Multiple Shooting, Pseudospectral, and Birkhoff Pseudospectral on the same brachistochrone problem and prints a side-by-side comparison with performance metrics and a convergence study.

## Advanced Usage

### Accessing Internal Matrices

```cpp
const auto &D = ps.diff_matrix();
const auto &w = ps.quadrature_weights();
```

### Notes

- `time_grid()` returns normalized `[0, 1]` values for API consistency with `DirectCollocation`.
- Internally, nodes and `D` are built on `[-1, 1]` and scaled correctly in constraints/objectives.
- Pseudospectral methods are strongest for smooth trajectories; sharp bang-bang controls can need mesh refinement or alternative transcription.

## See Also

- [Transcription Methods Guide](transcription_methods.md) -- Comparison of all four transcription methods
- [Direct Collocation Guide](collocation.md) -- Local polynomial defect-based transcription
- [Multiple Shooting Guide](multiple_shooting.md) -- Integrator-based transcription
- [Birkhoff Pseudospectral Guide](birkhoff_pseudospectral.md) -- Birkhoff-form with integration matrix
- [transcription_comparison_demo.cpp](../../examples/optimization/transcription_comparison_demo.cpp) -- Unified comparison example
- [Pseudospectral.hpp](../../include/metis/optimization/Pseudospectral.hpp) -- API reference
