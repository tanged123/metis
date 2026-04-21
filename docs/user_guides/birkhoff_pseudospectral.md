# Birkhoff Pseudospectral

`metis::BirkhoffPseudospectral` is a Birkhoff-form pseudospectral transcription where state-derivative variables are collocated directly and states are recovered through a linear integration matrix. Unlike classical pseudospectral methods that use a dense differentiation matrix `D*X`, the Birkhoff form keeps dense coupling mostly in linear constraints (`X = x_a * 1 + B * V`) while dynamics constraints are pointwise (`V_i = (dt/2) * f(X_i, U_i, t_i)`). This gives improved numerical conditioning at higher node counts. Works in **symbolic mode** via the `metis::Opti` interface. The class lives in `<metis/optimization/BirkhoffPseudospectral.hpp>`.

## Quick Start

```cpp
#include <metis/metis.hpp>

metis::Opti opti;
metis::BirkhoffPseudospectral bk(opti);

auto [x, u, tau] = bk.setup(
    3, 1, 0.0, 2.0,
    {.scheme = metis::BirkhoffScheme::LGL, .n_nodes = 31}
);

bk.set_dynamics([](const metis::SymbolicVector& x,
                    const metis::SymbolicVector& u,
                    const metis::SymbolicScalar& t) {
    metis::SymbolicVector dxdt(3);
    dxdt(0) = x(2) * metis::sin(u(0));
    dxdt(1) = -x(2) * metis::cos(u(0));
    dxdt(2) = 9.81 * metis::cos(u(0));
    return dxdt;
});

bk.add_dynamics_constraints();
bk.set_initial_state(metis::NumericVector{{0.0, 10.0, 0.001}});
bk.set_final_state(0, 10.0);
bk.set_final_state(1, 5.0);

opti.minimize(/* objective */);
auto sol = opti.solve();
```

## Core API

| Method | Description |
|--------|-------------|
| `BirkhoffPseudospectral(opti)` | Construct with a `metis::Opti` instance |
| `setup(n_states, n_controls, t0, tf, opts)` | Create decision variables and time grid |
| `set_dynamics(ode)` | Set the ODE function: `(x, u, t) -> dxdt` |
| `add_dynamics_constraints()` | Apply Birkhoff dynamics and state-recovery constraints |
| `add_defect_constraints()` | Alias for `add_dynamics_constraints()` |
| `set_initial_state(x0)` | Set initial boundary condition |
| `set_final_state(xf)` | Set final boundary condition |
| `n_nodes()` | Number of collocation nodes |
| `time_grid()` | Normalized time grid `[0, 1]` |
| `integration_matrix()` | Access the Birkhoff integration matrix `B` |
| `quadrature_weights()` | Access the quadrature weight vector |
| `quadrature(integrand)` | Compute weighted integral of an integrand vector |
| `virtual_vars()` | Access the virtual (state-derivative) decision variables `V` |

### Core Formulation

For nodes on `[-1, 1]`:

- **Dynamics collocation** (pointwise):
```
V_i = (dt / 2) * f(X_i, U_i, t_i)
```
- **State recovery** (linear):
```
X = x_a * 1 + B * V
```

`B` is the Birkhoff integration matrix with entries:
```
B_ij = integral_{tau_0}^{tau_i} ell_j(s) ds
```

## Usage Patterns

### Free Final Time

```cpp
auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);
auto [x, u, tau] = bk.setup(
    3, 1, 0.0, T,
    {.scheme = metis::BirkhoffScheme::LGL, .n_nodes = 31}
);

bk.set_dynamics(my_ode);
bk.add_dynamics_constraints();
bk.set_initial_state(x0);
bk.set_final_state(0, xf_x);
bk.set_final_state(1, xf_y);

opti.minimize(T);
auto sol = opti.solve();
```

### Quadrature for Running Costs

```cpp
metis::SymbolicVector integrand(bk.n_nodes());
for (int k = 0; k < bk.n_nodes(); ++k) {
    integrand(k) = u(k, 0) * u(k, 0);
}
opti.minimize(bk.quadrature(integrand));
```

### Unified Comparison Example

The file `examples/optimization/transcription_comparison_demo.cpp` compares Direct Collocation, Multiple Shooting, classical Pseudospectral, and Birkhoff Pseudospectral on the same problem with performance and convergence output.

## Advanced Usage

### Accessing Internal Matrices and Variables

```cpp
const auto &B = bk.integration_matrix();
const auto &w = bk.quadrature_weights();
const auto &V = bk.virtual_vars();
```

The virtual variables `V` represent the scaled state derivatives at each node. They are additional decision variables beyond `X` and `U`, which is why the Birkhoff formulation has more decision variables than classical pseudospectral but trades that for better conditioning and pointwise nonlinear structure.

## See Also

- [Transcription Methods Guide](transcription_methods.md) -- Comparison of all four transcription methods
- [Pseudospectral Guide](pseudospectral.md) -- Classical pseudospectral with differentiation matrix
- [Direct Collocation Guide](collocation.md) -- Local polynomial defect-based transcription
- [Multiple Shooting Guide](multiple_shooting.md) -- Integrator-based transcription
- [transcription_comparison_demo.cpp](../../examples/optimization/transcription_comparison_demo.cpp) -- Unified comparison example
- [BirkhoffPseudospectral.hpp](../../include/metis/optimization/BirkhoffPseudospectral.hpp) -- API reference
