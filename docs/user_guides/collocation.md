# Direct Collocation

Direct collocation transforms continuous-time optimal control problems into large sparse NLPs by discretizing time into nodes and enforcing dynamics at each segment using defect constraints. Metis provides the `metis::DirectCollocation` class in `<metis/optimization/Collocation.hpp>`. It supports trapezoidal (2nd order) and Hermite-Simpson (4th order) schemes and works in **symbolic mode** via the `metis::Opti` interface.

## Quick Start

```cpp
#include <metis/metis.hpp>

metis::Opti opti;
metis::DirectCollocation dc(opti);

auto [x, u, tau] = dc.setup(
    2, 1, 0.0, 2.0,
    {.scheme = metis::CollocationScheme::HermiteSimpson, .n_nodes = 31}
);

dc.set_dynamics([](const auto& x, const auto& u, const auto& t) {
    metis::SymbolicVector dxdt(2);
    dxdt(0) = x(1);
    dxdt(1) = u(0);
    return dxdt;
});

dc.add_defect_constraints();
dc.set_initial_state(metis::NumericVector{{0.0, 0.0}});
dc.set_final_state(metis::NumericVector{{1.0, 0.0}});

// Minimize control effort: sum of u^2 at each node
metis::SymbolicScalar obj = 0;
for (int k = 0; k < dc.n_nodes(); ++k)
    obj = obj + u(k, 0) * u(k, 0);
opti.minimize(obj);
auto sol = opti.solve();
```

## Core API

| Method | Description |
|--------|-------------|
| `DirectCollocation(opti)` | Construct with a `metis::Opti` instance |
| `setup(n_states, n_controls, t0, tf, opts)` | Create decision variables and time grid |
| `set_dynamics(ode)` | Set the ODE function: `(x, u, t) -> dxdt` |
| `add_defect_constraints()` | Apply collocation defect constraints |
| `add_dynamics_constraints()` | Unified alias for `add_defect_constraints()` |
| `set_initial_state(x0)` | Set initial boundary condition (full vector or per-index) |
| `set_final_state(xf)` | Set final boundary condition (full vector or per-index) |
| `n_nodes()` | Number of collocation nodes |
| `time_grid()` | Normalized time grid `[0, 1]` |

**Collocation schemes:**

| Scheme | Order | Description |
|--------|-------|-------------|
| `CollocationScheme::Trapezoidal` | 2nd | Uses average of endpoint derivatives |
| `CollocationScheme::HermiteSimpson` | 4th | Uses cubic interpolation with midpoint |

**Trapezoidal:**
```text
x[k+1] - x[k] = 0.5 * h * (f[k] + f[k+1])
```

**Hermite-Simpson:**
```text
x_mid = 0.5*(x[k] + x[k+1]) + h/8*(f[k] - f[k+1])
x[k+1] - x[k] = h/6 * (f[k] + 4*f_mid + f[k+1])
```

## Usage Patterns

### Brachistochrone Example

The brachistochrone problem finds the fastest path for a bead sliding under gravity.

**Dynamics:**
```cpp
metis::SymbolicVector brachistochrone_dynamics(
    const metis::SymbolicVector &state,    // [x, y, v]
    const metis::SymbolicVector &control,  // [theta]
    const metis::SymbolicScalar &t)
{
    metis::SymbolicScalar v = state(2);
    metis::SymbolicScalar theta = control(0);

    metis::SymbolicVector dxdt(3);
    dxdt(0) = v * metis::sin(theta);    // x' = v*sin(theta)
    dxdt(1) = -v * metis::cos(theta);   // y' = -v*cos(theta)
    dxdt(2) = 9.81 * metis::cos(theta); // v' = g*cos(theta)
    return dxdt;
}
```

**Setup:**
```cpp
metis::Opti opti;
metis::DirectCollocation dc(opti);

auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

auto [x, u, tau] = dc.setup(3, 1, 0.0, T,
    {.scheme = metis::CollocationScheme::HermiteSimpson, .n_nodes = 31});

dc.set_dynamics(brachistochrone_dynamics);
dc.add_defect_constraints();

dc.set_initial_state(metis::NumericVector{{0.0, 10.0, 0.001}});
dc.set_final_state(0, 10.0);  // Final x
dc.set_final_state(1, 5.0);   // Final y

opti.minimize(T);  // Minimize time
auto sol = opti.solve();
```

**Result:** T* = 1.8016s (matches Dymos reference: 1.8019s, error < 0.02%)

### Free vs Fixed Final Time

**Fixed time:** Pass `double` for `tf`
```cpp
dc.setup(n_states, n_controls, 0.0, 2.0, opts);
```

**Free time:** Pass `SymbolicScalar` for `tf`
```cpp
auto T = opti.variable(2.0);  // Decision variable
dc.setup(n_states, n_controls, 0.0, T, opts);
opti.minimize(T);  // Minimize time
```

### Manual vs DirectCollocation Comparison

**Manual collocation** (50+ lines):
```cpp
for (int i = 0; i < N - 1; ++i) {
    metis::SymbolicVector state_i(3), state_ip1(3);
    state_i << x(i), y(i), v(i);
    state_ip1 << x(i+1), y(i+1), v(i+1);
    auto f_i = ode(state_i, theta(i));
    auto f_ip1 = ode(state_ip1, theta(i+1));
    opti.subject_to(x(i+1) - x(i) == 0.5 * dt * (f_i(0) + f_ip1(0)));
    // ... repeat for each state ...
}
```

**DirectCollocation** (~10 lines):
```cpp
dc.set_dynamics(ode);
dc.add_defect_constraints();
dc.set_initial_state(x0);
dc.set_final_state(xf);
```

### Unified Comparison Example

The file `examples/optimization/transcription_comparison_demo.cpp` solves the same brachistochrone with Direct Collocation, Multiple Shooting, Pseudospectral, and Birkhoff Pseudospectral, printing single-grid performance stats and a multi-grid convergence table.

### When to Use

| Problem | Use Collocation? |
|---------|-----------------|
| Trajectory optimization | Yes |
| Minimum-time problems | Yes (free tf) |
| Path constraints | Yes |
| Stiff systems | Yes (implicit) |
| Bang-bang control | Consider multiple shooting |

## See Also

- [Transcription Methods Guide](transcription_methods.md) -- Comparison of all four transcription methods
- [Multiple Shooting Guide](multiple_shooting.md) -- Alternative transcription via numerical integration
- [Pseudospectral Guide](pseudospectral.md) -- Global polynomial transcription
- [Birkhoff Pseudospectral Guide](birkhoff_pseudospectral.md) -- Birkhoff-form transcription
- [transcription_comparison_demo.cpp](../../examples/optimization/transcription_comparison_demo.cpp) -- Unified comparison example
- [Collocation.hpp](../../include/metis/optimization/Collocation.hpp) -- API reference
