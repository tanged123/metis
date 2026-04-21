# Multiple Shooting

`metis::MultipleShooting` provides a transcription method for optimal control problems that enforces continuity via high-accuracy numerical integration (using CasADi's integrator interface, e.g., CVODES or IDAS). It divides the time horizon into intervals with piecewise-constant controls and connects them with continuity constraints. This works in **symbolic mode** via the `metis::Opti` interface. The class lives in `<metis/optimization/MultiShooting.hpp>`.

## Quick Start

```cpp
#include <metis/metis.hpp>

metis::Opti opti;
metis::MultipleShooting ms(opti);

metis::MultiShootingOptions opts;
opts.n_intervals = 20;
opts.integrator = "cvodes";
opts.tol = 1e-6;

auto [x, u, tau] = ms.setup(3, 1, 0.0, 2.0, opts);

ms.set_dynamics([](const metis::SymbolicVector& x,
                    const metis::SymbolicVector& u,
                    const metis::SymbolicScalar& t) {
    metis::SymbolicVector dxdt(3);
    dxdt(0) = x(2) * metis::sin(u(0));
    dxdt(1) = -x(2) * metis::cos(u(0));
    dxdt(2) = 9.81 * metis::cos(u(0));
    return dxdt;
});

ms.add_continuity_constraints();
ms.set_initial_state(metis::NumericVector{{0.0, 10.0, 0.001}});
ms.set_final_state(0, 10.0);
ms.set_final_state(1, 5.0);

opti.minimize(/* objective */);
auto sol = opti.solve();
```

## Core API

| Method | Description |
|--------|-------------|
| `MultipleShooting(opti)` | Construct with a `metis::Opti` instance |
| `setup(n_states, n_controls, t0, tf, opts)` | Create decision variables and time grid |
| `set_dynamics(ode)` | Set the ODE function: `(x, u, t) -> dxdt` |
| `add_continuity_constraints()` | Apply integrator-based continuity constraints |
| `add_dynamics_constraints()` | Unified alias for `add_continuity_constraints()` |
| `set_initial_state(x0)` | Set initial boundary condition |
| `set_final_state(xf)` | Set final boundary condition |
| `n_intervals()` | Number of shooting intervals |
| `time_grid()` | Normalized time grid `[0, 1]` |

**`MultiShootingOptions`** exposes:
- `n_intervals`: number of shooting intervals
- `integrator`: integrator type (`"cvodes"`, `"rk"`, `"idas"`)
- `tol`: integrator tolerance

### Advantages over Direct Collocation
- **High Accuracy**: Uses variable-step/variable-order integrators instead of fixed-order polynomials.
- **Stiffness Handling**: Better suited for stiff systems where low-order schemes fail.
- **Sparse Structure**: Retains the block-sparse structure of the NLP.

### Disadvantages
- **Cost**: Evaluating integrator sensitivities can be computationally expensive compared to polynomial defects.
- **Initialization**: Can be harder to initialize if dynamics are unstable.

## Usage Patterns

### Basic Workflow

```cpp
metis::Opti opti;
metis::MultipleShooting ms(opti);

metis::MultiShootingOptions opts;
opts.n_intervals = 20;
opts.integrator = "cvodes";
opts.tol = 1e-6;

auto T = opti.variable(2.0); // Variable final time
auto [x, u, tau] = ms.setup(n_states, n_controls, 0.0, T, opts);

ms.set_dynamics([](const metis::SymbolicVector& x,
                    const metis::SymbolicVector& u,
                    const metis::SymbolicScalar& t) {
    return /* dxdt */;
});

ms.add_continuity_constraints();
ms.set_initial_state(x0);
ms.set_final_state(xf);

opti.minimize(T);
auto sol = opti.solve();
```

### Unified Comparison Example

The file `examples/optimization/transcription_comparison_demo.cpp` runs and compares Direct Collocation, Multiple Shooting, Pseudospectral, and Birkhoff Pseudospectral on the same brachistochrone problem so you can compare solver behavior and NLP structure directly. It includes solve-time performance statistics and a convergence sweep across grid sizes.

### When to Use

| Use Case | Why |
|----------|-----|
| High-fidelity simulation | Integrator accuracy |
| Stiff ODEs | Adaptive implicit methods |
| Matching simulation results | Same integration scheme |
| Fewer decision variables | Accuracy without more nodes |

## See Also

- [Transcription Methods Guide](transcription_methods.md) -- Comparison of all four transcription methods
- [Direct Collocation Guide](collocation.md) -- Polynomial defect-based transcription
- [Pseudospectral Guide](pseudospectral.md) -- Global polynomial transcription
- [Birkhoff Pseudospectral Guide](birkhoff_pseudospectral.md) -- Birkhoff-form transcription
- [transcription_comparison_demo.cpp](../../examples/optimization/transcription_comparison_demo.cpp) -- Unified comparison example
- [MultiShooting.hpp](../../include/metis/optimization/MultiShooting.hpp) -- API reference
