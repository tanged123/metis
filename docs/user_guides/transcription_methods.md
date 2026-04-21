# Transcription Methods

Metis provides four transcription methods for converting continuous-time optimal control problems into discrete NLPs: **Direct Collocation**, **Multiple Shooting**, **Pseudospectral**, and **Birkhoff Pseudospectral**. This guide compares all four, explains when to choose each, and documents the unified API they share. All transcription classes work in **symbolic mode** via the `metis::Opti` interface.

## Quick Start

```cpp
#include <metis/metis.hpp>

metis::Opti opti;

// Pick any transcription -- the API is the same
metis::DirectCollocation transcription(opti);
// metis::MultipleShooting transcription(opti);
// metis::Pseudospectral transcription(opti);
// metis::BirkhoffPseudospectral transcription(opti);

auto [x, u, tau] = transcription.setup(n_states, n_controls, t0, tf, opts);

transcription.set_dynamics(my_ode);
transcription.add_dynamics_constraints();
transcription.set_initial_state(x0);
transcription.set_final_state(xf);

opti.minimize(objective);
auto sol = opti.solve();
```

## Core API

All transcription classes share a unified API for interchangeability:

| Unified Method | Description |
|----------------|-------------|
| `setup(n_states, n_controls, t0, tf, opts)` | Create decision variables and time grid |
| `set_dynamics(ode)` | Set the ODE function |
| `add_dynamics_constraints()` | Add transcription-specific dynamics constraints |
| `set_initial_state(x0)` | Set initial boundary condition |
| `set_final_state(xf)` | Set final boundary condition |
| `n_nodes()` | Number of collocation/discretization nodes |
| `time_grid()` | Normalized time grid `[0, 1]` |

Method-specific aliases:

| Unified Method | DirectCollocation | MultipleShooting | Pseudospectral | Birkhoff Pseudospectral |
|----------------|-------------------|------------------|----------------|-------------------------|
| `add_dynamics_constraints()` | `add_defect_constraints()` | `add_continuity_constraints()` | `add_defect_constraints()` | `add_defect_constraints()` |
| `n_nodes()` | `n_nodes()` | -- | `n_nodes()` | `n_nodes()` |
| `n_intervals()` | -- | `n_intervals()` | -- | -- |

## Usage Patterns

### Method Comparison

| Aspect | Direct Collocation | Multiple Shooting | Pseudospectral | Birkhoff Pseudospectral |
|--------|-------------------|-------------------|----------------|-------------------------|
| **Dynamics Enforcement** | Local defect constraints | Numerical integration | Global differentiation matrix | Pointwise derivative collocation |
| **State Coupling** | Local between neighbors | Local across intervals | Dense nonlinear coupling via `D*X` | Linear coupling via integration matrix `B` |
| **Accuracy Source** | Fixed-order scheme (2nd/4th) | Adaptive-step integrator | Spectral convergence (smooth) | Spectral-like with better conditioning |
| **Control Representation** | Values at each node | Piecewise constant per interval | Values at each node | Values at each node |
| **Computational Cost** | Lower per iteration | Higher (integrator calls) | Moderate-high | Moderate |

### Direct Collocation

Uses polynomial interpolation to approximate state trajectories and enforces dynamics via **defect constraints**.

**How it works:**
```text
Trapezoidal:      x[k+1] - x[k] = h/2 * (f[k] + f[k+1])
Hermite-Simpson:  x[k+1] - x[k] = h/6 * (f[k] + 4*f_mid + f[k+1])
```

**Strengths:** Fast iterations, easy initialization, dense trajectory output, robust convergence.
**Weaknesses:** Fixed accuracy (2nd or 4th order), may struggle with stiff systems, needs more nodes for accuracy.

**Best for:** Smooth trajectories, real-time MPC, poor initial guesses, teaching/prototyping.

### Multiple Shooting

Divides time into intervals and integrates dynamics numerically. Continuity constraints connect intervals.

**How it works:** For each interval, integrate the ODE from state `x_k` using control `u_k`, then constrain `x_{k+1} = Integrate(x_k, u_k, dt)`.

**Strengths:** High accuracy via adaptive integrators (CVODES/IDAS), handles stiff systems, fewer intervals needed.
**Weaknesses:** Expensive derivatives (integrator sensitivities), initialization sensitivity, coarser trajectory.

**Best for:** High-fidelity simulation, stiff ODEs, matching simulation results, fewer decision variables.

### Pseudospectral

Uses global polynomial interpolation with Lobatto nodes and a spectral differentiation matrix: `D * X = (dt / 2) * F(X, U, t)`.

**Strengths:** High accuracy per node for smooth dynamics, built-in high-order quadrature, good low-node performance.
**Weaknesses:** Dense coupling across nodes, less robust for discontinuous/bang-bang controls.

**Best for:** Smooth OCPs with tight accuracy targets, low node budgets, accurate running-cost integration.

### Birkhoff Pseudospectral

Uses an integration matrix formulation: `X = x_a * 1 + B * V`, `V = (dt / 2) * F(X, U, t)`. Dense coupling stays mostly in linear constraints while dynamics are pointwise.

**Strengths:** Improved conditioning versus classical `D*X` at higher node counts, pointwise nonlinear dynamics, natural quadrature.
**Weaknesses:** More decision variables (`X` and `V` both present), still sensitive on non-smooth controls.

**Best for:** Smooth OCPs with larger node counts, problems where classical PS becomes iteration-heavy, high-order pseudospectral experiments.

### Side-by-Side Performance on Brachistochrone

| Method | Nodes/Intervals | Optimal Time | Error vs Reference |
|--------|-----------------|--------------|-------------------|
| Collocation (Hermite-Simpson) | 31 nodes | 1.8019 s | < 0.01% |
| Multiple Shooting (CVODES) | 20 intervals | 1.8019 s | < 0.01% |
| Pseudospectral (LGL) | 31 nodes | 1.8019 s | < 0.01% |
| Birkhoff Pseudospectral (LGL) | 31 nodes | 1.8019 s | < 0.01% |

### Problem Structure

```text
Collocation (31 nodes, 3 states, 1 control):
  Decision variables: 31x3 + 31x1 = 124
  Constraints: 30x3 (defects) + BCs

Multiple Shooting (20 intervals, 3 states, 1 control):
  Decision variables: 21x3 + 20x1 = 83
  Constraints: 20x3 (continuity) + BCs

Pseudospectral (31 nodes, 3 states, 1 control):
  Decision variables: 31x3 + 31x1 = 124
  Constraints: 31x3 (global dynamics) + BCs

Birkhoff Pseudospectral (31 nodes, 3 states, 1 control):
  Decision variables: 31x3 + 31x3 + 31x1 = 217
  Constraints: 31x3 (pointwise dynamics) + 31x3 (linear recovery) + BCs
```

### Decision Flowchart

```text
+----------------------------------------------+
| Is the system stiff or integration-sensitive? |
+--------------+-------------------------------+
|      YES     |              NO               |
|      v       |              v                |
| Multiple     | Are dynamics/controls smooth  |
| Shooting     | enough for global polynomials?|
|              +--------------+----------------+
|              |      YES     |       NO       |
|              |      v       |       v        |
|              | Need better  | Direct         |
|              | conditioning | Collocation    |
|              +-------+------+                |
|              |  YES  |  NO  |                |
|              |   v   |   v  |                |
|              |Birkhoff|Pseudo|                |
+--------------+-------+------+----------------+
```

### Quick Reference

**Choose Direct Collocation when:**
- You need fast solver iterations (MPC, real-time)
- Initial guess is poor or unknown
- Problem is smooth and non-stiff
- You want dense trajectory output

**Choose Multiple Shooting when:**
- High integration accuracy is required
- System is stiff (chemical kinetics, some robotics)
- You are matching high-fidelity simulation
- Fewer decision variables are preferred

**Choose Pseudospectral when:**
- Dynamics and controls are smooth
- You need high accuracy with relatively few nodes
- Running-cost integration accuracy is important

**Choose Birkhoff Pseudospectral when:**
- You want pseudospectral accuracy but better conditioning behavior
- Node counts are moderate-to-high
- You want pointwise nonlinear dynamics with linear global recovery

## See Also

- [Direct Collocation Guide](collocation.md) -- Detailed collocation usage
- [Multiple Shooting Guide](multiple_shooting.md) -- Detailed multiple shooting usage
- [Pseudospectral Guide](pseudospectral.md) -- Detailed pseudospectral usage
- [Birkhoff Pseudospectral Guide](birkhoff_pseudospectral.md) -- Detailed Birkhoff usage
- [transcription_comparison_demo.cpp](../../examples/optimization/transcription_comparison_demo.cpp) -- Unified comparison example
- [TranscriptionBase.hpp](../../include/metis/optimization/TranscriptionBase.hpp) -- Base class API reference
