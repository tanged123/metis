# ODE Integration

Metis provides explicit ODE integrators for simulation with dual-mode (numeric/symbolic) support, plus structure-preserving and stiff mass-matrix integrators for the cases where generic RK integration is the wrong tool. All step functions work in both numeric and symbolic modes, making them suitable for simulation, trajectory optimization, and sensitivity analysis.

## Quick Start

```cpp
#include <metis/metis.hpp>

// Define dynamics: dy/dt = f(t, y)
auto dynamics = [](double t, const metis::NumericVector& y) {
    return -0.5 * y;  // Exponential decay
};

metis::NumericVector y(1);
y(0) = 1.0;
double t = 0.0, dt = 0.1;

// Single step
y = metis::rk4_step(dynamics, y, t, dt);
```

## Core API

### Step Integrators

| Function | Order | Evaluations | Use Case |
|----------|-------|-------------|----------|
| `metis::euler_step` | 1st | 1 | Fast, low accuracy |
| `metis::rk2_step` | 2nd | 2 | Moderate accuracy |
| `metis::rk4_step` | 4th | 4 | General purpose |
| `metis::rk45_step` | 4th/5th | 7 | Adaptive stepping |
| `metis::stormer_verlet_step` | 2nd | 2 accel evals | Symplectic mechanical/orbital systems |
| `metis::rkn4_step` | 4th | 4 accel evals | Second-order systems without state augmentation |

### Trajectory Solvers

*   **`metis::solve_ivp(dynamics, t_span, y0, n_eval)`**: Full trajectory integration for first-order systems.
*   **`metis::solve_second_order_ivp(accel, t_span, q0, v0, n_eval, opts)`**: Trajectory integration for second-order systems with separate coordinates and velocities.
*   **`metis::solve_ivp_mass_matrix(rhs, mass, t_span, y0, n_eval, opts)`**: Stiff and mass-matrix systems `M(t,y) y' = f(t,y)`.

## Usage Patterns

### RK4 Step (Recommended)

```cpp
// Harmonic oscillator: y'' = -w^2*y
// State: [y, v] where dy/dt=v, dv/dt=-w^2*y
double omega = 2.0;

auto dynamics = [omega](double t, const metis::NumericVector& s) {
    metis::NumericVector ds(2);
    ds << s(1), -omega * omega * s(0);
    return ds;
};

metis::NumericVector state(2);
state << 1.0, 0.0;  // y=1, v=0
double t = 0.0;

for (int i = 0; i < 100; ++i) {
    state = metis::rk4_step(dynamics, state, t, 0.01);
    t += 0.01;
}
```

### Adaptive Stepping with RK45

```cpp
double dt = 0.1;
double tol = 1e-6;

while (t < t_final) {
    auto result = metis::rk45_step(dynamics, y, t, dt);

    if (result.error < tol) {
        y = result.y5;  // Accept step
        t += dt;
        dt *= 1.5;      // Grow step
    } else {
        dt *= 0.5;      // Shrink step and retry
    }
}
```

### Structure-Preserving Second-Order Steppers

When your dynamics are naturally written as `q'' = a(t, q)`, it is better to integrate that form directly than to augment it into a generic first-order state and run plain RK4.

```cpp
metis::NumericVector q(1), v(1);
q << 1.0;
v << 0.0;

auto accel = [](double t, const metis::NumericVector& q_state) {
    return (-q_state).eval();  // Harmonic oscillator
};

auto step = metis::stormer_verlet_step(accel, q, v, 0.0, 0.1);
q = step.q;
v = step.v;
```

- `metis::stormer_verlet_step(...)` is symplectic and keeps long-horizon energy error bounded for separable Hamiltonian systems.
- `metis::rkn4_step(...)` is a higher-order Runge-Kutta-Nystrom method for the same `q'' = a(t, q)` API.

### Trajectory Solver

For full trajectory integration, use `solve_ivp`:

```cpp
auto sol = metis::solve_ivp(
    dynamics,
    {0.0, 10.0},  // t_span
    {1.0},        // y0 (initializer list)
    100           // n_eval points
);

// sol.t - time points
// sol.y - solution matrix (rows=states, cols=time)
```

For second-order systems, use `solve_second_order_ivp(...)` to keep coordinates and velocities separate:

```cpp
metis::SecondOrderIvpOptions opts;
opts.method = metis::SecondOrderIntegratorMethod::StormerVerlet;

auto sol = metis::solve_second_order_ivp(
    accel,
    {0.0, 100.0},
    {1.0},   // q0
    {0.0},   // v0
    1000,
    opts
);

// sol.q - generalized coordinates, rows = coordinates, cols = time
// sol.v - generalized velocities, rows = velocities, cols = time
```

### Stiff and Mass-Matrix Systems

For systems of the form `M(t, y) y' = f(t, y)`, Metis provides a dedicated solver surface:

```cpp
metis::MassMatrixIvpOptions opts;
opts.method = metis::MassMatrixIntegratorMethod::Bdf1;
opts.substeps = 2;

auto sol = metis::solve_ivp_mass_matrix(
    [](double t, const metis::NumericVector& y) {
        metis::NumericVector rhs(2);
        rhs << y(1), 1.0 - y(0) - y(1);
        return rhs;
    },
    [](double t, const metis::NumericVector& y) {
        metis::NumericMatrix M = metis::NumericMatrix::Zero(2, 2);
        M(0, 0) = 1.0;  // singular mass matrix encoding an algebraic constraint
        return M;
    },
    {0.0, 1.0},
    {0.0, 1.0},
    50,
    opts
);
```

- `MassMatrixIntegratorMethod::RosenbrockEuler` is a one-stage linearly implicit stiff integrator.
- `MassMatrixIntegratorMethod::Bdf1` solves the backward-Euler residual directly and can handle simple singular mass-matrix systems.
- `solve_ivp_mass_matrix_expr(...)` uses CasADi IDAS on the symbolic expression path by rewriting the mass-matrix system into a semi-explicit DAE.

### Symbolic Mode

All step functions work with symbolic types for optimization:

```cpp
auto x = metis::sym_vec("x", 2);
auto t = metis::sym("t");
auto dt = metis::sym("dt");

auto x_next = metis::rk4_step(
    [](auto t, const auto& x) {
        metis::SymbolicVector dx(2);
        dx << x(1), -4.0 * x(0);
        return dx;
    },
    x, t, dt
);

// x_next is symbolic - can be used in optimization
casadi::Function step_fn("step", {metis::to_mx(x), t, dt}, {metis::to_mx(x_next)});
```

The structure-preserving step APIs are also symbolic-safe:

```cpp
auto q = metis::sym_vec("q", 1);
auto v = metis::sym_vec("v", 1);
auto t = metis::sym("t");
auto dt = metis::sym("dt");

auto step = metis::stormer_verlet_step(
    [](auto time, const auto& q_state) { return (-q_state).eval(); },
    q, v, t, dt
);
```

### Component-Based Integration (Icarus Pattern)

For simulation frameworks with component models:

```cpp
template <typename Scalar>
class IntegrableState {
public:
    virtual MetisVector<Scalar> get_state() const = 0;
    virtual void set_state(const MetisVector<Scalar>& s) = 0;
    virtual MetisVector<Scalar> compute_derivative(Scalar t) const = 0;
};

class RigidBody : public IntegrableState<double> {
    Vec3<double> position, velocity, acceleration;

    MetisVector<double> get_state() const override { /* ... */ }
    void set_state(const MetisVector<double>& s) override { /* ... */ }
    MetisVector<double> compute_derivative(double t) const override {
        NumericVector dydt(6);
        dydt << velocity, acceleration;
        return dydt;
    }
};
```

## See Also

- [Symbolic Computing Guide](symbolic_computing.md) - Symbolic mode for optimization
- [Optimization Guide](optimization.md) - Using integrators in trajectory optimization
- [`examples/integration_demo.cpp`](../../examples/integration_demo.cpp) - Complete integration example
- [`examples/simulation/brachistochrone.cpp`](../../examples/simulation/brachistochrone.cpp) - Brachistochrone simulation
- [`include/metis/math/IntegratorStep.hpp`](../../include/metis/math/IntegratorStep.hpp) - Step integrator API
- [`include/metis/math/Integrate.hpp`](../../include/metis/math/Integrate.hpp) - Trajectory solver API
