/**
 * @file integration_demo.cpp
 * @brief Demo: Using Metis integrators with Icarus-style component models
 *
 * This example explores several integration patterns:
 * 1. Monolithic state vector (traditional approach)
 * 2. Component-based state management (Icarus-style)
 * 3. Structure-preserving second-order propagation
 * 4. Stiff and constrained mass-matrix systems
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

using namespace metis;

// ============================================================================
// Pattern 1: Monolithic State Vector (Traditional)
// ============================================================================
// All state variables in one vector, one dynamics function.
// Good for small systems, trajectory optimization.

void monolithic_example() {
    std::cout << "=== Pattern 1: Monolithic State Vector ===\n";

    // Harmonic oscillator: y'' = -ω²y
    // State: [y, v] where y'=v, v'=-ω²y
    double omega = 2.0;

    auto dynamics = [omega](double t, const NumericVector &state) {
        NumericVector dydt(2);
        dydt << state(1), -omega * omega * state(0);
        return dydt;
    };

    NumericVector state(2);
    state << 1.0, 0.0; // y=1, v=0
    double t = 0.0;
    double dt = 0.01;

    std::cout << "t=0: y=" << state(0) << ", v=" << state(1) << "\n";

    // Simulate for 1 second
    for (int i = 0; i < 100; ++i) {
        state = rk4_step(dynamics, state, t, dt);
        t += dt;
    }

    std::cout << "t=1: y=" << state(0) << " (expected: " << std::cos(omega) << ")\n\n";
}

// ============================================================================
// Pattern 1b: Structure-Preserving Second-Order Integration
// ============================================================================

double oscillator_energy(double q, double v, double omega) {
    return 0.5 * (v * v + omega * omega * q * q);
}

void structure_preserving_example() {
    std::cout << "=== Pattern 1b: Structure-Preserving Second-Order Integration ===\n";

    const double omega = 1.0;
    const double dt = 0.2;
    const int n_steps = 20000;

    auto accel = [omega](double t, const NumericVector &q_state) {
        return (-omega * omega * q_state).eval();
    };

    NumericVector first_order_state(2);
    first_order_state << 1.0, 0.0;

    NumericVector q(1);
    NumericVector v(1);
    q << 1.0;
    v << 0.0;

    const double energy0 = oscillator_energy(1.0, 0.0, omega);
    double rk4_max_drift = 0.0;
    double verlet_max_drift = 0.0;
    double rk4_final_drift = 0.0;
    double verlet_final_drift = 0.0;
    double t = 0.0;

    auto first_order_dynamics = [omega](double t, const NumericVector &state) {
        NumericVector dydt(2);
        dydt << state(1), -omega * omega * state(0);
        return dydt;
    };

    for (int i = 0; i < n_steps; ++i) {
        first_order_state = rk4_step(first_order_dynamics, first_order_state, t, dt);
        auto verlet_step = stormer_verlet_step(accel, q, v, t, dt);
        q = verlet_step.q;
        v = verlet_step.v;
        t += dt;

        rk4_max_drift =
            std::max(rk4_max_drift,
                     std::abs(oscillator_energy(first_order_state(0), first_order_state(1), omega) -
                              energy0));
        verlet_max_drift =
            std::max(verlet_max_drift, std::abs(oscillator_energy(q(0), v(0), omega) - energy0));
    }

    rk4_final_drift =
        std::abs(oscillator_energy(first_order_state(0), first_order_state(1), omega) - energy0);
    verlet_final_drift = std::abs(oscillator_energy(q(0), v(0), omega) - energy0);

    std::cout << "Long-horizon harmonic oscillator with dt=" << dt << " for " << n_steps
              << " steps\n";
    std::cout << "  RK4 final energy drift          = " << rk4_final_drift
              << " (max: " << rk4_max_drift << ")\n";
    std::cout << "  Stormer-Verlet final drift      = " << verlet_final_drift
              << " (bounded max: " << verlet_max_drift << ")\n";

    SecondOrderIvpOptions verlet_opts;
    verlet_opts.method = SecondOrderIntegratorMethod::StormerVerlet;
    auto verlet_traj =
        solve_second_order_ivp(accel, {0.0, 2.0 * M_PI}, {1.0}, {0.0}, 200, verlet_opts);

    SecondOrderIvpOptions rkn_opts;
    rkn_opts.method = SecondOrderIntegratorMethod::RungeKuttaNystrom4;
    auto rkn_traj = solve_second_order_ivp(accel, {0.0, 2.0 * M_PI}, {1.0}, {0.0}, 200, rkn_opts);

    std::cout << "One-period solve_second_order_ivp comparison\n";
    std::cout << "  Stormer-Verlet: q(tf)=" << verlet_traj.q(0, verlet_traj.q.cols() - 1)
              << ", v(tf)=" << verlet_traj.v(0, verlet_traj.v.cols() - 1) << "\n";
    std::cout << "  RKN4:            q(tf)=" << rkn_traj.q(0, rkn_traj.q.cols() - 1)
              << ", v(tf)=" << rkn_traj.v(0, rkn_traj.v.cols() - 1) << "\n\n";
}

// ============================================================================
// Pattern 2: Component-Based State (Icarus-Style)
// ============================================================================
// Each component owns its state and provides derivative computation.

/**
 * @brief Base class for integrable state variables
 *
 * Components inherit from this to expose their ODE state to the integrator.
 */
template <typename Scalar> class IntegrableState {
  public:
    virtual ~IntegrableState() = default;

    /// Get current state as a vector
    virtual MetisVector<Scalar> get_state() const = 0;

    /// Set state from a vector
    virtual void set_state(const MetisVector<Scalar> &state) = 0;

    /// Compute derivative: dx/dt = f(t, x)
    virtual MetisVector<Scalar> compute_derivative(Scalar t) const = 0;

    /// Number of state variables
    virtual int state_dim() const = 0;
};

/**
 * @brief Example: 6DOF Body State Component
 *
 * Manages position, velocity, orientation, angular velocity.
 * In real Icarus, this would receive forces/torques from other components.
 */
class RigidBody : public IntegrableState<double> {
  public:
    // State: [x, y, z, vx, vy, vz] (simplified - no rotation for demo)
    Vec3<double> position{0, 0, 0};
    Vec3<double> velocity{0, 0, 0};

    // Inputs (set by other components)
    Vec3<double> acceleration{0, 0, 0}; // From forces

    int state_dim() const override { return 6; }

    MetisVector<double> get_state() const override {
        NumericVector s(6);
        s << position(0), position(1), position(2), velocity(0), velocity(1), velocity(2);
        return s;
    }

    void set_state(const MetisVector<double> &state) override {
        position << state(0), state(1), state(2);
        velocity << state(3), state(4), state(5);
    }

    MetisVector<double> compute_derivative(double t) const override {
        NumericVector dydt(6);
        // dx/dt = v, dv/dt = a
        dydt << velocity(0), velocity(1), velocity(2), acceleration(0), acceleration(1),
            acceleration(2);
        return dydt;
    }
};

/**
 * @brief Example: Gravity Model Component
 *
 * Computes gravitational acceleration for a body.
 */
class GravityModel {
  public:
    double mu = 3.986e14; // Earth GM

    void update(RigidBody &body) {
        double r = body.position.norm();
        if (r > 0) {
            body.acceleration = -mu / (r * r * r) * body.position;
        }
    }
};

/**
 * @brief Integrator that works with IntegrableState components
 */
template <typename Scalar> class ComponentIntegrator {
  public:
    void step_rk4(IntegrableState<Scalar> &component, Scalar t, Scalar dt) {
        // Wrap component's derivative function for metis::rk4_step
        auto dynamics = [&component](Scalar t, const MetisVector<Scalar> &state) {
            // Temporarily set state for derivative computation
            auto saved_state = component.get_state();
            component.set_state(state);
            auto deriv = component.compute_derivative(t);
            component.set_state(saved_state); // Restore original
            return deriv;
        };

        auto new_state = rk4_step(dynamics, component.get_state(), t, dt);
        component.set_state(new_state);
    }
};

void component_based_example() {
    std::cout << "=== Pattern 2: Component-Based State ===\n";

    // Create components
    RigidBody satellite;
    satellite.position << 7000e3, 0, 0; // LEO altitude
    satellite.velocity << 0, 7500, 0;   // Circular orbit velocity

    GravityModel gravity;
    ComponentIntegrator<double> integrator;

    double t = 0.0;
    double dt = 10.0; // 10 second steps
    double period = 2 * M_PI * std::sqrt(std::pow(7000e3, 3) / 3.986e14);

    std::cout << "Orbital period: " << period << " seconds\n";
    std::cout << "Initial position: [" << satellite.position.transpose() << "]\n";

    // Simulate half an orbit
    int steps = static_cast<int>(period / 2 / dt);
    for (int i = 0; i < steps; ++i) {
        gravity.update(satellite); // Compute forces
        integrator.step_rk4(satellite, t, dt);
        t += dt;
    }

    std::cout << "After half orbit: [" << satellite.position.transpose() << "]\n";
    std::cout << "(Expected x ≈ -7000 km)\n\n";
}

// ============================================================================
// Pattern 3: ODE Variable Wrapper (Alternative Design)
// ============================================================================

/**
 * @brief Lightweight wrapper for a single ODE variable
 *
 * Alternative to full IntegrableState - just wraps value and derivative.
 * Good for simple scalar or vector states without component overhead.
 */
template <typename T> struct OdeVar {
    T value;
    std::function<T(double, const T &)> derivative;

    void step_euler(double t, double dt) { value = value + dt * derivative(t, value); }

    void step_rk4(double t, double dt) {
        T k1 = derivative(t, value);
        T k2 = derivative(t + dt / 2, value + dt / 2 * k1);
        T k3 = derivative(t + dt / 2, value + dt / 2 * k2);
        T k4 = derivative(t + dt, value + dt * k3);
        value = value + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    }
};

void ode_var_example() {
    std::cout << "=== Pattern 3: ODE Variable Wrapper ===\n";

    // Simple exponential decay
    OdeVar<double> temperature{.value = 100.0,
                               .derivative = [](double t, const double &T) { return -0.1 * T; }};

    double t = 0.0;
    double dt = 0.1;

    std::cout << "t=0: T=" << temperature.value << "\n";

    for (int i = 0; i < 100; ++i) {
        temperature.step_rk4(t, dt);
        t += dt;
    }

    std::cout << "t=10: T=" << temperature.value << " (expected: " << 100 * std::exp(-1) << ")\n\n";
}

// ============================================================================
// Pattern 4: Stiff And Constrained Mass-Matrix Systems
// ============================================================================

void mass_matrix_example() {
    std::cout << "=== Pattern 4: Stiff And Constrained Mass-Matrix Systems ===\n";

    MassMatrixIvpOptions rosen_opts;
    rosen_opts.method = MassMatrixIntegratorMethod::RosenbrockEuler;
    rosen_opts.substeps = 2;

    auto stiff_sol = solve_ivp_mass_matrix(
        [](double t, const NumericVector &y) {
            NumericVector rhs(1);
            rhs(0) = -20.0 * y(0);
            return rhs;
        },
        [](double t, const NumericVector &y) {
            NumericMatrix M(1, 1);
            M(0, 0) = 2.0;
            return M;
        },
        {0.0, 0.5}, {1.0}, 80, rosen_opts);

    std::cout << "Rosenbrock-Euler on 2*y' = -20*y\n";
    std::cout << "  y(0.5) = " << stiff_sol.y(0, stiff_sol.y.cols() - 1)
              << " (exact: " << std::exp(-5.0) << ")\n";

    MassMatrixIvpOptions bdf_opts;
    bdf_opts.method = MassMatrixIntegratorMethod::Bdf1;
    bdf_opts.substeps = 2;

    auto constrained_sol = solve_ivp_mass_matrix(
        [](double t, const NumericVector &y) {
            NumericVector rhs(2);
            rhs << y(1), 1.0 - y(0) - y(1);
            return rhs;
        },
        [](double t, const NumericVector &y) {
            NumericMatrix M = NumericMatrix::Zero(2, 2);
            M(0, 0) = 1.0;
            return M;
        },
        {0.0, 1.0}, {0.0, 1.0}, 50, bdf_opts);

    std::cout << "BDF1 on a singular-mass constrained system\n";
    std::cout << "  x(tf) = " << constrained_sol.y(0, constrained_sol.y.cols() - 1)
              << ", z(tf) = " << constrained_sol.y(1, constrained_sol.y.cols() - 1) << "\n";

    auto t_sym = metis::sym("t");
    auto y_sym = metis::sym("y", 2);

    casadi::MX rhs_sym = casadi::MX::vertcat({y_sym(1), 1.0 - y_sym(0) - y_sym(1)});
    casadi::MX M_sym = casadi::MX::zeros(2, 2);
    M_sym(0, 0) = 1.0;

    MassMatrixIvpOptions idas_opts;
    idas_opts.abstol = 1e-10;
    idas_opts.reltol = 1e-10;

    NumericVector y0(2);
    y0 << 0.0, 1.0;
    auto symbolic_sol =
        solve_ivp_mass_matrix_expr(rhs_sym, M_sym, t_sym, y_sym, {0.0, 1.0}, y0, 40, idas_opts);

    std::cout << "IDAS symbolic mass-matrix path on the same constrained system\n";
    std::cout << "  x(tf) = " << symbolic_sol.y(0, symbolic_sol.y.cols() - 1)
              << ", z(tf) = " << symbolic_sol.y(1, symbolic_sol.y.cols() - 1) << "\n\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "Metis Integration Demo\n";
    std::cout << "======================\n\n";
    std::cout << std::fixed << std::setprecision(6);

    monolithic_example();
    structure_preserving_example();
    component_based_example();
    ode_var_example();
    mass_matrix_example();

    std::cout << "Recommendation for Icarus:\n";
    std::cout << "--------------------------\n";
    std::cout << "Use Pattern 2 (Component-Based) for simulation.\n";
    std::cout << "Each component implements IntegrableState interface.\n";
    std::cout << "Integrator collects all components and steps them together.\n";
    std::cout << "Use Pattern 1b when the system is naturally second-order and long-horizon\n";
    std::cout << "energy behavior matters.\n";
    std::cout << "Use Pattern 4 when the system is stiff or carries algebraic constraints.\n";

    return 0;
}
