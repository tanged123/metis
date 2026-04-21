/**
 * @file test_collocation.cpp
 * @brief Tests for DirectCollocation trajectory optimization
 */

#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/optimization/Collocation.hpp>
#include <metis/optimization/Opti.hpp>
#include <string>

using namespace metis;

// ============================================================================
// Double Integrator Tests (x'' = u)
// ============================================================================

namespace {
/**
 * @brief Simple double integrator dynamics
 * state = [position, velocity]
 * control = acceleration
 */
SymbolicVector double_integrator_ode(const SymbolicVector &x, const SymbolicVector &u,
                                     const SymbolicScalar & /*t*/) {
    SymbolicVector dxdt(2);
    dxdt(0) = x(1); // dx/dt = v
    dxdt(1) = u(0); // dv/dt = u (acceleration)
    return dxdt;
}
} // namespace

TEST(CollocationTests, TrapezoidalDoubleIntegrator) {
    // Move from x=0,v=0 to x=1,v=0 in fixed time T=2
    // Optimal control is bang-bang, but trapezoidal will approximate

    Opti opti;
    DirectCollocation dc(opti);

    CollocationOptions opts;
    opts.scheme = CollocationScheme::Trapezoidal;
    opts.n_nodes = 11;

    auto [x, u, tau] = dc.setup(2, 1, 0.0, 2.0, opts);

    dc.set_dynamics(double_integrator_ode);
    dc.add_defect_constraints();

    // Boundary conditions
    dc.set_initial_state(NumericVector{{0.0, 0.0}});
    dc.set_final_state(NumericVector{{1.0, 0.0}});

    // Bound control
    opti.subject_to_bounds(u.col(0), -10.0, 10.0);

    // Minimize control effort (L2 norm)
    SymbolicScalar obj = 0;
    for (int k = 0; k < dc.n_nodes(); ++k) {
        obj = obj + u(k, 0) * u(k, 0);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    // Check boundary conditions are satisfied
    auto x_sol = sol.value(x);
    EXPECT_NEAR(x_sol(0, 0), 0.0, 1e-4);                // Initial position
    EXPECT_NEAR(x_sol(0, 1), 0.0, 1e-4);                // Initial velocity
    EXPECT_NEAR(x_sol(dc.n_nodes() - 1, 0), 1.0, 1e-4); // Final position
    EXPECT_NEAR(x_sol(dc.n_nodes() - 1, 1), 0.0, 1e-4); // Final velocity
}

TEST(CollocationTests, HermiteSimpsonDoubleIntegrator) {
    // Same problem with Hermite-Simpson (should be more accurate)

    Opti opti;
    DirectCollocation dc(opti);

    CollocationOptions opts;
    opts.scheme = CollocationScheme::HermiteSimpson;
    opts.n_nodes = 11;

    auto [x, u, tau] = dc.setup(2, 1, 0.0, 2.0, opts);

    dc.set_dynamics(double_integrator_ode);
    dc.add_defect_constraints();

    dc.set_initial_state(NumericVector{{0.0, 0.0}});
    dc.set_final_state(NumericVector{{1.0, 0.0}});

    opti.subject_to_bounds(u.col(0), -10.0, 10.0);

    SymbolicScalar obj = 0;
    for (int k = 0; k < dc.n_nodes(); ++k) {
        obj = obj + u(k, 0) * u(k, 0);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    auto x_sol = sol.value(x);
    EXPECT_NEAR(x_sol(0, 0), 0.0, 1e-4);
    EXPECT_NEAR(x_sol(0, 1), 0.0, 1e-4);
    EXPECT_NEAR(x_sol(dc.n_nodes() - 1, 0), 1.0, 1e-4);
    EXPECT_NEAR(x_sol(dc.n_nodes() - 1, 1), 0.0, 1e-4);
}

// ============================================================================
// Free Final Time Test
// ============================================================================

TEST(CollocationTests, FreeTimeDoubleIntegrator) {
    // Move from x=0,v=0 to x=1,v=0 with control bounded |u| <= 1
    // Minimum time problem

    Opti opti;
    DirectCollocation dc(opti);

    // Create final time as decision variable
    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    CollocationOptions opts;
    opts.scheme = CollocationScheme::Trapezoidal;
    opts.n_nodes = 21;

    auto [x, u, tau] = dc.setup(2, 1, 0.0, T, opts);

    dc.set_dynamics(double_integrator_ode);
    dc.add_defect_constraints();

    dc.set_initial_state(NumericVector{{0.0, 0.0}});
    dc.set_final_state(NumericVector{{1.0, 0.0}});

    // Control bounds
    opti.subject_to_bounds(u.col(0), -1.0, 1.0);

    // Minimize time
    opti.minimize(T);

    auto sol = opti.solve({.verbose = false});

    // Analytic solution: T* = 2 (for unit displacement with unit control bound)
    double T_opt = sol.value(T);
    EXPECT_NEAR(T_opt, 2.0, 0.1); // Allow some tolerance for discretization

    auto x_sol = sol.value(x);
    EXPECT_NEAR(x_sol(dc.n_nodes() - 1, 0), 1.0, 1e-3);
    EXPECT_NEAR(x_sol(dc.n_nodes() - 1, 1), 0.0, 1e-3);
}

// ============================================================================
// Simple Harmonic Oscillator (conservation check)
// ============================================================================

TEST(CollocationTests, HarmonicOscillatorEnergy) {
    // x'' = -x (undamped oscillator)
    // Energy E = 0.5*(v^2 + x^2) should be conserved

    auto oscillator_ode = [](const SymbolicVector &x, const SymbolicVector & /*u*/,
                             const SymbolicScalar & /*t*/) {
        SymbolicVector dxdt(2);
        dxdt(0) = x(1);  // dx/dt = v
        dxdt(1) = -x(0); // dv/dt = -x
        return dxdt;
    };

    Opti opti;
    DirectCollocation dc(opti);

    CollocationOptions opts;
    opts.scheme = CollocationScheme::HermiteSimpson;
    opts.n_nodes = 51;

    // Simulate for one period (2*pi)
    auto [x, u, tau] = dc.setup(2, 0, 0.0, 2.0 * M_PI, opts);

    dc.set_dynamics(oscillator_ode);
    dc.add_defect_constraints();

    // Initial conditions: x=1, v=0 (E = 0.5)
    dc.set_initial_state(NumericVector{{1.0, 0.0}});

    // No objective (just satisfy dynamics)
    opti.minimize(SymbolicScalar(0));

    auto sol = opti.solve({.verbose = false});

    auto x_sol = sol.value(x);

    // Check energy conservation at start and end
    double E_init = 0.5 * (x_sol(0, 1) * x_sol(0, 1) + x_sol(0, 0) * x_sol(0, 0));
    double E_final = 0.5 * (x_sol(dc.n_nodes() - 1, 1) * x_sol(dc.n_nodes() - 1, 1) +
                            x_sol(dc.n_nodes() - 1, 0) * x_sol(dc.n_nodes() - 1, 0));

    EXPECT_NEAR(E_init, 0.5, 1e-3);
    EXPECT_NEAR(E_final, 0.5, 0.05); // Some drift OK due to discretization

    // After one period, should return to (1, 0)
    EXPECT_NEAR(x_sol(dc.n_nodes() - 1, 0), 1.0, 0.1);
    EXPECT_NEAR(x_sol(dc.n_nodes() - 1, 1), 0.0, 0.1);
}

TEST(CollocationTests, SetDynamicsAfterConstraintsThrows) {
    Opti opti;
    DirectCollocation dc(opti);

    CollocationOptions opts;
    opts.scheme = CollocationScheme::Trapezoidal;
    opts.n_nodes = 5;

    dc.setup(2, 1, 0.0, 1.0, opts);
    dc.set_dynamics(double_integrator_ode);
    dc.add_dynamics_constraints();

    try {
        dc.set_dynamics(double_integrator_ode);
        FAIL() << "Expected set_dynamics() to throw after add_dynamics_constraints()";
    } catch (const metis::RuntimeError &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("set_dynamics"), std::string::npos);
        EXPECT_NE(msg.find("add_dynamics_constraints"), std::string::npos);
    }
}
