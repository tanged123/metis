/**
 * @file test_multishoot.cpp
 * @brief Tests for MultipleShooting trajectory optimization
 */

#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/optimization/MultiShooting.hpp>
#include <metis/optimization/Opti.hpp>

using namespace metis;

// ============================================================================
// Double Integrator Tests (x'' = u)
// ============================================================================

namespace {
/**
 * @brief Simple double integrator dynamics
 */
SymbolicVector double_integrator_ode(const SymbolicVector &x, const SymbolicVector &u,
                                     const SymbolicScalar & /*t*/) {
    SymbolicVector dxdt(2);
    dxdt(0) = x(1); // dx/dt = v
    dxdt(1) = u(0); // dv/dt = u (acceleration)
    return dxdt;
}
} // namespace

TEST(MultipleShootingTests, SimpleIntegrator) {
    // 1D motion: x' = v, v' = u
    // Move 0->1 in T=2 with min control

    Opti opti;
    MultipleShooting ms(opti);

    MultiShootingOptions opts;
    opts.n_intervals = 10;
    opts.integrator = "cvodes"; // Using Sundials CVODES
    opts.tol = 1e-6;

    auto [x, u, tau] = ms.setup(2, 1, 0.0, 2.0, opts);

    ms.set_dynamics(double_integrator_ode);
    ms.add_continuity_constraints();

    // Boundary conditions
    ms.set_initial_state(NumericVector{{0.0, 0.0}});
    ms.set_final_state(NumericVector{{1.0, 0.0}});

    // Control bounds
    opti.subject_to_bounds(u.col(0), -10.0, 10.0);

    // Minimize control effort
    SymbolicScalar obj = 0;
    for (int k = 0; k < opts.n_intervals; ++k) {
        obj = obj + u(k, 0) * u(k, 0);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    auto x_sol = sol.value(x);
    // double integrator optimal control solution is bang-bang or smooth?
    // L2 norm objective -> smooth linear u.
    // x(t) cubic, v(t) quadratic, u(t) linear.

    // Check boundary
    EXPECT_NEAR(x_sol(0, 0), 0.0, 1e-4);
    EXPECT_NEAR(x_sol(opts.n_intervals, 0), 1.0, 1e-4);
    EXPECT_NEAR(x_sol(opts.n_intervals, 1), 0.0, 1e-4);
}

// ============================================================================
// Variable Time Test
// ============================================================================

TEST(MultipleShootingTests, FreeTimeDoubleIntegrator) {
    // Move from x=0,v=0 to x=1,v=0 with bounded control |u| <= 1
    // Minimize Time

    Opti opti;
    MultipleShooting ms(opti);

    auto T = opti.variable(2.0, std::nullopt, 0.5, 5.0);

    MultiShootingOptions opts;
    opts.n_intervals = 10;
    opts.integrator = "cvodes";

    auto [x, u, tau] = ms.setup(2, 1, 0.0, T, opts);

    ms.set_dynamics(double_integrator_ode);
    ms.add_continuity_constraints();

    ms.set_initial_state(NumericVector{{0.0, 0.0}});
    ms.set_final_state(NumericVector{{1.0, 0.0}});

    opti.subject_to_bounds(u.col(0), -1.0, 1.0);

    opti.minimize(T);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(T), 2.0, 0.05); // Analytic T* = 2
}

// ============================================================================
// Stiff System / Oscillator
// ============================================================================

TEST(MultipleShootingTests, OscillatorEnergy) {
    // x'' = -x
    auto ode = [](const SymbolicVector &x, const SymbolicVector &, const SymbolicScalar &) {
        SymbolicVector dxdt(2);
        dxdt(0) = x(1);
        dxdt(1) = -x(0);
        return dxdt;
    };

    Opti opti;
    MultipleShooting ms(opti);

    MultiShootingOptions opts;
    opts.n_intervals = 20;
    opts.integrator = "cvodes";
    // CVODES is a variable-step integrator, should handle this accurately

    ms.setup(2, 0, 0.0, 2.0 * M_PI, opts);
    ms.set_dynamics(ode);
    ms.add_continuity_constraints();

    ms.set_initial_state(NumericVector{{1.0, 0.0}}); // E=0.5

    // No objective
    opti.minimize(0);

    auto sol = opti.solve({.verbose = false});
    auto x_sol = sol.value(ms.states());

    // Check return to start after 2*pi
    int N = opts.n_intervals;
    EXPECT_NEAR(x_sol(N, 0), 1.0, 1e-4);
    EXPECT_NEAR(x_sol(N, 1), 0.0, 1e-4);
}
