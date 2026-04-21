/**
 * @file test_birkhoff_pseudospectral.cpp
 * @brief Tests for Birkhoff pseudospectral transcription
 */

#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Trig.hpp>
#include <metis/optimization/BirkhoffPseudospectral.hpp>
#include <metis/optimization/Opti.hpp>
#include <metis/optimization/Pseudospectral.hpp>

#include <cmath>

using namespace metis;

namespace {

SymbolicVector double_integrator_ode(const SymbolicVector &x, const SymbolicVector &u,
                                     const SymbolicScalar & /*t*/) {
    SymbolicVector dxdt(2);
    dxdt(0) = x(1);
    dxdt(1) = u(0);
    return dxdt;
}

SymbolicVector brachistochrone_ode(const SymbolicVector &state, const SymbolicVector &control,
                                   const SymbolicScalar & /*t*/) {
    constexpr double g = 9.81;
    SymbolicScalar v = state(2);
    SymbolicScalar theta = control(0);

    SymbolicVector dxdt(3);
    dxdt(0) = v * metis::sin(theta);
    dxdt(1) = -v * metis::cos(theta);
    dxdt(2) = g * metis::cos(theta);
    return dxdt;
}

double solve_double_integrator_birkhoff(BirkhoffScheme scheme, int n_nodes) {
    Opti opti;
    BirkhoffPseudospectral bk(opti);

    BirkhoffPseudospectralOptions opts;
    opts.scheme = scheme;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = bk.setup(2, 1, 0.0, 1.0, opts);
    (void)x;
    (void)tau;

    bk.set_dynamics(double_integrator_ode);
    bk.add_dynamics_constraints();

    bk.set_initial_state(NumericVector{{0.0, 0.0}});
    bk.set_final_state(NumericVector{{1.0, 0.0}});

    opti.subject_to_bounds(u.col(0), -20.0, 20.0);

    SymbolicVector integrand(bk.n_nodes());
    for (int k = 0; k < bk.n_nodes(); ++k) {
        integrand(k) = u(k, 0) * u(k, 0);
    }

    SymbolicScalar J = bk.quadrature(integrand);
    opti.minimize(J);

    auto sol = opti.solve({.verbose = false});
    return sol.value(J);
}

double solve_brachistochrone_birkhoff(int n_nodes) {
    Opti opti;
    BirkhoffPseudospectral bk(opti);

    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    BirkhoffPseudospectralOptions opts;
    opts.scheme = BirkhoffScheme::LGL;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = bk.setup(3, 1, 0.0, T, opts);
    (void)tau;

    bk.set_dynamics(brachistochrone_ode);
    bk.add_dynamics_constraints();

    bk.set_initial_state(NumericVector{{0.0, 10.0, 0.001}});
    bk.set_final_state(0, 10.0);
    bk.set_final_state(1, 5.0);

    opti.subject_to_bounds(u.col(0), 0.01, M_PI - 0.01);
    opti.subject_to_lower(x.col(2), 0.0);
    opti.minimize(T);

    auto sol = opti.solve({.max_iter = 500, .verbose = false});
    return sol.value(T);
}

double solve_double_integrator_classical_ps(int n_nodes) {
    Opti opti;
    Pseudospectral ps(opti);

    PseudospectralOptions opts;
    opts.scheme = PseudospectralScheme::LGL;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = ps.setup(2, 1, 0.0, 1.0, opts);
    (void)x;
    (void)tau;

    ps.set_dynamics(double_integrator_ode);
    ps.add_dynamics_constraints();

    ps.set_initial_state(NumericVector{{0.0, 0.0}});
    ps.set_final_state(NumericVector{{1.0, 0.0}});

    opti.subject_to_bounds(u.col(0), -20.0, 20.0);

    SymbolicVector integrand(ps.n_nodes());
    for (int k = 0; k < ps.n_nodes(); ++k) {
        integrand(k) = u(k, 0) * u(k, 0);
    }
    SymbolicScalar J = ps.quadrature(integrand);
    opti.minimize(J);

    auto sol = opti.solve({.verbose = false});
    return sol.value(J);
}

} // namespace

TEST(BirkhoffPseudospectralTests, DoubleIntegratorLGLObjectiveMatchesAnalytic) {
    const double J = solve_double_integrator_birkhoff(BirkhoffScheme::LGL, 21);
    EXPECT_NEAR(J, 12.0, 1e-6);
}

TEST(BirkhoffPseudospectralTests, DoubleIntegratorCGLObjectiveMatchesAnalytic) {
    const double J = solve_double_integrator_birkhoff(BirkhoffScheme::CGL, 21);
    EXPECT_NEAR(J, 12.0, 1e-6);
}

TEST(BirkhoffPseudospectralTests, DoubleIntegratorMatchesClassicalPseudospectral) {
    const double J_bk = solve_double_integrator_birkhoff(BirkhoffScheme::LGL, 21);
    const double J_ps = solve_double_integrator_classical_ps(21);
    EXPECT_NEAR(J_bk, J_ps, 1e-7);
}

TEST(BirkhoffPseudospectralTests, BrachistochroneFreeFinalTimeLGL) {
    const double T_opt = solve_brachistochrone_birkhoff(31);
    EXPECT_NEAR(T_opt, 1.80185, 2e-3);
}

TEST(BirkhoffPseudospectralTests, HighNodeCountRemainsStable) {
    const double J = solve_double_integrator_birkhoff(BirkhoffScheme::LGL, 51);
    EXPECT_NEAR(J, 12.0, 1e-5);
}
