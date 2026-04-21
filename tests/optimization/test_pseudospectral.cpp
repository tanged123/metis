/**
 * @file test_pseudospectral.cpp
 * @brief Tests for pseudospectral transcription
 */

#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Trig.hpp>
#include <metis/optimization/Collocation.hpp>
#include <metis/optimization/Opti.hpp>
#include <metis/optimization/Pseudospectral.hpp>

#include <array>
#include <cmath>
#include <vector>

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

double solve_double_integrator_ps(PseudospectralScheme scheme, int n_nodes) {
    Opti opti;
    Pseudospectral ps(opti);

    PseudospectralOptions opts;
    opts.scheme = scheme;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = ps.setup(2, 1, 0.0, 1.0, opts);
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

double solve_brachistochrone_ps(int n_nodes) {
    Opti opti;
    Pseudospectral ps(opti);

    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    PseudospectralOptions opts;
    opts.scheme = PseudospectralScheme::LGL;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = ps.setup(3, 1, 0.0, T, opts);
    (void)tau;

    ps.set_dynamics(brachistochrone_ode);
    ps.add_dynamics_constraints();

    ps.set_initial_state(NumericVector{{0.0, 10.0, 0.001}});
    ps.set_final_state(0, 10.0);
    ps.set_final_state(1, 5.0);

    opti.subject_to_bounds(u.col(0), 0.01, M_PI - 0.01);
    opti.subject_to_lower(x.col(2), 0.0);
    opti.minimize(T);

    auto sol = opti.solve({.max_iter = 500, .verbose = false});
    return sol.value(T);
}

double solve_brachistochrone_dc(int n_nodes) {
    Opti opti;
    DirectCollocation dc(opti);

    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    CollocationOptions opts;
    opts.scheme = CollocationScheme::HermiteSimpson;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = dc.setup(3, 1, 0.0, T, opts);
    (void)tau;

    dc.set_dynamics(brachistochrone_ode);
    dc.add_dynamics_constraints();

    dc.set_initial_state(NumericVector{{0.0, 10.0, 0.001}});
    dc.set_final_state(0, 10.0);
    dc.set_final_state(1, 5.0);

    opti.subject_to_bounds(u.col(0), 0.01, M_PI - 0.01);
    opti.subject_to_lower(x.col(2), 0.0);
    opti.minimize(T);

    auto sol = opti.solve({.max_iter = 500, .verbose = false});
    return sol.value(T);
}

double solve_exp_decay_terminal_error(int n_nodes) {
    Opti opti;
    Pseudospectral ps(opti);

    PseudospectralOptions opts;
    opts.scheme = PseudospectralScheme::LGL;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = ps.setup(1, 1, 0.0, 1.0, opts);
    (void)tau;

    ps.set_dynamics(
        [](const SymbolicVector &xv, const SymbolicVector &uv, const SymbolicScalar & /*t*/) {
            SymbolicVector dxdt(1);
            dxdt(0) = -xv(0) + uv(0);
            return dxdt;
        });
    ps.add_dynamics_constraints();
    ps.set_initial_state(NumericVector{{1.0}});

    SymbolicVector integrand(ps.n_nodes());
    for (int k = 0; k < ps.n_nodes(); ++k) {
        integrand(k) = u(k, 0) * u(k, 0);
    }
    opti.minimize(ps.quadrature(integrand));
    auto sol = opti.solve({.verbose = false});

    auto x_sol = sol.value(x);
    return std::abs(x_sol(n_nodes - 1, 0) - std::exp(-1.0));
}

} // namespace

TEST(PseudospectralTests, DoubleIntegratorLGLObjectiveMatchesAnalytic) {
    const double J = solve_double_integrator_ps(PseudospectralScheme::LGL, 21);
    EXPECT_NEAR(J, 12.0, 1e-6);
}

TEST(PseudospectralTests, DoubleIntegratorCGLObjectiveMatchesAnalytic) {
    const double J = solve_double_integrator_ps(PseudospectralScheme::CGL, 21);
    EXPECT_NEAR(J, 12.0, 1e-6);
}

TEST(PseudospectralTests, DoubleIntegratorLGLAndCGLAgree) {
    const double J_lgl = solve_double_integrator_ps(PseudospectralScheme::LGL, 21);
    const double J_cgl = solve_double_integrator_ps(PseudospectralScheme::CGL, 21);
    EXPECT_NEAR(J_lgl, J_cgl, 1e-7);
}

TEST(PseudospectralTests, HarmonicOscillatorEnergyConservation) {
    Opti opti;
    Pseudospectral ps(opti);

    PseudospectralOptions opts;
    opts.scheme = PseudospectralScheme::LGL;
    opts.n_nodes = 41;

    auto [x, u, tau] = ps.setup(2, 1, 0.0, 2.0 * M_PI, opts);
    (void)tau;

    ps.set_dynamics([](const SymbolicVector &xv, const SymbolicVector &uv, const SymbolicScalar &) {
        SymbolicVector dxdt(2);
        dxdt(0) = xv(1);
        dxdt(1) = -xv(0) + uv(0);
        return dxdt;
    });
    ps.add_dynamics_constraints();
    ps.set_initial_state(NumericVector{{1.0, 0.0}});
    ps.set_final_state(NumericVector{{1.0, 0.0}});

    SymbolicVector integrand(ps.n_nodes());
    for (int k = 0; k < ps.n_nodes(); ++k) {
        integrand(k) = u(k, 0) * u(k, 0);
    }
    SymbolicScalar J = ps.quadrature(integrand);
    opti.minimize(J);

    auto sol = opti.solve({.verbose = false});

    auto x_sol = sol.value(x);
    auto u_sol = sol.value(u);
    const int last = ps.n_nodes() - 1;

    const double E0 = 0.5 * (x_sol(0, 1) * x_sol(0, 1) + x_sol(0, 0) * x_sol(0, 0));
    const double Ef = 0.5 * (x_sol(last, 1) * x_sol(last, 1) + x_sol(last, 0) * x_sol(last, 0));

    EXPECT_NEAR(E0, 0.5, 1e-6);
    EXPECT_NEAR(Ef, 0.5, 1e-6);
    EXPECT_NEAR(x_sol(last, 0), 1.0, 1e-6);
    EXPECT_NEAR(x_sol(last, 1), 0.0, 1e-6);
    EXPECT_LT(sol.value(J), 1e-8);
    EXPECT_LT(u_sol.cwiseAbs().maxCoeff(), 1e-4);
}

TEST(PseudospectralTests, BrachistochroneFreeFinalTimeLGL) {
    const double T_opt = solve_brachistochrone_ps(31);
    EXPECT_NEAR(T_opt, 1.80185, 2e-3);
}

TEST(PseudospectralTests, BrachistochroneMatchesCollocation) {
    const double T_ps = solve_brachistochrone_ps(31);
    const double T_dc = solve_brachistochrone_dc(31);
    EXPECT_NEAR(T_ps, T_dc, 2e-3);
}

TEST(PseudospectralTests, ExponentialDecayConvergenceStudy) {
    const std::array<int, 5> nodes = {5, 9, 13, 17, 21};
    std::vector<double> errors;
    errors.reserve(nodes.size());

    for (int n : nodes) {
        errors.push_back(solve_exp_decay_terminal_error(n));
    }

    EXPECT_LT(errors.back(), errors.front() * 1e-4);
    EXPECT_LT(errors.back(), 1e-10);
}
