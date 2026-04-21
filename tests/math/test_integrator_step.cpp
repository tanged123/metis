#include "../utils/TestUtils.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/IntegratorStep.hpp>

// ============================================================================
// Tests for euler_step
// ============================================================================

TEST(IntegratorStep, EulerStepExponential) {
    // dy/dt = -y, y(0) = 1
    // After one step with dt = 0.1: y ≈ 1 + 0.1*(-1) = 0.9
    metis::NumericVector y0(1);
    y0(0) = 1.0;

    auto y1 =
        metis::euler_step([](double t, const metis::NumericVector &y) { return -y; }, y0, 0.0, 0.1);

    // Euler step: y1 = y0 + dt * (-y0) = 1 - 0.1 = 0.9
    EXPECT_NEAR(y1(0), 0.9, 1e-12);
}

TEST(IntegratorStep, EulerStepLinear) {
    // dy/dt = 2, y(0) = 0
    // After one step with dt = 0.5: y = 0 + 0.5*2 = 1.0
    metis::NumericVector y0(1);
    y0(0) = 0.0;

    auto y1 = metis::euler_step(
        [](double t, const metis::NumericVector &y) {
            metis::NumericVector dydt(1);
            dydt(0) = 2.0;
            return dydt;
        },
        y0, 0.0, 0.5);

    EXPECT_NEAR(y1(0), 1.0, 1e-12);
}

// ============================================================================
// Tests for rk2_step
// ============================================================================

TEST(IntegratorStep, RK2StepExponential) {
    // dy/dt = -y, y(0) = 1
    // RK2 should be more accurate than Euler
    metis::NumericVector y0(1);
    y0(0) = 1.0;
    double dt = 0.1;

    auto y1 =
        metis::rk2_step([](double t, const metis::NumericVector &y) { return -y; }, y0, 0.0, dt);

    double exact = std::exp(-dt);
    // RK2 is 2nd order, so error should be O(dt^3) ≈ 1e-3
    EXPECT_NEAR(y1(0), exact, 1e-3);
}

// ============================================================================
// Tests for rk4_step
// ============================================================================

TEST(IntegratorStep, RK4StepExponential) {
    // dy/dt = -y, y(0) = 1
    // RK4 should be very accurate
    metis::NumericVector y0(1);
    y0(0) = 1.0;
    double dt = 0.1;

    auto y1 =
        metis::rk4_step([](double t, const metis::NumericVector &y) { return -y; }, y0, 0.0, dt);

    double exact = std::exp(-dt);
    // RK4 is 4th order, error should be O(dt^5) ≈ 1e-5
    EXPECT_NEAR(y1(0), exact, 1e-5);
}

TEST(IntegratorStep, RK4StepHarmonicOscillator) {
    // y'' = -ω²y => state = [y, v], d/dt[y, v] = [v, -ω²y]
    // y(0) = 1, v(0) = 0 => y(t) = cos(ωt)
    double omega = 2.0;
    metis::NumericVector state0(2);
    state0 << 1.0, 0.0;

    auto state1 = metis::rk4_step(
        [omega](double t, const metis::NumericVector &s) {
            metis::NumericVector ds(2);
            ds << s(1), -omega * omega * s(0);
            return ds;
        },
        state0, 0.0, 0.1);

    double t1 = 0.1;
    double y_exact = std::cos(omega * t1);
    double v_exact = -omega * std::sin(omega * t1);

    EXPECT_NEAR(state1(0), y_exact, 1e-5);
    EXPECT_NEAR(state1(1), v_exact, 1e-4);
}

TEST(IntegratorStep, RK4MultipleSteps) {
    // Integrate dy/dt = -λy over [0, 1] using multiple RK4 steps
    double lambda = 0.5;
    double y0_val = 2.5;
    int n_steps = 100;
    double dt = 1.0 / n_steps;

    metis::NumericVector y(1);
    y(0) = y0_val;
    double t = 0.0;

    for (int i = 0; i < n_steps; ++i) {
        y = metis::rk4_step(
            [lambda](double t, const metis::NumericVector &y) { return -lambda * y; }, y, t, dt);
        t += dt;
    }

    double exact = y0_val * std::exp(-lambda * 1.0);
    EXPECT_NEAR(y(0), exact, 1e-8);
}

// ============================================================================
// Tests for structure-preserving second-order steppers
// ============================================================================

TEST(IntegratorStep, StormerVerletEnergyBoundedOnHarmonicOscillator) {
    metis::NumericVector q(1);
    metis::NumericVector v(1);
    q(0) = 1.0;
    v(0) = 0.0;

    const double omega = 1.0;
    const double dt = 0.2;
    const int n_steps = 2000;
    const double energy0 = 0.5 * (v.squaredNorm() + omega * omega * q.squaredNorm());
    double max_energy_drift = 0.0;
    double t = 0.0;

    for (int i = 0; i < n_steps; ++i) {
        auto step = metis::stormer_verlet_step(
            [omega](double time, const metis::NumericVector &q_state) {
                return (-omega * omega * q_state).eval();
            },
            q, v, t, dt);
        q = step.q;
        v = step.v;
        t += dt;

        const double energy = 0.5 * (v.squaredNorm() + omega * omega * q.squaredNorm());
        max_energy_drift = std::max(max_energy_drift, std::abs(energy - energy0));
    }

    EXPECT_LT(max_energy_drift, 6e-3);
}

TEST(IntegratorStep, Rkn4StepHarmonicOscillator) {
    metis::NumericVector q0(1);
    metis::NumericVector v0(1);
    q0(0) = 1.0;
    v0(0) = 0.0;

    const double omega = 2.0;
    const double dt = 0.1;

    auto step = metis::rkn4_step(
        [omega](double time, const metis::NumericVector &q) { return (-omega * omega * q).eval(); },
        q0, v0, 0.0, dt);

    EXPECT_NEAR(step.q(0), std::cos(omega * dt), 1e-6);
    EXPECT_NEAR(step.v(0), -omega * std::sin(omega * dt), 5e-6);
}

// ============================================================================
// Tests for rk45_step (adaptive step with error estimate)
// ============================================================================

TEST(IntegratorStep, RK45StepExponential) {
    // dy/dt = -y, y(0) = 1
    metis::NumericVector y0(1);
    y0(0) = 1.0;
    double dt = 0.1;

    auto result =
        metis::rk45_step([](double t, const metis::NumericVector &y) { return -y; }, y0, 0.0, dt);

    double exact = std::exp(-dt);

    // 5th order solution should be very accurate
    EXPECT_NEAR(result.y5(0), exact, 1e-6);

    // 4th order solution should also be accurate but slightly less
    EXPECT_NEAR(result.y4(0), exact, 1e-5);

    // Error estimate should be small (difference between y4 and y5)
    EXPECT_LT(result.error, 1e-6);
}

TEST(IntegratorStep, RK45ErrorEstimateStiff) {
    // For a stiff-ish problem, error should be larger with bigger steps
    metis::NumericVector y0(1);
    y0(0) = 1.0;

    auto result_small = metis::rk45_step(
        [](double t, const metis::NumericVector &y) { return -10.0 * y; }, y0, 0.0, 0.01);

    auto result_large = metis::rk45_step(
        [](double t, const metis::NumericVector &y) { return -10.0 * y; }, y0, 0.0, 0.1);

    // Larger step should have larger error estimate
    EXPECT_LT(result_small.error, result_large.error);
}

// ============================================================================
// Convergence order verification
// ============================================================================

TEST(IntegratorStep, ConvergenceOrderEuler) {
    // Verify O(h) convergence for Euler
    // Error should halve when step size halves (approximately)
    metis::NumericVector y0(1);
    y0(0) = 1.0;

    auto compute_error = [&](double dt) {
        auto y1 = metis::euler_step([](double t, const metis::NumericVector &y) { return -y; }, y0,
                                    0.0, dt);
        return std::abs(y1(0) - std::exp(-dt));
    };

    double err1 = compute_error(0.1);
    double err2 = compute_error(0.05);

    // Ratio should be ~2 for O(h) method (but can be higher due to higher-order terms)
    double ratio = err1 / err2;
    EXPECT_GT(ratio, 1.8); // At least O(h) convergence
}

TEST(IntegratorStep, ConvergenceOrderRK4) {
    // Verify O(h^4) convergence for RK4
    // Error should decrease by factor of 16 when step size halves
    metis::NumericVector y0(1);
    y0(0) = 1.0;

    auto compute_error = [&](double dt) {
        auto y1 = metis::rk4_step([](double t, const metis::NumericVector &y) { return -y; }, y0,
                                  0.0, dt);
        return std::abs(y1(0) - std::exp(-dt));
    };

    double err1 = compute_error(0.1);
    double err2 = compute_error(0.05);

    // Ratio should be ~16 for O(h^4) method (but can be higher due to higher-order terms)
    double ratio = err1 / err2;
    EXPECT_GT(ratio, 14.0); // At least O(h^4) convergence
}

// ============================================================================
// Symbolic mode tests
// ============================================================================

TEST(IntegratorStep, SymbolicRK4Step) {
    // Test that rk4_step produces valid symbolic expressions
    auto x = metis::sym_vec("x", 1);
    auto t = metis::sym("t");
    auto dt = metis::sym("dt");

    // Simple linear ODE: dx/dt = -x
    auto x_next = metis::rk4_step([](auto t, const auto &x) { return -x; }, x, t, dt);

    // Verify we can create a CasADi function from the result
    casadi::Function step_fn("step_fn", {metis::to_mx(x), t, dt}, {metis::to_mx(x_next)});

    // Evaluate numerically
    std::vector<casadi::DM> args = {
        casadi::DM(1.0), // x = 1
        casadi::DM(0.0), // t = 0
        casadi::DM(0.1)  // dt = 0.1
    };
    auto res = step_fn(args);

    double x_next_numeric = static_cast<double>(res[0]);
    double exact = std::exp(-0.1);

    EXPECT_NEAR(x_next_numeric, exact, 1e-5);
}

TEST(IntegratorStep, SymbolicEulerStep) {
    // Test Euler step in symbolic mode
    auto x = metis::sym_vec("x", 1);
    auto t = metis::sym("t");
    auto dt = metis::sym("dt");

    auto x_next = metis::euler_step([](auto t, const auto &x) { return -x; }, x, t, dt);

    casadi::Function step_fn("euler_step", {metis::to_mx(x), t, dt}, {metis::to_mx(x_next)});

    std::vector<casadi::DM> args = {casadi::DM(1.0), casadi::DM(0.0), casadi::DM(0.1)};
    auto res = step_fn(args);

    double x_next_numeric = static_cast<double>(res[0]);
    // Euler: x_next = x + dt * (-x) = 1 - 0.1 = 0.9
    EXPECT_NEAR(x_next_numeric, 0.9, 1e-10);
}

TEST(IntegratorStep, SymbolicStormerVerletStep) {
    auto q = metis::sym_vec("q", 1);
    auto v = metis::sym_vec("v", 1);
    auto t = metis::sym("t");
    auto dt = metis::sym("dt");

    auto step = metis::stormer_verlet_step(
        [](auto time, const auto &q_state) { return (-q_state).eval(); }, q, v, t, dt);

    casadi::Function step_fn("stormer_verlet_step", {metis::to_mx(q), metis::to_mx(v), t, dt},
                             {metis::to_mx(step.q), metis::to_mx(step.v)});

    std::vector<casadi::DM> args = {
        casadi::DM(1.0),
        casadi::DM(0.0),
        casadi::DM(0.0),
        casadi::DM(0.1),
    };
    auto res = step_fn(args);

    EXPECT_NEAR(static_cast<double>(res[0]), 0.995, 1e-10);
    EXPECT_NEAR(static_cast<double>(res[1]), -0.09975, 1e-10);
}
