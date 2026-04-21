#include <gtest/gtest.h>
#include <metis/math/Arithmetic.hpp>
#include <metis/math/IntegratorStep.hpp>
#include <metis/math/Trig.hpp>
#include <metis/utils/GTestDiffTest.hpp>

// ============================================================================
// euler_step — actual IntegratorStep.hpp:62 API
// ============================================================================

TEST(IntegrateDiffTests, EulerStep) {
    // Euler step for dy/dt = -y: y_next = y + dt * (-y)
    // Test as function of initial state y0
    metis::diff_test::expect_differentiable(
        [](auto y0) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> state(1);
            state(0) = y0;
            S t = 0.0;
            S dt = 0.1;
            auto result = metis::euler_step(
                [](S t_, const metis::MetisVector<S> &y) { return (-y).eval(); }, state, t, dt);
            return result(0);
        },
        {{1.0}, {2.0}, {-0.5}});
}

// ============================================================================
// rk2_step — actual IntegratorStep.hpp:88 API
// ============================================================================

TEST(IntegrateDiffTests, RK2Step) {
    // RK2 step for dy/dt = -2*y
    metis::diff_test::expect_differentiable(
        [](auto y0) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> state(1);
            state(0) = y0;
            S t = 0.0;
            S dt = 0.05;
            auto result = metis::rk2_step(
                [](S t_, const metis::MetisVector<S> &y) { return (-2.0 * y).eval(); }, state, t,
                dt);
            return result(0);
        },
        {{1.0}, {0.5}, {3.0}});
}

// ============================================================================
// rk4_step — actual IntegratorStep.hpp:131 API
// ============================================================================

TEST(IntegrateDiffTests, RK4Step) {
    // RK4 step for harmonic oscillator: [y,v]' = [v, -y]
    // Test as function of initial (y0, v0)
    metis::diff_test::expect_differentiable(
        [](auto y0, auto v0) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> state(2);
            state(0) = y0;
            state(1) = v0;
            S t = 0.0;
            S dt = 0.01;
            auto result = metis::rk4_step(
                [](S t_, const metis::MetisVector<S> &s) {
                    metis::MetisVector<S> ds(2);
                    ds(0) = s(1);
                    ds(1) = -s(0);
                    return ds;
                },
                state, t, dt);
            return result(0); // y at next step
        },
        {{1.0, 0.0}, {0.0, 1.0}, {0.5, -0.3}});
}

TEST(IntegrateDiffTests, RK4StepVelocity) {
    // Same as above but test velocity output
    metis::diff_test::expect_differentiable(
        [](auto y0, auto v0) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> state(2);
            state(0) = y0;
            state(1) = v0;
            S t = 0.0;
            S dt = 0.01;
            auto result = metis::rk4_step(
                [](S t_, const metis::MetisVector<S> &s) {
                    metis::MetisVector<S> ds(2);
                    ds(0) = s(1);
                    ds(1) = -s(0);
                    return ds;
                },
                state, t, dt);
            return result(1); // v at next step
        },
        {{1.0, 0.0}, {0.0, 1.0}});
}

// ============================================================================
// stormer_verlet_step — actual IntegratorStep.hpp:167 API
// ============================================================================

TEST(IntegrateDiffTests, StormerVerletStepQ) {
    // Stormer-Verlet for q'' = -q (harmonic oscillator), position output
    metis::diff_test::expect_differentiable(
        [](auto q0, auto v0) {
            using S = std::decay_t<decltype(q0)>;
            metis::MetisVector<S> q(1), v(1);
            q(0) = q0;
            v(0) = v0;
            S t = 0.0;
            S dt = 0.01;
            auto result = metis::stormer_verlet_step(
                [](S t_, const metis::MetisVector<S> &pos) { return (-pos).eval(); }, q, v, t, dt);
            return result.q(0);
        },
        {{1.0, 0.0}, {0.0, 1.0}, {0.5, -0.3}});
}

TEST(IntegrateDiffTests, StormerVerletStepV) {
    // Stormer-Verlet velocity output
    metis::diff_test::expect_differentiable(
        [](auto q0, auto v0) {
            using S = std::decay_t<decltype(q0)>;
            metis::MetisVector<S> q(1), v(1);
            q(0) = q0;
            v(0) = v0;
            S t = 0.0;
            S dt = 0.01;
            auto result = metis::stormer_verlet_step(
                [](S t_, const metis::MetisVector<S> &pos) { return (-pos).eval(); }, q, v, t, dt);
            return result.v(0);
        },
        {{1.0, 0.0}, {0.0, 1.0}});
}

// ============================================================================
// rk45_step — actual IntegratorStep.hpp:250 API
// ============================================================================

TEST(IntegrateDiffTests, RK45StepY5) {
    // RK45 step for dy/dt = -y, test 5th-order output
    metis::diff_test::expect_differentiable(
        [](auto y0) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> state(1);
            state(0) = y0;
            S t = 0.0;
            S dt = 0.1;
            auto result = metis::rk45_step(
                [](S t_, const metis::MetisVector<S> &y) { return (-y).eval(); }, state, t, dt);
            return result.y5(0);
        },
        {{1.0}, {2.0}, {-0.5}});
}

TEST(IntegrateDiffTests, RK45StepY4) {
    // RK45 step, 4th-order output (used for error estimation)
    metis::diff_test::expect_differentiable(
        [](auto y0) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> state(1);
            state(0) = y0;
            S t = 0.0;
            S dt = 0.1;
            auto result = metis::rk45_step(
                [](S t_, const metis::MetisVector<S> &y) { return (-y).eval(); }, state, t, dt);
            return result.y4(0);
        },
        {{1.0}, {2.0}, {-0.5}});
}

TEST(IntegrateDiffTests, RK45StepError) {
    // RK45 step, error estimate
    metis::diff_test::expect_differentiable(
        [](auto y0) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> state(1);
            state(0) = y0;
            S t = 0.0;
            S dt = 0.1;
            auto result = metis::rk45_step(
                [](S t_, const metis::MetisVector<S> &y) { return (-y).eval(); }, state, t, dt);
            return result.error;
        },
        {{1.0}, {2.0}, {-0.5}});
}

// ============================================================================
// rkn4_step — actual IntegratorStep.hpp:193 API
// ============================================================================

TEST(IntegrateDiffTests, RKN4StepQ) {
    // RKN4 step for q'' = -q, position output
    metis::diff_test::expect_differentiable(
        [](auto q0, auto v0) {
            using S = std::decay_t<decltype(q0)>;
            metis::MetisVector<S> q(1), v(1);
            q(0) = q0;
            v(0) = v0;
            S t = 0.0;
            S dt = 0.01;
            auto result = metis::rkn4_step(
                [](S t_, const metis::MetisVector<S> &pos) { return (-pos).eval(); }, q, v, t, dt);
            return result.q(0);
        },
        {{1.0, 0.0}, {0.0, 1.0}});
}

TEST(IntegrateDiffTests, RKN4StepV) {
    // RKN4 step velocity output
    metis::diff_test::expect_differentiable(
        [](auto q0, auto v0) {
            using S = std::decay_t<decltype(q0)>;
            metis::MetisVector<S> q(1), v(1);
            q(0) = q0;
            v(0) = v0;
            S t = 0.0;
            S dt = 0.01;
            auto result = metis::rkn4_step(
                [](S t_, const metis::MetisVector<S> &pos) { return (-pos).eval(); }, q, v, t, dt);
            return result.v(0);
        },
        {{1.0, 0.0}, {0.0, 1.0}});
}
