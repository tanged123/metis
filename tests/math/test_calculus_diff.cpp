#include <gtest/gtest.h>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/Arithmetic.hpp>
#include <janus/math/Calculus.hpp>
#include <janus/math/Trig.hpp>
#include <janus/utils/GTestDiffTest.hpp>

// ============================================================================
// janus::trapz — trapezoidal integration
// ============================================================================

TEST(CalculusDiffTests, Trapz) {
    // trapz([y0, y1, y2], [x0, x1, x2]) — integral depends on y values
    // Fix x grid, differentiate w.r.t. y values
    janus::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2) {
            using S = std::decay_t<decltype(y0)>;
            janus::JanusVector<S> y(3), x(3);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            x(0) = S(0.0);
            x(1) = S(1.0);
            x(2) = S(2.0);
            return janus::trapz(y, x);
        },
        {{1.0, 2.0, 3.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 1.0}});
}

// ============================================================================
// janus::cumtrapz — cumulative trapezoidal integration
// ============================================================================

TEST(CalculusDiffTests, CumtrapzLastElement) {
    // cumtrapz returns a vector; test the last element (total integral)
    janus::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2) {
            using S = std::decay_t<decltype(y0)>;
            janus::JanusVector<S> y(3), x(3);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            x(0) = S(0.0);
            x(1) = S(1.0);
            x(2) = S(2.0);
            auto result = janus::cumtrapz(y, x);
            return result(2); // total integral
        },
        {{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}});
}

TEST(CalculusDiffTests, CumtrapzMiddleElement) {
    janus::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2) {
            using S = std::decay_t<decltype(y0)>;
            janus::JanusVector<S> y(3), x(3);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            x(0) = S(0.0);
            x(1) = S(1.0);
            x(2) = S(2.0);
            auto result = janus::cumtrapz(y, x);
            return result(1); // partial integral
        },
        {{1.0, 2.0, 3.0}});
}

// ============================================================================
// janus::gradient_1d — numerical gradient
// ============================================================================

TEST(CalculusDiffTests, Gradient1d) {
    // gradient_1d returns a vector; test the interior point gradient
    janus::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2, auto y3) {
            using S = std::decay_t<decltype(y0)>;
            janus::JanusVector<S> y(4), x(4);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            y(3) = y3;
            x(0) = S(0.0);
            x(1) = S(1.0);
            x(2) = S(2.0);
            x(3) = S(3.0);
            auto grad = janus::gradient_1d(y, x);
            return grad(1); // interior central difference
        },
        {{0.0, 1.0, 4.0, 9.0}, {1.0, 2.0, 3.0, 4.0}});
}

// ============================================================================
// janus::gradient — second-order accurate gradient
// ============================================================================

TEST(CalculusDiffTests, GradientUniform) {
    // gradient with uniform spacing
    janus::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2, auto y3) {
            using S = std::decay_t<decltype(y0)>;
            janus::JanusVector<S> y(4);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            y(3) = y3;
            auto grad = janus::gradient(y, 1.0);
            return grad(2); // interior point
        },
        {{0.0, 1.0, 4.0, 9.0}, {1.0, 3.0, 5.0, 7.0}});
}

// ============================================================================
// janus::diff — adjacent differences
// ============================================================================

TEST(CalculusDiffTests, Diff) {
    // diff returns v[i+1] - v[i]; test one element
    janus::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2) {
            using S = std::decay_t<decltype(y0)>;
            janus::JanusVector<S> y(3);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            auto d = janus::diff(y);
            return d(0); // y1 - y0
        },
        {{1.0, 3.0, 7.0}, {-1.0, 0.0, 2.0}});
}
