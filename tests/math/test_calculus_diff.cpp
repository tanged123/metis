#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Arithmetic.hpp>
#include <metis/math/Calculus.hpp>
#include <metis/math/Trig.hpp>
#include <metis/utils/GTestDiffTest.hpp>

// ============================================================================
// metis::trapz — trapezoidal integration
// ============================================================================

TEST(CalculusDiffTests, Trapz) {
    // trapz([y0, y1, y2], [x0, x1, x2]) — integral depends on y values
    // Fix x grid, differentiate w.r.t. y values
    metis::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> y(3), x(3);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            x(0) = S(0.0);
            x(1) = S(1.0);
            x(2) = S(2.0);
            return metis::trapz(y, x);
        },
        {{1.0, 2.0, 3.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 1.0}});
}

// ============================================================================
// metis::cumtrapz — cumulative trapezoidal integration
// ============================================================================

TEST(CalculusDiffTests, CumtrapzLastElement) {
    // cumtrapz returns a vector; test the last element (total integral)
    metis::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> y(3), x(3);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            x(0) = S(0.0);
            x(1) = S(1.0);
            x(2) = S(2.0);
            auto result = metis::cumtrapz(y, x);
            return result(2); // total integral
        },
        {{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}});
}

TEST(CalculusDiffTests, CumtrapzMiddleElement) {
    metis::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> y(3), x(3);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            x(0) = S(0.0);
            x(1) = S(1.0);
            x(2) = S(2.0);
            auto result = metis::cumtrapz(y, x);
            return result(1); // partial integral
        },
        {{1.0, 2.0, 3.0}});
}

// ============================================================================
// metis::gradient_1d — numerical gradient
// ============================================================================

TEST(CalculusDiffTests, Gradient1d) {
    // gradient_1d returns a vector; test the interior point gradient
    metis::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2, auto y3) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> y(4), x(4);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            y(3) = y3;
            x(0) = S(0.0);
            x(1) = S(1.0);
            x(2) = S(2.0);
            x(3) = S(3.0);
            auto grad = metis::gradient_1d(y, x);
            return grad(1); // interior central difference
        },
        {{0.0, 1.0, 4.0, 9.0}, {1.0, 2.0, 3.0, 4.0}});
}

// ============================================================================
// metis::gradient — second-order accurate gradient
// ============================================================================

TEST(CalculusDiffTests, GradientUniform) {
    // gradient with uniform spacing
    metis::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2, auto y3) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> y(4);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            y(3) = y3;
            auto grad = metis::gradient(y, 1.0);
            return grad(2); // interior point
        },
        {{0.0, 1.0, 4.0, 9.0}, {1.0, 3.0, 5.0, 7.0}});
}

// ============================================================================
// metis::diff — adjacent differences
// ============================================================================

TEST(CalculusDiffTests, Diff) {
    // diff returns v[i+1] - v[i]; test one element
    metis::diff_test::expect_differentiable(
        [](auto y0, auto y1, auto y2) {
            using S = std::decay_t<decltype(y0)>;
            metis::MetisVector<S> y(3);
            y(0) = y0;
            y(1) = y1;
            y(2) = y2;
            auto d = metis::diff(y);
            return d(0); // y1 - y0
        },
        {{1.0, 3.0, 7.0}, {-1.0, 0.0, 2.0}});
}
