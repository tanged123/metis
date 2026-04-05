#include <gtest/gtest.h>
#include <janus/math/Interpolate.hpp>
#include <janus/utils/GTestDiffTest.hpp>

// ============================================================================
// 1D Interpolation — Interpolator::operator()(Scalar)
// Grid is structural (captured), query point is differentiated
// ============================================================================

TEST(InterpolateDiffTests, Linear1D) {
    // Linear interpolation of y = x^2 on [0, 4]
    janus::NumericVector x(5), y(5);
    x << 0.0, 1.0, 2.0, 3.0, 4.0;
    y << 0.0, 1.0, 4.0, 9.0, 16.0;
    janus::Interpolator interp(x, y, janus::InterpolationMethod::Linear);

    janus::diff_test::expect_differentiable([&interp](auto q) { return interp(q); },
                                            {{0.5}, {1.5}, {2.5}, {3.5}});
}

TEST(InterpolateDiffTests, BSpline1D) {
    // B-spline interpolation of sin(x) — C2 smooth, good for optimization
    janus::NumericVector x(7), y(7);
    for (int i = 0; i < 7; ++i) {
        x(i) = i * 0.5;
        y(i) = std::sin(x(i));
    }
    janus::Interpolator interp(x, y, janus::InterpolationMethod::BSpline);

    janus::diff_test::expect_differentiable([&interp](auto q) { return interp(q); },
                                            {{0.25}, {0.75}, {1.5}, {2.0}});
}

// ============================================================================
// 2D Interpolation — via janus::interpn free function
// ============================================================================

TEST(InterpolateDiffTests, Linear2D) {
    // 2D linear interpolation of f(x, y) = x + 2*y on a 3x3 grid
    janus::NumericVector xg(3), yg(3);
    xg << 0.0, 1.0, 2.0;
    yg << 0.0, 1.0, 2.0;
    std::vector<janus::NumericVector> points = {xg, yg};

    // Values in Fortran order
    janus::NumericVector vals(9);
    for (int iy = 0; iy < 3; ++iy) {
        for (int ix = 0; ix < 3; ++ix) {
            vals(iy * 3 + ix) = xg(ix) + 2.0 * yg(iy);
        }
    }

    janus::diff_test::expect_differentiable(
        [&points, &vals](auto qx, auto qy) {
            using S = std::decay_t<decltype(qx)>;
            janus::JanusMatrix<S> xi(1, 2);
            xi(0, 0) = qx;
            xi(0, 1) = qy;
            auto result = janus::interpn<S>(points, vals, xi, janus::InterpolationMethod::Linear);
            return result(0);
        },
        {{0.5, 0.5}, {1.0, 1.5}, {0.3, 0.7}});
}

// ============================================================================
// 1D Nearest — dual-mode only (non-differentiable, but modes should agree)
// ============================================================================

TEST(InterpolateDiffTests, Nearest1DDualMode) {
    janus::NumericVector x(5), y(5);
    x << 0.0, 1.0, 2.0, 3.0, 4.0;
    y << 0.0, 1.0, 4.0, 9.0, 16.0;
    janus::Interpolator interp(x, y, janus::InterpolationMethod::Nearest);

    janus::diff_test::expect_dual_mode([&interp](auto q) { return interp(q); },
                                       {{0.3}, {0.7}, {1.2}, {2.8}, {3.5}});
}

// Note: Hermite throws in symbolic mode (uses runtime comparisons incompatible with MX).
// This is a documented limitation. Linear and BSpline are the optimization-grade methods.
