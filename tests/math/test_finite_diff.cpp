#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/MetisError.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/FiniteDifference.hpp>

template <typename Scalar> void test_finite_difference() {
    using Vector = metis::MetisVector<Scalar>;

    // Case 1: Central difference for 1st derivative
    // Stencil: [-1, 0, 1] at x0=0
    Vector x(3);
    x << -1.0, 0.0, 1.0;

    auto coeffs = metis::finite_difference_coefficients(x, Scalar(0.0), 1);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(coeffs.size(), 3);
        EXPECT_NEAR(coeffs(0), -0.5, 1e-9);
        EXPECT_NEAR(coeffs(1), 0.0, 1e-9);
        EXPECT_NEAR(coeffs(2), 0.5, 1e-9);
    } else {
        auto coeffs_eval = metis::eval(coeffs);
        EXPECT_EQ(coeffs_eval.size(), 3);
        EXPECT_NEAR(coeffs_eval(0), -0.5, 1e-9);
        EXPECT_NEAR(coeffs_eval(1), 0.0, 1e-9);
        EXPECT_NEAR(coeffs_eval(2), 0.5, 1e-9);
    }

    // Case 2: Central difference for 2nd derivative
    // Stencil: [-1, 0, 1] at x0=0 -> [1, -2, 1]
    auto coeffs2 = metis::finite_difference_coefficients(x, Scalar(0.0), 2);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(coeffs2(0), 1.0, 1e-9);
        EXPECT_NEAR(coeffs2(1), -2.0, 1e-9);
        EXPECT_NEAR(coeffs2(2), 1.0, 1e-9);
    } else {
        auto coeffs2_eval = metis::eval(coeffs2);
        EXPECT_NEAR(coeffs2_eval(0), 1.0, 1e-9);
        EXPECT_NEAR(coeffs2_eval(1), -2.0, 1e-9);
        EXPECT_NEAR(coeffs2_eval(2), 1.0, 1e-9);
    }

    // Case 3: Forward difference (one-sided)
    // Stencil: [0, 1, 2] at x0=0 -> [-1.5, 2, -0.5]
    Vector x_fwd(3);
    x_fwd << 0.0, 1.0, 2.0;
    auto coeffs_fwd = metis::finite_difference_coefficients(x_fwd, Scalar(0.0), 1);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(coeffs_fwd(0), -1.5, 1e-9);
        EXPECT_NEAR(coeffs_fwd(1), 2.0, 1e-9);
        EXPECT_NEAR(coeffs_fwd(2), -0.5, 1e-9);
    } else {
        auto coeffs_fwd_eval = metis::eval(coeffs_fwd);
        EXPECT_NEAR(coeffs_fwd_eval(0), -1.5, 1e-9);
        EXPECT_NEAR(coeffs_fwd_eval(1), 2.0, 1e-9);
        EXPECT_NEAR(coeffs_fwd_eval(2), -0.5, 1e-9);
    }

    // Case 4: Non-uniform grid
    // x = [0, 1, 3] at x0=0
    // Analytic for f(x) = x^2 (derivative 2x at 0 is 0)
    // f(0)=0, f(1)=1, f(3)=9
    // coeffs dot [0, 1, 9] should be derivative approximations.
    // 1st derivative of x^2 is 2x -> at 0 is 0.
    // 2nd derivative is 2 -> at 0 is 2.

    // Let's check coefficients directly for 1st derivative?
    // Or check against property.
    Vector x_nonuni(3);
    x_nonuni << 0.0, 1.0, 3.0;
    auto coeffs_nu = metis::finite_difference_coefficients(x_nonuni, Scalar(0.0), 1);

    // Check property on f(x) = x
    // f(0)=0, f(1)=1, f(3)=3 -> approx deriv = c0*0 + c1*1 + c2*3 = 1
    // f(x) = 1 -> c0+c1+c2 = 0
    // f(x) = x^2 -> c0*0 + c1*1 + c2*9 = 0 (deriv at 0 is 0)

    // c1 + 3c2 = 1
    // c1 + 9c2 = 0
    // -> 6c2 = -1 -> c2 = -1/6
    // c1 = 1 - 3(-1/6) = 1.5
    // c0 = -c1 - c2 = -1.5 + 0.1666 = -1.333
    // c0 = -3/2 - (-1/6) = -9/6 + 1/6 = -8/6 = -4/3

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(coeffs_nu(0), -4.0 / 3.0, 1e-9);
        EXPECT_NEAR(coeffs_nu(1), 1.5, 1e-9);
        EXPECT_NEAR(coeffs_nu(2), -1.0 / 6.0, 1e-9);
    } else {
        auto c = metis::eval(coeffs_nu);
        EXPECT_NEAR(c(0), -4.0 / 3.0, 1e-9);
        EXPECT_NEAR(c(1), 1.5, 1e-9);
        EXPECT_NEAR(c(2), -1.0 / 6.0, 1e-9);
    }
}

TEST(FiniteDiffTests, Numeric) { test_finite_difference<double>(); }
TEST(FiniteDiffTests, Symbolic) { test_finite_difference<metis::SymbolicScalar>(); }

// =============================================================================
// Parse Integration Method Tests
// =============================================================================

TEST(FiniteDiffTests, ParseIntegrationMethod_Trapezoidal) {
    EXPECT_EQ(metis::parse_integration_method("trapezoidal"),
              metis::IntegrationMethod::Trapezoidal);
    EXPECT_EQ(metis::parse_integration_method("trapezoid"), metis::IntegrationMethod::Trapezoidal);
    EXPECT_EQ(metis::parse_integration_method("midpoint"), metis::IntegrationMethod::Trapezoidal);
}

TEST(FiniteDiffTests, ParseIntegrationMethod_ForwardEuler) {
    EXPECT_EQ(metis::parse_integration_method("forward_euler"),
              metis::IntegrationMethod::ForwardEuler);
    EXPECT_EQ(metis::parse_integration_method("forward euler"),
              metis::IntegrationMethod::ForwardEuler);
}

TEST(FiniteDiffTests, ParseIntegrationMethod_BackwardEuler) {
    EXPECT_EQ(metis::parse_integration_method("backward_euler"),
              metis::IntegrationMethod::BackwardEuler);
    EXPECT_EQ(metis::parse_integration_method("backward euler"),
              metis::IntegrationMethod::BackwardEuler);
    EXPECT_EQ(metis::parse_integration_method("backwards_euler"),
              metis::IntegrationMethod::BackwardEuler);
    EXPECT_EQ(metis::parse_integration_method("backwards euler"),
              metis::IntegrationMethod::BackwardEuler);
}

TEST(FiniteDiffTests, ParseIntegrationMethod_Invalid) {
    EXPECT_THROW(metis::parse_integration_method("unknown"), metis::InvalidArgument);
    EXPECT_THROW(metis::parse_integration_method("runge_kutta"), metis::InvalidArgument);
}

// =============================================================================
// Weight Function Tests
// =============================================================================

TEST(FiniteDiffTests, ForwardEulerWeights) {
    auto [w0, w1] = metis::forward_euler_weights(0.1);
    EXPECT_NEAR(w0, -10.0, 1e-10);
    EXPECT_NEAR(w1, 10.0, 1e-10);
}

TEST(FiniteDiffTests, BackwardEulerWeights) {
    auto [w0, w1] = metis::backward_euler_weights(0.5);
    EXPECT_NEAR(w0, -2.0, 1e-10);
    EXPECT_NEAR(w1, 2.0, 1e-10);
}

TEST(FiniteDiffTests, CentralDifferenceWeights) {
    auto [wm, wp] = metis::central_difference_weights(1.0);
    EXPECT_NEAR(wm, -0.5, 1e-10);
    EXPECT_NEAR(wp, 0.5, 1e-10);
}

TEST(FiniteDiffTests, TrapezoidalWeights) {
    auto [w0, w1] = metis::trapezoidal_weights(2.0);
    EXPECT_NEAR(w0, 1.0, 1e-10);
    EXPECT_NEAR(w1, 1.0, 1e-10);
}

// =============================================================================
// Difference Function Tests
// =============================================================================

TEST(FiniteDiffTests, ForwardDifference) {
    metis::NumericVector f(4);
    f << 0.0, 1.0, 4.0, 9.0; // f(x) = x^2 at x = 0, 1, 2, 3

    metis::NumericVector x(4);
    x << 0.0, 1.0, 2.0, 3.0;

    auto df = metis::forward_difference(f, x);

    EXPECT_EQ(df.size(), 3);
    EXPECT_NEAR(df(0), 1.0, 1e-10); // (1-0)/(1-0)
    EXPECT_NEAR(df(1), 3.0, 1e-10); // (4-1)/(2-1)
    EXPECT_NEAR(df(2), 5.0, 1e-10); // (9-4)/(3-2)
}

TEST(FiniteDiffTests, ForwardDifference_SizeMismatch) {
    metis::NumericVector f(3);
    f << 0.0, 1.0, 4.0;

    metis::NumericVector x(4);
    x << 0.0, 1.0, 2.0, 3.0;

    EXPECT_THROW(metis::forward_difference(f, x), metis::InvalidArgument);
}

TEST(FiniteDiffTests, ForwardDifference_TooFewPoints) {
    metis::NumericVector f(1);
    f << 0.0;

    metis::NumericVector x(1);
    x << 0.0;

    EXPECT_THROW(metis::forward_difference(f, x), metis::InvalidArgument);
}

TEST(FiniteDiffTests, BackwardDifference) {
    metis::NumericVector f(3);
    f << 1.0, 2.0, 5.0;

    metis::NumericVector x(3);
    x << 0.0, 1.0, 3.0;

    auto df = metis::backward_difference(f, x);

    EXPECT_EQ(df.size(), 2);
    EXPECT_NEAR(df(0), 1.0, 1e-10); // (2-1)/(1-0)
    EXPECT_NEAR(df(1), 1.5, 1e-10); // (5-2)/(3-1)
}

TEST(FiniteDiffTests, CentralDifference) {
    metis::NumericVector f(5);
    f << 0.0, 1.0, 4.0, 9.0, 16.0; // f(x) = x^2

    metis::NumericVector x(5);
    x << 0.0, 1.0, 2.0, 3.0, 4.0;

    auto df = metis::central_difference(f, x);

    EXPECT_EQ(df.size(), 3);
    EXPECT_NEAR(df(0), 2.0, 1e-10); // (4-0)/(2-0)
    EXPECT_NEAR(df(1), 4.0, 1e-10); // (9-1)/(3-1)
    EXPECT_NEAR(df(2), 6.0, 1e-10); // (16-4)/(4-2)
}

TEST(FiniteDiffTests, CentralDifference_SizeMismatch) {
    metis::NumericVector f(3);
    f << 0.0, 1.0, 4.0;

    metis::NumericVector x(5);
    x << 0.0, 1.0, 2.0, 3.0, 4.0;

    EXPECT_THROW(metis::central_difference(f, x), metis::InvalidArgument);
}

TEST(FiniteDiffTests, CentralDifference_TooFewPoints) {
    metis::NumericVector f(2);
    f << 0.0, 1.0;

    metis::NumericVector x(2);
    x << 0.0, 1.0;

    EXPECT_THROW(metis::central_difference(f, x), metis::InvalidArgument);
}

// =============================================================================
// Integration Defects Tests
// =============================================================================

TEST(FiniteDiffTests, IntegrationDefects_Trapezoidal) {
    // Test: x(t) = t^2, xdot(t) = 2t
    metis::NumericVector t(4);
    t << 0.0, 1.0, 2.0, 3.0;

    metis::NumericVector x(4);
    x << 0.0, 1.0, 4.0, 9.0;

    metis::NumericVector xdot(4);
    xdot << 0.0, 2.0, 4.0, 6.0;

    auto defects = metis::integration_defects(x, xdot, t, metis::IntegrationMethod::Trapezoidal);

    EXPECT_EQ(defects.size(), 3);
    // Trapezoidal should be exact for linear xdot integrated
    EXPECT_NEAR(defects(0), 0.0, 1e-10);
    EXPECT_NEAR(defects(1), 0.0, 1e-10);
    EXPECT_NEAR(defects(2), 0.0, 1e-10);
}

TEST(FiniteDiffTests, IntegrationDefects_ForwardEuler) {
    // x(t) = t, xdot(t) = 1
    metis::NumericVector t(3);
    t << 0.0, 1.0, 2.0;

    metis::NumericVector x(3);
    x << 0.0, 1.0, 2.0;

    metis::NumericVector xdot(3);
    xdot << 1.0, 1.0, 1.0;

    auto defects = metis::integration_defects(x, xdot, t, metis::IntegrationMethod::ForwardEuler);

    EXPECT_EQ(defects.size(), 2);
    EXPECT_NEAR(defects(0), 0.0, 1e-10);
    EXPECT_NEAR(defects(1), 0.0, 1e-10);
}

TEST(FiniteDiffTests, IntegrationDefects_BackwardEuler) {
    // x(t) = t, xdot(t) = 1
    metis::NumericVector t(3);
    t << 0.0, 1.0, 2.0;

    metis::NumericVector x(3);
    x << 0.0, 1.0, 2.0;

    metis::NumericVector xdot(3);
    xdot << 1.0, 1.0, 1.0;

    auto defects = metis::integration_defects(x, xdot, t, metis::IntegrationMethod::BackwardEuler);

    EXPECT_EQ(defects.size(), 2);
    EXPECT_NEAR(defects(0), 0.0, 1e-10);
    EXPECT_NEAR(defects(1), 0.0, 1e-10);
}

TEST(FiniteDiffTests, IntegrationDefects_SizeMismatch) {
    metis::NumericVector t(3);
    t << 0.0, 1.0, 2.0;

    metis::NumericVector x(4);
    x << 0.0, 1.0, 2.0, 3.0;

    metis::NumericVector xdot(3);
    xdot << 1.0, 1.0, 1.0;

    EXPECT_THROW(metis::integration_defects(x, xdot, t), metis::InvalidArgument);
}

TEST(FiniteDiffTests, IntegrationDefects_TooFewPoints) {
    metis::NumericVector t(1);
    t << 0.0;

    metis::NumericVector x(1);
    x << 0.0;

    metis::NumericVector xdot(1);
    xdot << 1.0;

    EXPECT_THROW(metis::integration_defects(x, xdot, t), metis::InvalidArgument);
}
