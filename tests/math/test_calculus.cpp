#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/AutoDiff.hpp>
#include <metis/math/Calculus.hpp>
#include <metis/math/Linalg.hpp>

template <typename Scalar> void test_gradient_uniform() {
    using Vector = metis::MetisVector<Scalar>;

    // Test case 1: Linear function y = 2x
    // dy/dx should be 2 everywhere
    Vector x(5);
    x << 0.0, 1.0, 2.0, 3.0, 4.0;
    Vector y = 2.0 * x.array();

    auto grad = metis::gradient(y, 1.0, 1, 1);

    if constexpr (std::is_same_v<Scalar, double>) {
        // All points should have gradient = 2
        for (int i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad(i), 2.0, 1e-10);
        }
    } else {
        auto grad_eval = metis::eval(grad);
        for (int i = 0; i < grad_eval.size(); ++i) {
            EXPECT_NEAR(grad_eval(i), 2.0, 1e-9);
        }
    }
}

template <typename Scalar> void test_gradient_quadratic() {
    using Vector = metis::MetisVector<Scalar>;

    // Test case 2: Quadratic y = x^2
    // dy/dx = 2x
    Vector x(11);
    for (int i = 0; i < 11; ++i) {
        x(i) = static_cast<Scalar>(i - 5);
    }
    Vector y = x.array().square();

    // Test with edge_order = 2 for better boundary accuracy
    auto grad = metis::gradient(y, 1.0, 2, 1);
    Vector expected = 2.0 * x.array();

    if constexpr (std::is_same_v<Scalar, double>) {
        for (int i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad(i), expected(i), 1e-10);
        }
    } else {
        auto grad_eval = metis::eval(grad);
        auto expected_eval = metis::eval(expected);
        for (int i = 0; i < grad_eval.size(); ++i) {
            EXPECT_NEAR(grad_eval(i), expected_eval(i), 1e-9);
        }
    }
}

template <typename Scalar> void test_gradient_second_derivative() {
    using Vector = metis::MetisVector<Scalar>;

    // Test case 3: Quadratic y = x^2
    // d^2y/dx^2 = 2 everywhere
    Vector x(11);
    for (int i = 0; i < 11; ++i) {
        x(i) = static_cast<Scalar>(i);
    }
    Vector y = x.array().square();

    auto grad2 = metis::gradient(y, 1.0, 1, 2);

    if constexpr (std::is_same_v<Scalar, double>) {
        for (int i = 0; i < grad2.size(); ++i) {
            EXPECT_NEAR(grad2(i), 2.0, 1e-10);
        }
    } else {
        auto grad2_eval = metis::eval(grad2);
        for (int i = 0; i < grad2_eval.size(); ++i) {
            EXPECT_NEAR(grad2_eval(i), 2.0, 1e-9);
        }
    }
}

template <typename Scalar> void test_gradient_nonuniform() {
    using Vector = metis::MetisVector<Scalar>;

    // Test case 4: Non-uniform grid
    Vector x(5);
    x << 0.0, 1.0, 3.0, 6.0, 10.0;
    Vector y = x.array().square();

    // gradient should handle non-uniform spacing via x vector
    auto grad = metis::gradient(y, x, 2, 1);
    Vector expected = 2.0 * x.array();

    if constexpr (std::is_same_v<Scalar, double>) {
        for (int i = 0; i < grad.size(); ++i) {
            EXPECT_NEAR(grad(i), expected(i), 1e-8);
        }
    } else {
        auto grad_eval = metis::eval(grad);
        auto expected_eval = metis::eval(expected);
        for (int i = 0; i < grad_eval.size(); ++i) {
            EXPECT_NEAR(grad_eval(i), expected_eval(i), 1e-8);
        }
    }
}

template <typename Scalar> void test_gradient_cubic() {
    using Vector = metis::MetisVector<Scalar>;

    // Test case 5: Cubic y = x^3
    // dy/dx = 3x^2
    Vector x(9);
    for (int i = 0; i < 9; ++i) {
        x(i) = static_cast<Scalar>(i - 4);
    }
    Vector y = x.array().cube();

    auto grad = metis::gradient(y, 1.0, 2, 1);
    Vector expected = 3.0 * x.array().square();

    if constexpr (std::is_same_v<Scalar, double>) {
        for (int i = 1; i < grad.size() - 1; ++i) {
            // Interior points: second-order formula has O(h^2) error for cubics
            // With h=1, we expect errors ~1
            EXPECT_NEAR(grad(i), expected(i), 2.0);
        }
        // Boundaries have larger error
        EXPECT_NEAR(grad(0), expected(0), 5.0);
        EXPECT_NEAR(grad(8), expected(8), 5.0);
    } else {
        auto grad_eval = metis::eval(grad);
        auto expected_eval = metis::eval(expected);
        for (int i = 1; i < grad_eval.size() - 1; ++i) {
            EXPECT_NEAR(grad_eval(i), expected_eval(i), 2.0);
        }
    }
}

template <typename Scalar> void test_gradient_edge_cases() {
    using Vector = metis::MetisVector<Scalar>;

    // Test with 2 points
    Vector x2(2);
    x2 << 0.0, 1.0;
    Vector y2 = x2.array() * 3.0;

    auto grad2 = metis::gradient(y2, 1.0, 1, 1);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(grad2(0), 3.0, 1e-10);
        EXPECT_NEAR(grad2(1), 3.0, 1e-10);
    } else {
        auto grad2_eval = metis::eval(grad2);
        EXPECT_NEAR(grad2_eval(0), 3.0, 1e-9);
        EXPECT_NEAR(grad2_eval(1), 3.0, 1e-9);
    }
}

TEST(CalculusTests, GradientUniformNumeric) { test_gradient_uniform<double>(); }

TEST(CalculusTests, GradientUniformSymbolic) { test_gradient_uniform<metis::SymbolicScalar>(); }

TEST(CalculusTests, GradientQuadraticNumeric) { test_gradient_quadratic<double>(); }

TEST(CalculusTests, GradientQuadraticSymbolic) { test_gradient_quadratic<metis::SymbolicScalar>(); }

TEST(CalculusTests, GradientSecondDerivativeNumeric) { test_gradient_second_derivative<double>(); }

TEST(CalculusTests, GradientSecondDerivativeSymbolic) {
    test_gradient_second_derivative<metis::SymbolicScalar>();
}

TEST(CalculusTests, GradientNonuniformNumeric) { test_gradient_nonuniform<double>(); }

TEST(CalculusTests, GradientNonuniformSymbolic) {
    test_gradient_nonuniform<metis::SymbolicScalar>();
}

TEST(CalculusTests, GradientCubicNumeric) { test_gradient_cubic<double>(); }

TEST(CalculusTests, GradientCubicSymbolic) { test_gradient_cubic<metis::SymbolicScalar>(); }

TEST(CalculusTests, GradientEdgeCasesNumeric) { test_gradient_edge_cases<double>(); }

TEST(CalculusTests, GradientEdgeCasesSymbolic) {
    test_gradient_edge_cases<metis::SymbolicScalar>();
}

// --- Tests for diff, trapz, gradient_1d ---

template <typename Scalar> void test_diff_trapz_gradient1d() {
    using Vector = metis::MetisVector<Scalar>;

    // Test diff
    Vector v(4);
    v << 0.0, 1.0, 4.0, 9.0;
    auto res_diff = metis::diff(v); // [1, 3, 5]

    // Test trapz
    Vector y(2);
    y << 1.0, 1.0;
    Vector x(2);
    x << 0.0, 1.0;
    auto res_trapz = metis::trapz(y, x);

    // Test gradient_1d
    Vector x_grad(5);
    x_grad << 0.0, 1.0, 2.0, 3.0, 4.0;
    Vector y_grad(5);
    y_grad << 0.0, 1.0, 4.0, 9.0, 16.0;
    auto res_grad = metis::gradient_1d(y_grad, x_grad);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(res_diff.size(), 3);
        EXPECT_DOUBLE_EQ(res_diff(0), 1.0);
        EXPECT_DOUBLE_EQ(res_diff(1), 3.0);

        EXPECT_DOUBLE_EQ(res_trapz, 1.0);

        EXPECT_DOUBLE_EQ(res_grad(1), 2.0); // Exact for quadratic
        EXPECT_DOUBLE_EQ(res_grad(2), 4.0);
    } else {
        EXPECT_EQ(res_diff.size(), 3);
        auto res_diff_eval = metis::eval(res_diff);
        EXPECT_DOUBLE_EQ(res_diff_eval(0), 1.0);
        EXPECT_DOUBLE_EQ(res_diff_eval(1), 3.0);

        EXPECT_DOUBLE_EQ(metis::eval(res_trapz), 1.0);

        auto res_grad_eval = metis::eval(res_grad);
        EXPECT_DOUBLE_EQ(res_grad_eval(1), 2.0);
        EXPECT_DOUBLE_EQ(res_grad_eval(2), 4.0);
    }
}

template <typename Scalar> void test_cumtrapz_nonuniform() {
    using Vector = metis::MetisVector<Scalar>;

    Vector x(3);
    x << 0.0, 1.0, 3.0;
    Vector y(3);
    y << 0.0, 1.0, 9.0;

    auto res = metis::cumtrapz(y, x);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(res.size(), 3);
        EXPECT_DOUBLE_EQ(res(0), 0.0);
        EXPECT_DOUBLE_EQ(res(1), 0.5);
        EXPECT_DOUBLE_EQ(res(2), 10.5);
    } else {
        auto res_eval = metis::eval(res);
        EXPECT_EQ(res_eval.size(), 3);
        EXPECT_DOUBLE_EQ(res_eval(0), 0.0);
        EXPECT_DOUBLE_EQ(res_eval(1), 0.5);
        EXPECT_DOUBLE_EQ(res_eval(2), 10.5);
    }
}

TEST(CalculusTests, CumtrapzNonuniformNumeric) { test_cumtrapz_nonuniform<double>(); }

TEST(CalculusTests, CumtrapzNonuniformSymbolic) {
    test_cumtrapz_nonuniform<metis::SymbolicScalar>();
}

TEST(CalculusTests, CumtrapzUniformSpacingNumeric) {
    metis::MetisVector<double> y(3);
    y << 0.0, 2.0, 4.0;

    auto res = metis::cumtrapz(y, 1.0);

    EXPECT_EQ(res.size(), 3);
    EXPECT_DOUBLE_EQ(res(0), 0.0);
    EXPECT_DOUBLE_EQ(res(1), 1.0);
    EXPECT_DOUBLE_EQ(res(2), 4.0);
}

TEST(CalculusTests, CumtrapzSymbolicGraph) {
    metis::NumericVector x(3);
    x << 0.0, 1.0, 2.0;

    auto [y, y_mx] = metis::sym_vec_pair("y", 3);
    auto cum = metis::cumtrapz(y, x);

    metis::Function f({y_mx}, {metis::to_mx(cum)});
    metis::NumericVector y_val(3);
    y_val << 0.0, 2.0, 4.0;
    auto cum_val = f.eval(y_val);

    EXPECT_DOUBLE_EQ(cum_val(0), 0.0);
    EXPECT_DOUBLE_EQ(cum_val(1), 1.0);
    EXPECT_DOUBLE_EQ(cum_val(2), 4.0);

    auto J = metis::jacobian({metis::to_mx(cum)}, {y_mx});
    metis::Function jac_fn({y_mx}, {J});
    auto jac_val = jac_fn.eval(y_val);

    EXPECT_DOUBLE_EQ(jac_val(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(jac_val(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(jac_val(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(jac_val(1, 0), 0.5);
    EXPECT_DOUBLE_EQ(jac_val(1, 1), 0.5);
    EXPECT_DOUBLE_EQ(jac_val(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(jac_val(2, 0), 0.5);
    EXPECT_DOUBLE_EQ(jac_val(2, 1), 1.0);
    EXPECT_DOUBLE_EQ(jac_val(2, 2), 0.5);
}

TEST(CalculusTests, DiffTrapzGradient1dNumeric) { test_diff_trapz_gradient1d<double>(); }

TEST(CalculusTests, DiffTrapzGradient1dSymbolic) {
    test_diff_trapz_gradient1d<metis::SymbolicScalar>();
}

// --- Periodic and Error Tests ---

template <typename Scalar> void test_gradient_periodic_wraparound() {
    using Vector = metis::MetisVector<Scalar>;
    Vector y(4);
    y << 0.0, 1.0, 0.0, -1.0; // sin(theta) sampled on [0, 2pi) at 90 deg increments

    auto grad = metis::gradient_periodic(y, M_PI / 2.0, 2.0 * M_PI);

    const double expected = 2.0 / M_PI;

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(grad.size(), 4);
        EXPECT_NEAR(grad(0), expected, 1e-10);
        EXPECT_NEAR(grad(1), 0.0, 1e-10);
        EXPECT_NEAR(grad(2), -expected, 1e-10);
        EXPECT_NEAR(grad(3), 0.0, 1e-10);
    } else {
        auto g = metis::eval(grad);
        EXPECT_EQ(g.size(), 4);
        EXPECT_NEAR(g(0), expected, 1e-10);
        EXPECT_NEAR(g(1), 0.0, 1e-10);
        EXPECT_NEAR(g(2), -expected, 1e-10);
        EXPECT_NEAR(g(3), 0.0, 1e-10);
    }
}

TEST(CalculusTests, GradientPeriodic) {
    test_gradient_periodic_wraparound<double>();
    test_gradient_periodic_wraparound<metis::SymbolicScalar>();
}

TEST(CalculusTests, GradientPeriodicRejectsDuplicateEndpointSamples) {
    metis::MetisVector<double> y(5);
    y << 0.0, 1.0, 0.0, -1.0, 0.0;

    EXPECT_THROW(metis::gradient_periodic(y, M_PI / 2.0, 2.0 * M_PI), metis::InvalidArgument);
}

TEST(CalculusTests, Errors) {
    metis::MetisVector<double> x(5);
    x.setZero();
    metis::MetisVector<double> y = x;

    // Invalid dx size (must be scalar, N, or N-1)
    metis::MetisVector<double> bad_dx(2);
    EXPECT_THROW(metis::gradient(y, bad_dx), metis::InvalidArgument);

    // Invalid edge_order
    EXPECT_THROW(metis::gradient(y, 1.0, 3), metis::InvalidArgument);

    // Invalid n (derivative order)
    EXPECT_THROW(metis::gradient(y, 1.0, 1, 3), metis::InvalidArgument);

    // cumtrapz input size mismatch
    metis::MetisVector<double> bad_x(4);
    bad_x.setZero();
    EXPECT_THROW(metis::cumtrapz(y, bad_x), metis::InvalidArgument);
}
