#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/MetisError.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/IntegrateDiscrete.hpp>

template <typename Scalar> void test_integrate_core() {
    // x = [0, 1, 2]
    // f = [0, 2, 4] (linear 2x)

    // Intervals: [0, 1], [1, 2]. dx = 1.

    using Vector = metis::MetisVector<Scalar>;
    Vector x(3);
    x << 0.0, 1.0, 2.0;
    Vector f(3);
    f << 0.0, 2.0, 4.0;

    // Trapezoidal: Exact for linear.
    // Int 1: (0+2)/2 * 1 = 1.
    // Int 2: (2+4)/2 * 1 = 3.
    auto trapz = metis::integrate_discrete_intervals(f, x, true, "trapezoidal");

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(trapz.size(), 2);
        EXPECT_NEAR(trapz(0), 1.0, 1e-9);
        EXPECT_NEAR(trapz(1), 3.0, 1e-9);
    } else {
        auto t = metis::eval(trapz);
        EXPECT_NEAR(t(0), 1.0, 1e-9);
        EXPECT_NEAR(t(1), 3.0, 1e-9);
    }

    // Euler
    auto fwd = metis::integrate_discrete_intervals(f, x, true, "forward_euler");
    // [0, 2] * 1 = [0, 2]
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(fwd(0), 0.0, 1e-9);
        EXPECT_NEAR(fwd(1), 2.0, 1e-9);
    } else {
        auto t = metis::eval(fwd);
        EXPECT_NEAR(t(0), 0.0, 1e-9);
        EXPECT_NEAR(t(1), 2.0, 1e-9);
    }

    auto bwd = metis::integrate_discrete_intervals(f, x, true, "backward_euler");
    // [2, 4] * 1 = [2, 4]
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(bwd(0), 2.0, 1e-9);
        EXPECT_NEAR(bwd(1), 4.0, 1e-9);
    } else {
        auto t = metis::eval(bwd);
        EXPECT_NEAR(t(0), 2.0, 1e-9);
        EXPECT_NEAR(t(1), 4.0, 1e-9);
    }

    // Simpson (Quadratic exactness)
    // Use quadratic data: f(x) = x^2 -> f = [0, 1, 4]
    Vector f_quad(3);
    f_quad << 0.0, 1.0, 4.0;

    // forward_simpson calculates [0, 1] exactly (1/3).
    // remaining_end = 1, so recurse on [1, 2] with Trapezoidal.
    // Trapz on [1, 2]: (1+4)/2 * 1 = 2.5.
    auto simpson_fwd =
        metis::integrate_discrete_intervals(f_quad, x, true, "forward_simpson", "lower_order");

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(simpson_fwd.size(), 2);
        EXPECT_NEAR(simpson_fwd(0), 1.0 / 3.0, 1e-9); // Simpson exact
        EXPECT_NEAR(simpson_fwd(1), 2.5, 1e-9);       // Trapz fallback
    } else {
        auto t = metis::eval(simpson_fwd);
        EXPECT_NEAR(t(0), 1.0 / 3.0, 1e-9);
        EXPECT_NEAR(t(1), 2.5, 1e-9);
    }

    // Cubic method test (needs 4+ points)
    // x = [0, 1, 2, 3], f(x) = x^3 -> f = [0, 1, 8, 27]
    // Cubic should be exact for cubic polynomials
    Vector x_cubic(4);
    x_cubic << 0.0, 1.0, 2.0, 3.0;
    Vector f_cubic(4);
    f_cubic << 0.0, 1.0, 8.0, 27.0; // x^3

    // Integral of x^3 = x^4/4
    // Exact: [0,1] = 0.25, [1,2] = 3.75, [2,3] = 16.25
    // But endpoints use trapezoidal fallback (lower_order):
    // Trapz [0,1]: (0+1)/2 = 0.5
    // Cubic [1,2]: exact = 3.75
    // Trapz [2,3]: (8+27)/2 = 17.5
    auto cubic_res =
        metis::integrate_discrete_intervals(f_cubic, x_cubic, true, "cubic", "lower_order");

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(cubic_res.size(), 3);
        EXPECT_NEAR(cubic_res(0), 0.5, 1e-6);  // First interval (trapz fallback)
        EXPECT_NEAR(cubic_res(1), 3.75, 1e-6); // Middle interval (cubic exact)
        EXPECT_NEAR(cubic_res(2), 17.5, 1e-6); // Last interval (trapz fallback)
    } else {
        auto t = metis::eval(cubic_res);
        EXPECT_NEAR(t(0), 0.5, 1e-6);
        EXPECT_NEAR(t(1), 3.75, 1e-6);
        EXPECT_NEAR(t(2), 17.5, 1e-6);
    }

    // Test squared curvature (for regularization)
    // f(x) = x^2 has f''(x) = 2, so ∫(f'')² dx = 4 * dx
    // For x = [0, 1, 2], intervals are 1 each, so total = 4 + 4 = 8
    auto curv_res = metis::integrate_discrete_squared_curvature(f_quad, x, "simpson");

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(curv_res.size(), 2);
        // Curvature should be approximately 4 per interval for f(x)=x^2
        EXPECT_GT(curv_res(0), 0.0); // Positive curvature
        EXPECT_GT(curv_res(1), 0.0);
    } else {
        auto t = metis::eval(curv_res);
        EXPECT_GT(t(0), 0.0);
        EXPECT_GT(t(1), 0.0);
    }
}

template <typename Scalar> void test_integrate_extensions() {
    using Vector = metis::MetisVector<Scalar>;
    Vector x(4);
    x << 0.0, 1.0, 2.0, 3.0;
    Vector f = x.array().square(); // [0, 1, 4, 9]

    // 1. Test "simpson" (fused)
    // Simpson rule exact for quadratic.
    // Int x^2 from 0 to 3 is x^3/3 = 9.
    auto simpson_res = metis::integrate_discrete_intervals(f, x, true, "simpson");

    // Simpson returns intervals. Sum them up to get total integral.
    // Intervals: [0, 1], [1, 2], [2, 3]
    // Values should be approx: 1/3, 7/3, 19/3
    // Sum = 27/3 = 9.

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(simpson_res.size(), 3);
        EXPECT_NEAR(simpson_res.sum(), 9.0, 1e-9);
    } else {
        auto t = metis::eval(simpson_res);
        EXPECT_NEAR(t.sum(), 9.0, 1e-9);
    }

    // 2. Test "backward_simpson"
    // Backward simpson on [0, 1, 2, 3]
    // Starts at index 1 -> [1, 2] using [0, 1, 2]
    // Then [2, 3] using [1, 2, 3]
    // First interval [0, 1] is missed (remaining_start=1), handled by fallback or ignored?
    // Let's test "method_endpoints" default ("lower_order") which fills it with trapezoidal.
    auto bwd_res =
        metis::integrate_discrete_intervals(f, x, true, "backward_simpson", "lower_order");

    if constexpr (std::is_same_v<Scalar, double>) {
        // [0, 1]: Trapezoidal = 0.5
        // [1, 2]: Backward simpson using 0,1,2 = exact for x^2
        // [2, 3]: Backward simpson using 1,2,3 = exact for x^2
        // Exact [1, 2] = 7/3 ~ 2.333
        // Exact [2, 3] = 19/3 ~ 6.333
        EXPECT_NEAR(bwd_res(0), 0.5, 1e-9);
        EXPECT_NEAR(bwd_res(1), 7.0 / 3.0, 1e-9);
        EXPECT_NEAR(bwd_res(2), 19.0 / 3.0, 1e-9);
    }

    // 3. Test "method_endpoints" = "ignore"
    // Should return smaller vector
    auto ignore_res = metis::integrate_discrete_intervals(f, x, true, "backward_simpson", "ignore");
    // Should miss first interval
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(ignore_res.size(), 2);
        EXPECT_NEAR(ignore_res(0), 7.0 / 3.0, 1e-9); // Corresponds to interval [1, 2]
    }

    // 4. Test "hybrid_simpson_cubic" for squared curvature
    // f = x^2, f'' = 2, f''^2 = 4. Integral = 4*L.
    auto curv_hybrid = metis::integrate_discrete_squared_curvature(f, x, "hybrid_simpson_cubic");
    if constexpr (std::is_same_v<Scalar, double>) {
        // For quadratic, cubic spline approx might be exact or close?
        // f(x)=x^2. Spline logic uses gradients.
        // It provides an approximation. Just check it runs and is positive.
        EXPECT_GT(curv_hybrid.sum(), 0.0);
    }
}

TEST(IntegrateDiscrete, ExtensionsNumeric) { test_integrate_extensions<double>(); }
TEST(IntegrateDiscrete, ExtensionsSymbolic) { test_integrate_extensions<metis::SymbolicScalar>(); }

TEST(IntegrateDiscrete, Errors) {
    metis::MetisVector<double> x(3);
    x << 0, 1, 2;
    metis::MetisVector<double> f = x;

    EXPECT_THROW(metis::integrate_discrete_intervals(f, x, true, "invalid_method"),
                 metis::IntegrationError);
    EXPECT_THROW(metis::integrate_discrete_squared_curvature(f, x, "invalid_method"),
                 metis::IntegrationError);

    // Check Simpon generic error
    EXPECT_THROW(metis::integrate_discrete_intervals(f, x, true, "simpson_invalid"),
                 metis::IntegrationError);
}

TEST(IntegrateDiscrete, CoreNumeric) { test_integrate_core<double>(); }
TEST(IntegrateDiscrete, CoreSymbolic) { test_integrate_core<metis::SymbolicScalar>(); }
