#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/MetisIO.hpp> // for metis::eval
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Spacing.hpp>
#include <numbers>

template <typename Scalar> void test_spacing_funcs() {
    using VectorType = metis::MetisVector<Scalar>;

    Scalar start = 0.0;
    Scalar end = 10.0;
    int n = 5;

    // --- linspace ---
    VectorType res_lin = metis::linspace(start, end, n);

    // --- cosine_spacing ---
    VectorType res_cos = metis::cosine_spacing(start, end, n);

    // --- sinspace ---
    Scalar sin_start = 0.0;
    Scalar sin_end = 1.0;
    VectorType res_sin = metis::sinspace(sin_start, sin_end, n);
    VectorType res_sin_rev = metis::sinspace(sin_start, sin_end, n, true);

    // --- logspace ---
    Scalar log_start = 0.0; // 10^0 = 1
    Scalar log_end = 2.0;   // 10^2 = 100
    VectorType res_log = metis::logspace(log_start, log_end, 3);

    // --- geomspace ---
    Scalar geom_start = 1.0;
    Scalar geom_end = 100.0;
    VectorType res_geom = metis::geomspace(geom_start, geom_end, 3);

    if constexpr (std::is_same_v<Scalar, double>) {
        // linspace
        EXPECT_EQ(res_lin.size(), 5);
        EXPECT_NEAR(res_lin(2), 5.0, 1e-9);
        EXPECT_NEAR(res_lin(4), 10.0, 1e-9);

        // cosine_spacing
        EXPECT_NEAR(res_cos(0), 0.0, 1e-9);
        EXPECT_NEAR(res_cos(2), 5.0, 1e-9);

        // sinspace
        EXPECT_NEAR(res_sin(0), 0.0, 1e-9);
        EXPECT_NEAR(res_sin(n - 1), 1.0, 1e-9);
        EXPECT_LT(res_sin(1) - res_sin(0), res_sin(n - 1) - res_sin(n - 2));

        EXPECT_GT(res_sin_rev(1) - res_sin_rev(0), res_sin_rev(n - 1) - res_sin_rev(n - 2));

        // logspace
        EXPECT_NEAR(res_log(0), 1.0, 1e-9);
        EXPECT_NEAR(res_log(1), 10.0, 1e-9);
        EXPECT_NEAR(res_log(2), 100.0, 1e-9);

        // geomspace
        EXPECT_NEAR(res_geom(0), 1.0, 1e-9);
        EXPECT_NEAR(res_geom(1), 10.0, 1e-9);
        EXPECT_NEAR(res_geom(2), 100.0, 1e-9);

    } else {
        // Validation logic for Symbolic
        auto num_lin = metis::eval(res_lin);
        EXPECT_EQ(num_lin.size(), 5);
        EXPECT_NEAR(num_lin(2), 5.0, 1e-9);
        EXPECT_NEAR(num_lin(4), 10.0, 1e-9);

        auto num_cos = metis::eval(res_cos);
        EXPECT_NEAR(num_cos(0), 0.0, 1e-9);
        EXPECT_NEAR(num_cos(2), 5.0, 1e-9);

        metis::NumericVector num_sin = metis::eval(res_sin);
        EXPECT_NEAR(num_sin(0), 0.0, 1e-9);
        EXPECT_NEAR(num_sin(n - 1), 1.0, 1e-9);
        EXPECT_LT(num_sin(1) - num_sin(0), num_sin(n - 1) - num_sin(n - 2));

        metis::NumericVector num_sin_rev = metis::eval(res_sin_rev);
        EXPECT_GT(num_sin_rev(1) - num_sin_rev(0), num_sin_rev(n - 1) - num_sin_rev(n - 2));

        metis::NumericVector num_log = metis::eval(res_log);
        EXPECT_NEAR(num_log(0), 1.0, 1e-9);
        EXPECT_NEAR(num_log(1), 10.0, 1e-9);
        EXPECT_NEAR(num_log(2), 100.0, 1e-9);

        metis::NumericVector num_geom = metis::eval(res_geom);
        EXPECT_NEAR(num_geom(0), 1.0, 1e-9);
        EXPECT_NEAR(num_geom(1), 10.0, 1e-9);
        EXPECT_NEAR(num_geom(2), 100.0, 1e-9);
    }
}

TEST(SpacingTests, Numeric) { test_spacing_funcs<double>(); }

TEST(SpacingTests, Symbolic) { test_spacing_funcs<metis::SymbolicScalar>(); }

TEST(SpacingTests, CoverageDegenerate) {
    // n < 2 cases
    auto lin = metis::linspace(0.0, 10.0, 1);
    EXPECT_EQ(lin.size(), 1);
    EXPECT_NEAR(lin(0), 0.0, 1e-9);

    auto cos = metis::cosine_spacing(0.0, 10.0, 1);
    EXPECT_EQ(cos.size(), 1);
    EXPECT_NEAR(cos(0), 0.0, 1e-9);

    auto sin = metis::sinspace(0.0, 10.0, 1);
    EXPECT_EQ(sin.size(), 1);
    EXPECT_NEAR(sin(0), 0.0, 1e-9);

    auto log = metis::logspace(0.0, 2.0, 1);
    EXPECT_EQ(log.size(), 1);
    EXPECT_NEAR(log(0), 1.0, 1e-9);

    // Reverse spacing for sinspace
    auto rev = metis::sinspace(0.0, 10.0, 5, true);
    // Should be bunched at end (large steps first, small steps last)
    // Differences:
    double d0 = rev(1) - rev(0);
    double d_last = rev(4) - rev(3);
    EXPECT_GT(d0, d_last);
}
