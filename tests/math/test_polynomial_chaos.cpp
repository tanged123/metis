/**
 * @file test_polynomial_chaos.cpp
 * @brief Tests for polynomial chaos basis construction and coefficient recovery.
 */

#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/OrthogonalPolynomials.hpp>
#include <metis/math/PolynomialChaos.hpp>

#include <cmath>

using namespace metis;

namespace {

NumericMatrix sample_matrix_from_row(const NumericVector &samples) {
    NumericMatrix out(samples.size(), 1);
    out.col(0) = samples;
    return out;
}

double legendre_p2(double x) { return 0.5 * (3.0 * x * x - 1.0); }

double expected_uniform_variance(double c1, double c2) { return c1 * c1 + c2 * c2; }

} // namespace

TEST(PolynomialChaosTests, UnivariateFamiliesMatchKnownLowOrderPolynomials) {
    const double x = 0.3;

    EXPECT_NEAR(pce_polynomial(hermite_dimension(), 0, x, false), 1.0, 1e-14);
    EXPECT_NEAR(pce_polynomial(hermite_dimension(), 2, x, false), x * x - 1.0, 1e-14);
    EXPECT_NEAR(pce_polynomial(hermite_dimension(), 3, x, false), x * x * x - 3.0 * x, 1e-14);

    EXPECT_NEAR(pce_polynomial(legendre_dimension(), 0, x, false), 1.0, 1e-14);
    EXPECT_NEAR(pce_polynomial(legendre_dimension(), 1, x, false), x, 1e-14);
    EXPECT_NEAR(pce_polynomial(legendre_dimension(), 2, x, false), legendre_p2(x), 1e-14);

    const double jacobi_x = 0.25;
    EXPECT_NEAR(pce_polynomial(jacobi_dimension(1.0, 2.0), 1, jacobi_x, false), 0.125, 1e-14);

    EXPECT_NEAR(pce_polynomial(laguerre_dimension(), 0, x, false), 1.0, 1e-14);
    EXPECT_NEAR(pce_polynomial(laguerre_dimension(), 1, x, false), 1.0 - x, 1e-14);
    EXPECT_NEAR(pce_polynomial(laguerre_dimension(), 2, x, false), 1.0 - 2.0 * x + 0.5 * x * x,
                1e-14);

    EXPECT_NEAR(pce_squared_norm(hermite_dimension(), 4, false), 24.0, 1e-12);
    EXPECT_NEAR(pce_squared_norm(legendre_dimension(), 2, false), 1.0 / 5.0, 1e-12);
    EXPECT_NEAR(pce_squared_norm(laguerre_dimension(), 3, false), 1.0, 1e-12);
}

TEST(PolynomialChaosTests, BasisConstructionAndOrderingMatchExpectedMultiIndices) {
    PolynomialChaosBasis total_order(
        {legendre_dimension(), hermite_dimension()}, 2,
        PolynomialChaosBasisOptions{.truncation = PolynomialChaosTruncation::TotalOrder,
                                    .normalized = false});
    ASSERT_EQ(total_order.size(), 6);

    std::vector<std::vector<int>> expected_total = {
        {0, 0}, {0, 1}, {1, 0}, {0, 2}, {1, 1}, {2, 0},
    };

    ASSERT_EQ(total_order.terms().size(), expected_total.size());
    for (std::size_t i = 0; i < expected_total.size(); ++i) {
        EXPECT_EQ(total_order.terms()[i].multi_index, expected_total[i]);
    }

    NumericVector point(2);
    point << 0.25, -0.4;
    NumericVector psi = total_order.evaluate(point);
    ASSERT_EQ(psi.size(), 6);
    EXPECT_NEAR(psi(0), 1.0, 1e-14);
    EXPECT_NEAR(psi(1), -0.4, 1e-14);
    EXPECT_NEAR(psi(2), 0.25, 1e-14);
    EXPECT_NEAR(psi(4), -0.1, 1e-14);

    PolynomialChaosBasis tensor_product(
        {legendre_dimension(), hermite_dimension()}, 2,
        PolynomialChaosBasisOptions{.truncation = PolynomialChaosTruncation::TensorProduct,
                                    .normalized = true});
    ASSERT_EQ(tensor_product.size(), 9);
    EXPECT_EQ(tensor_product.terms().front().multi_index, (std::vector<int>{0, 0}));
    EXPECT_EQ(tensor_product.terms().back().multi_index, (std::vector<int>{2, 2}));
}

TEST(PolynomialChaosTests, ProjectionRecoversNormalizedLegendreExpansion) {
    PolynomialChaosBasis basis({legendre_dimension()}, 2);

    const NumericVector nodes = lgl_nodes(5);
    const NumericVector weights = 0.5 * lgl_weights(5, nodes);
    const NumericMatrix samples = sample_matrix_from_row(nodes);

    NumericVector values(nodes.size());
    for (Eigen::Index i = 0; i < nodes.size(); ++i) {
        const double x = nodes(i);
        values(i) = 1.2 * pce_polynomial(legendre_dimension(), 0, x) +
                    0.5 * pce_polynomial(legendre_dimension(), 1, x) -
                    0.7 * pce_polynomial(legendre_dimension(), 2, x);
    }

    const NumericVector coeffs = pce_projection_coefficients(basis, samples, weights, values);
    ASSERT_EQ(coeffs.size(), 3);
    EXPECT_NEAR(coeffs(0), 1.2, 1e-12);
    EXPECT_NEAR(coeffs(1), 0.5, 1e-12);
    EXPECT_NEAR(coeffs(2), -0.7, 1e-12);
    EXPECT_NEAR(pce_mean(coeffs), 1.2, 1e-12);
    EXPECT_NEAR(pce_variance(basis, coeffs), expected_uniform_variance(0.5, -0.7), 1e-12);
}

TEST(PolynomialChaosTests, RegressionSupportsSymbolicSampleValuesAndMomentGradients) {
    PolynomialChaosBasis basis({legendre_dimension()}, 2);

    NumericVector nodes(6);
    nodes << -1.0, -0.6, -0.2, 0.2, 0.6, 1.0;
    const NumericMatrix samples = sample_matrix_from_row(nodes);

    SymbolicScalar a = sym("a");
    SymbolicVector sample_values(nodes.size());
    for (Eigen::Index i = 0; i < nodes.size(); ++i) {
        const double x = nodes(i);
        sample_values(i) = (1.0 + a) * pce_polynomial(legendre_dimension(), 0, x) -
                           0.4 * pce_polynomial(legendre_dimension(), 1, x) +
                           (0.25 * a) * pce_polynomial(legendre_dimension(), 2, x);
    }

    SymbolicVector coeffs = pce_regression_coefficients(basis, samples, sample_values, 0.0);
    SymbolicScalar mean = pce_mean(coeffs);
    SymbolicScalar variance = pce_variance(basis, coeffs);
    Function stats("pce_regression_stats", {a},
                   {to_mx(coeffs), mean, variance, jacobian(mean, a), jacobian(variance, a)});

    const auto outputs = stats(2.0);
    ASSERT_EQ(outputs.size(), 5u);

    const NumericMatrix coeff_matrix = outputs[0];
    ASSERT_EQ(coeff_matrix.rows(), 3);
    EXPECT_NEAR(coeff_matrix(0, 0), 3.0, 1e-10);
    EXPECT_NEAR(coeff_matrix(1, 0), -0.4, 1e-10);
    EXPECT_NEAR(coeff_matrix(2, 0), 0.5, 1e-10);

    EXPECT_NEAR(outputs[1](0, 0), 3.0, 1e-10);
    EXPECT_NEAR(outputs[2](0, 0), 0.41, 1e-10);
    EXPECT_NEAR(outputs[3](0, 0), 1.0, 1e-10);
    EXPECT_NEAR(outputs[4](0, 0), 0.25, 1e-10);
}

TEST(PolynomialChaosTests, RegressionRejectsRankDeficientDesignWithoutRegularization) {
    PolynomialChaosBasis basis({legendre_dimension()}, 2);

    NumericMatrix samples(3, 1);
    samples << 0.0, 0.0, 0.0;

    NumericVector values(3);
    values << 1.0, 1.0, 1.0;

    EXPECT_THROW(pce_regression_coefficients(basis, samples, values, 0.0), InvalidArgument);
}
