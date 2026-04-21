/**
 * @file test_quadrature.cpp
 * @brief Tests for stochastic collocation quadrature rules and sparse grids.
 */

#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/PolynomialChaos.hpp>
#include <metis/math/Quadrature.hpp>

#include <cmath>
#include <vector>

using namespace metis;

namespace {

double standard_normal_moment(int degree) {
    if (degree % 2 == 1) {
        return 0.0;
    }
    const int half = degree / 2;
    return std::exp(std::lgamma(static_cast<double>(degree) + 1.0) -
                    static_cast<double>(half) * std::log(2.0) -
                    std::lgamma(static_cast<double>(half) + 1.0));
}

double shifted_beta_moment(int degree, double alpha, double beta) {
    const double a = beta + 1.0;
    const double b = alpha + 1.0;

    double moment = 0.0;
    for (int j = 0; j <= degree; ++j) {
        const double u_moment =
            std::exp(std::lgamma(a + static_cast<double>(j)) + std::lgamma(a + b) - std::lgamma(a) -
                     std::lgamma(a + b + static_cast<double>(j)));
        const double coeff = std::pow(2.0, static_cast<double>(j)) *
                             (((degree - j) % 2) == 0 ? 1.0 : -1.0) *
                             std::tgamma(static_cast<double>(degree + 1)) /
                             (std::tgamma(static_cast<double>(j + 1)) *
                              std::tgamma(static_cast<double>(degree - j + 1)));
        moment += coeff * u_moment;
    }

    return moment;
}

template <typename MomentFunc>
void expect_rule_matches_moments(const UnivariateQuadratureRule &rule, int max_degree,
                                 MomentFunc &&moment_func, double tol) {
    ASSERT_EQ(rule.nodes.size(), rule.weights.size());
    EXPECT_NEAR(rule.weights.sum(), 1.0, tol);

    for (int degree = 0; degree <= max_degree; ++degree) {
        double approx = 0.0;
        for (Eigen::Index i = 0; i < rule.nodes.size(); ++i) {
            approx += rule.weights(i) * std::pow(rule.nodes(i), degree);
        }
        EXPECT_NEAR(approx, moment_func(degree), tol) << "degree=" << degree;
    }
}

double max_abs_diff(const NumericVector &a, const NumericVector &b) {
    return (a - b).cwiseAbs().maxCoeff();
}

} // namespace

TEST(QuadratureTests, GaussRulesMatchProbabilityMoments) {
    const auto legendre = stochastic_quadrature_rule(legendre_dimension(), 3);
    expect_rule_matches_moments(
        legendre, 5,
        [](int degree) {
            if (degree % 2 == 1) {
                return 0.0;
            }
            return 1.0 / static_cast<double>(degree + 1);
        },
        1e-12);

    const auto hermite = stochastic_quadrature_rule(hermite_dimension(), 4);
    expect_rule_matches_moments(hermite, 7, standard_normal_moment, 1e-11);

    const auto jacobi = stochastic_quadrature_rule(jacobi_dimension(1.0, 2.0), 3);
    expect_rule_matches_moments(
        jacobi, 5, [](int degree) { return shifted_beta_moment(degree, 1.0, 2.0); }, 1e-11);

    const auto laguerre = stochastic_quadrature_rule(laguerre_dimension(), 3);
    expect_rule_matches_moments(
        laguerre, 5, [](int degree) { return std::tgamma(static_cast<double>(degree + 1)); },
        1e-11);
}

TEST(QuadratureTests, NestedRulesRefineWithoutDroppingExistingNodes) {
    const auto cc_coarse = stochastic_quadrature_level(legendre_dimension(), 2,
                                                       StochasticQuadratureRule::ClenshawCurtis);
    const auto cc_fine = stochastic_quadrature_level(legendre_dimension(), 4,
                                                     StochasticQuadratureRule::ClenshawCurtis);

    EXPECT_TRUE(cc_coarse.nested);
    EXPECT_TRUE(cc_fine.nested);
    EXPECT_NEAR(cc_fine.weights.sum(), 1.0, 1e-12);

    for (Eigen::Index i = 0; i < cc_coarse.nodes.size(); ++i) {
        bool found = false;
        for (Eigen::Index j = 0; j < cc_fine.nodes.size(); ++j) {
            if (std::abs(cc_coarse.nodes(i) - cc_fine.nodes(j)) < 1e-14) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "missing coarse node " << cc_coarse.nodes(i);
    }

    const auto kronrod7 = stochastic_quadrature_level(legendre_dimension(), 1,
                                                      StochasticQuadratureRule::GaussKronrod15);
    const auto kronrod15 = stochastic_quadrature_level(legendre_dimension(), 2,
                                                       StochasticQuadratureRule::GaussKronrod15);
    EXPECT_EQ(kronrod7.nodes.size(), 7);
    EXPECT_EQ(kronrod15.nodes.size(), 15);
    EXPECT_NEAR(kronrod15.weights.sum(), 1.0, 1e-14);

    for (Eigen::Index i = 0; i < kronrod7.nodes.size(); ++i) {
        bool found = false;
        for (Eigen::Index j = 0; j < kronrod15.nodes.size(); ++j) {
            if (std::abs(kronrod7.nodes(i) - kronrod15.nodes(j)) < 1e-14) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "missing embedded node " << kronrod7.nodes(i);
    }
}

TEST(QuadratureTests, TensorProductGridRecoversPolynomialChaosCoefficients) {
    PolynomialChaosBasis basis({legendre_dimension(), hermite_dimension()}, 2);

    const auto x_rule = stochastic_quadrature_rule(legendre_dimension(), 3);
    const auto y_rule = stochastic_quadrature_rule(hermite_dimension(), 3);
    const auto grid = tensor_product_quadrature({x_rule, y_rule});

    ASSERT_EQ(grid.samples.cols(), 2);
    EXPECT_NEAR(grid.weights.sum(), 1.0, 1e-12);

    NumericVector expected = NumericVector::Zero(basis.size());
    expected << 1.5, -0.7, 0.25, 0.0, 0.4, 0.2;

    NumericVector values(grid.samples.rows());
    for (Eigen::Index i = 0; i < grid.samples.rows(); ++i) {
        const NumericVector point = grid.samples.row(i).transpose();
        values(i) = basis.evaluate(point).dot(expected);
    }

    const NumericVector recovered = pce_projection_coefficients(basis, grid, values);
    EXPECT_LT(max_abs_diff(recovered, expected), 1e-12);
}

TEST(QuadratureTests, SmolyakSparseGridMatchesLowOrderMomentsOnMixedFamilies) {
    const auto grid = smolyak_sparse_grid({legendre_dimension(), hermite_dimension()}, 3);

    EXPECT_EQ(grid.samples.cols(), 2);
    EXPECT_LT(grid.samples.rows(), 15);
    EXPECT_NEAR(grid.weights.sum(), 1.0, 1e-11);

    double ex = 0.0;
    double ey = 0.0;
    double ex2 = 0.0;
    double ey2 = 0.0;
    double exy = 0.0;
    for (Eigen::Index i = 0; i < grid.samples.rows(); ++i) {
        const double x = grid.samples(i, 0);
        const double y = grid.samples(i, 1);
        const double w = grid.weights(i);
        ex += w * x;
        ey += w * y;
        ex2 += w * x * x;
        ey2 += w * y * y;
        exy += w * x * y;
    }

    EXPECT_NEAR(ex, 0.0, 1e-11);
    EXPECT_NEAR(ey, 0.0, 1e-11);
    EXPECT_NEAR(ex2, 1.0 / 3.0, 1e-10);
    EXPECT_NEAR(ey2, 1.0, 1e-10);
    EXPECT_NEAR(exy, 0.0, 1e-11);
}
