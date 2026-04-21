#pragma once
/**
 * @file Quadrature.hpp
 * @brief Stochastic quadrature rules (Gauss, Clenshaw-Curtis, Smolyak sparse grids)
 * @see PolynomialChaos.hpp, OrthogonalPolynomials.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/OrthogonalPolynomials.hpp"
#include "metis/math/PolynomialChaos.hpp"
#include <Eigen/Eigenvalues>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace metis {

/**
 * @brief One-dimensional stochastic quadrature rule on a probability measure.
 */
struct UnivariateQuadratureRule {
    NumericVector nodes;
    NumericVector weights;
    bool nested = false;
    int level = 0;
};

/**
 * @brief Multidimensional stochastic quadrature grid with row-major sample layout.
 *
 * The sample matrix uses the same convention as `PolynomialChaosBasis`:
 * - rows: quadrature points
 * - cols: stochastic dimensions
 */
struct StochasticQuadratureGrid {
    NumericMatrix samples;
    NumericVector weights;
    bool nested = false;
    int level = 0;
};

/**
 * @brief One-dimensional rule family used to generate stochastic quadrature nodes.
 */
enum class StochasticQuadratureRule {
    Gauss,          ///< Family-appropriate Gauss rule
    ClenshawCurtis, ///< Nested Clenshaw-Curtis-like rule on bounded support
    GaussKronrod15, ///< Embedded 7/15-point Legendre rule reused from `quad(...)`
    AutoNested,     ///< Use Clenshaw-Curtis on bounded families, Gauss otherwise
};

/**
 * @brief Options for Smolyak sparse-grid construction.
 */
struct SmolyakQuadratureOptions {
    StochasticQuadratureRule rule = StochasticQuadratureRule::AutoNested;
    double merge_tolerance = 1e-12;
    double zero_weight_tolerance = 1e-14;
};

namespace detail {

struct EmbeddedQuadratureRule {
    std::array<double, 15> nodes;
    std::array<double, 15> primary_weights;
    std::array<double, 15> embedded_weights;
};

inline const EmbeddedQuadratureRule &gauss_kronrod_15_rule() {
    static const EmbeddedQuadratureRule rule = {
        std::array<double, 15>{-0.991455371120812639, -0.949107912342758525, -0.864864423359769073,
                               -0.741531185599394440, -0.586087235467691130, -0.405845151377397167,
                               -0.207784955007898468, 0.0, 0.207784955007898468,
                               0.405845151377397167, 0.586087235467691130, 0.741531185599394440,
                               0.864864423359769073, 0.949107912342758525, 0.991455371120812639},
        std::array<double, 15>{0.022935322010529225, 0.063092092629978553, 0.104790010322250184,
                               0.140653259715525919, 0.169004726639267903, 0.190350578064785410,
                               0.204432940075298892, 0.209482141084727828, 0.204432940075298892,
                               0.190350578064785410, 0.169004726639267903, 0.140653259715525919,
                               0.104790010322250184, 0.063092092629978553, 0.022935322010529225},
        std::array<double, 15>{0.0, 0.129484966168869693, 0.0, 0.279705391489276668, 0.0,
                               0.381830050505118945, 0.0, 0.417959183673469388, 0.0,
                               0.381830050505118945, 0.0, 0.279705391489276668, 0.0,
                               0.129484966168869693, 0.0},
    };
    return rule;
}

inline void validate_order(int order, const std::string &context) {
    if (order <= 0) {
        throw InvalidArgument(context + ": order must be positive");
    }
}

inline void validate_level(int level, const std::string &context) {
    if (level <= 0) {
        throw InvalidArgument(context + ": level must be positive");
    }
}

inline double binomial(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0;
    }
    if (k == 0 || k == n) {
        return 1.0;
    }
    k = std::min(k, n - k);
    double out = 1.0;
    for (int i = 1; i <= k; ++i) {
        out *= static_cast<double>(n - k + i) / static_cast<double>(i);
    }
    return out;
}

inline int clenshaw_curtis_order_from_level(int level) {
    validate_level(level, "clenshaw_curtis_order_from_level");
    if (level == 1) {
        return 1;
    }
    return (1 << (level - 1)) + 1;
}

inline int gauss_order_from_level(int level) {
    validate_level(level, "gauss_order_from_level");
    return level;
}

inline bool is_bounded_support(const PolynomialChaosDimension &dimension) {
    return dimension.family == PolynomialChaosFamily::Legendre ||
           dimension.family == PolynomialChaosFamily::Jacobi;
}

inline double standard_normal_moment(int degree) {
    if (degree % 2 == 1) {
        return 0.0;
    }
    const int half = degree / 2;
    return std::exp(std::lgamma(static_cast<double>(degree) + 1.0) -
                    static_cast<double>(half) * std::log(2.0) -
                    std::lgamma(static_cast<double>(half) + 1.0));
}

inline double shifted_beta_moment(int degree, double alpha, double beta) {
    const double a = beta + 1.0;
    const double b = alpha + 1.0;

    double moment = 0.0;
    for (int j = 0; j <= degree; ++j) {
        const double log_u_moment = std::lgamma(a + static_cast<double>(j)) + std::lgamma(a + b) -
                                    std::lgamma(a) - std::lgamma(a + b + static_cast<double>(j));
        const double u_moment = std::exp(log_u_moment);
        const double coeff = binomial(degree, j) * std::pow(2.0, static_cast<double>(j)) *
                             ((degree - j) % 2 == 0 ? 1.0 : -1.0);
        moment += coeff * u_moment;
    }

    return moment;
}

inline double probability_moment(const PolynomialChaosDimension &dimension, int degree) {
    if (degree < 0) {
        throw InvalidArgument("probability_moment: degree must be >= 0");
    }

    switch (dimension.family) {
    case PolynomialChaosFamily::Hermite:
        return standard_normal_moment(degree);

    case PolynomialChaosFamily::Legendre:
        if (degree % 2 == 1) {
            return 0.0;
        }
        return 1.0 / static_cast<double>(degree + 1);

    case PolynomialChaosFamily::Jacobi:
        return shifted_beta_moment(degree, dimension.alpha, dimension.beta);

    case PolynomialChaosFamily::Laguerre:
        return std::exp(std::lgamma(dimension.alpha + static_cast<double>(degree) + 1.0) -
                        std::lgamma(dimension.alpha + 1.0));
    }

    throw InvalidArgument("probability_moment: unsupported family");
}

inline NumericVector clenshaw_curtis_probability_weights(const PolynomialChaosDimension &dimension,
                                                         const NumericVector &nodes) {
    if (!is_bounded_support(dimension)) {
        throw InvalidArgument("clenshaw_curtis_probability_weights: Clenshaw-Curtis is only "
                              "supported for bounded Legendre/Jacobi families");
    }

    if (nodes.size() == 1) {
        NumericVector weights(1);
        weights(0) = 1.0;
        return weights;
    }

    if (dimension.family == PolynomialChaosFamily::Legendre) {
        return 0.5 * cgl_weights(static_cast<int>(nodes.size()), nodes);
    }

    const Eigen::Index n = nodes.size();
    NumericMatrix vandermonde(n, n);
    NumericVector moments(n);

    for (Eigen::Index k = 0; k < n; ++k) {
        moments(k) = probability_moment(dimension, static_cast<int>(k));
    }
    for (Eigen::Index j = 0; j < n; ++j) {
        double x_power = 1.0;
        for (Eigen::Index k = 0; k < n; ++k) {
            vandermonde(k, j) = x_power;
            x_power *= nodes(j);
        }
    }

    const NumericVector weights = vandermonde.colPivHouseholderQr().solve(moments);
    const NumericVector residual = vandermonde * weights - moments;
    if (residual.lpNorm<Eigen::Infinity>() > 1e-8) {
        throw RuntimeError("clenshaw_curtis_probability_weights: failed to recover stable "
                           "quadrature weights");
    }
    return weights;
}

inline UnivariateQuadratureRule
make_gauss_from_jacobi_matrix(const NumericVector &diag, const NumericVector &offdiag, int level) {
    NumericMatrix J = NumericMatrix::Zero(diag.size(), diag.size());
    J.diagonal() = diag;
    for (Eigen::Index i = 0; i < offdiag.size(); ++i) {
        J(i, i + 1) = offdiag(i);
        J(i + 1, i) = offdiag(i);
    }

    Eigen::SelfAdjointEigenSolver<NumericMatrix> solver(J);
    if (solver.info() != Eigen::Success) {
        throw RuntimeError("make_gauss_from_jacobi_matrix: eigendecomposition failed");
    }

    UnivariateQuadratureRule rule;
    rule.nodes = solver.eigenvalues();
    rule.weights = solver.eigenvectors().row(0).array().square().transpose();
    rule.nested = false;
    rule.level = level;
    return rule;
}

inline UnivariateQuadratureRule gauss_rule(const PolynomialChaosDimension &dimension, int order,
                                           int level) {
    validate_order(order, "gauss_rule");
    detail::validate_dimension(dimension, "gauss_rule");

    switch (dimension.family) {
    case PolynomialChaosFamily::Legendre: {
        const auto [nodes, weights_dx] = gauss_legendre_rule(order);
        UnivariateQuadratureRule rule;
        rule.nodes = nodes;
        rule.weights = 0.5 * weights_dx;
        rule.nested = false;
        rule.level = level;
        return rule;
    }

    case PolynomialChaosFamily::Hermite: {
        NumericVector diag = NumericVector::Zero(order);
        NumericVector offdiag(std::max(0, order - 1));
        for (int i = 1; i < order; ++i) {
            offdiag(i - 1) = std::sqrt(static_cast<double>(i));
        }
        return make_gauss_from_jacobi_matrix(diag, offdiag, level);
    }

    case PolynomialChaosFamily::Jacobi: {
        NumericVector diag(order);
        NumericVector offdiag(std::max(0, order - 1));
        const double alpha = dimension.alpha;
        const double beta = dimension.beta;

        diag(0) = (beta - alpha) / (alpha + beta + 2.0);
        for (int n = 1; n < order; ++n) {
            const double nn = static_cast<double>(n);
            const double nab = 2.0 * nn + alpha + beta;
            diag(n) = (beta * beta - alpha * alpha) / (nab * (nab + 2.0));
        }

        for (int n = 1; n < order; ++n) {
            const double nn = static_cast<double>(n);
            const double numerator = 4.0 * nn * (nn + alpha) * (nn + beta) * (nn + alpha + beta);
            const double denom = std::pow(2.0 * nn + alpha + beta, 2.0) *
                                 (2.0 * nn + alpha + beta - 1.0) * (2.0 * nn + alpha + beta + 1.0);
            offdiag(n - 1) = std::sqrt(numerator / denom);
        }

        return make_gauss_from_jacobi_matrix(diag, offdiag, level);
    }

    case PolynomialChaosFamily::Laguerre: {
        NumericVector diag(order);
        NumericVector offdiag(std::max(0, order - 1));
        const double alpha = dimension.alpha;
        for (int n = 0; n < order; ++n) {
            diag(n) = 2.0 * static_cast<double>(n) + alpha + 1.0;
        }
        for (int n = 1; n < order; ++n) {
            offdiag(n - 1) = std::sqrt(static_cast<double>(n) * (static_cast<double>(n) + alpha));
        }
        return make_gauss_from_jacobi_matrix(diag, offdiag, level);
    }
    }

    throw InvalidArgument("gauss_rule: unsupported family");
}

inline UnivariateQuadratureRule gauss_kronrod_rule(const PolynomialChaosDimension &dimension,
                                                   int order, int level) {
    if (dimension.family != PolynomialChaosFamily::Legendre) {
        throw InvalidArgument("gauss_kronrod_rule: Gauss-Kronrod 15 is only supported for "
                              "Legendre / uniform dimensions");
    }
    if (order != 7 && order != 15) {
        throw InvalidArgument("gauss_kronrod_rule: order must be 7 or 15");
    }

    const auto &embedded = gauss_kronrod_15_rule();
    UnivariateQuadratureRule rule;
    rule.nodes.resize(order);
    rule.weights.resize(order);
    rule.nested = true;
    rule.level = level;

    if (order == 15) {
        for (int i = 0; i < 15; ++i) {
            rule.nodes(i) = embedded.nodes[static_cast<std::size_t>(i)];
            rule.weights(i) = 0.5 * embedded.primary_weights[static_cast<std::size_t>(i)];
        }
        return rule;
    }

    int cursor = 0;
    for (int i = 0; i < 15; ++i) {
        const double w = embedded.embedded_weights[static_cast<std::size_t>(i)];
        if (w == 0.0) {
            continue;
        }
        rule.nodes(cursor) = embedded.nodes[static_cast<std::size_t>(i)];
        rule.weights(cursor) = 0.5 * w;
        ++cursor;
    }
    return rule;
}

inline UnivariateQuadratureRule clenshaw_curtis_rule(const PolynomialChaosDimension &dimension,
                                                     int order, int level) {
    if (!is_bounded_support(dimension)) {
        throw InvalidArgument("clenshaw_curtis_rule: Clenshaw-Curtis is only supported for "
                              "bounded Legendre/Jacobi dimensions");
    }

    validate_order(order, "clenshaw_curtis_rule");
    UnivariateQuadratureRule rule;
    rule.nodes = (order == 1) ? NumericVector::Zero(1) : cgl_nodes(order);
    rule.weights = clenshaw_curtis_probability_weights(dimension, rule.nodes);
    rule.nested = true;
    rule.level = level;
    return rule;
}

inline std::vector<std::vector<int>> positive_compositions(int dim, int sum) {
    std::vector<std::vector<int>> out;
    std::vector<int> current(static_cast<std::size_t>(dim), 1);

    std::function<void(int, int)> recurse = [&](int axis, int remaining) {
        if (axis == dim - 1) {
            current[static_cast<std::size_t>(axis)] = remaining;
            out.push_back(current);
            return;
        }

        for (int value = 1; value <= remaining - (dim - axis - 1); ++value) {
            current[static_cast<std::size_t>(axis)] = value;
            recurse(axis + 1, remaining - value);
        }
    };

    recurse(0, sum);
    return out;
}

inline std::string sample_key(const NumericVector &point, double tolerance) {
    if (tolerance <= 0.0) {
        throw InvalidArgument("sample_key: tolerance must be positive");
    }

    const double scale = 1.0 / tolerance;
    std::string key;
    key.reserve(static_cast<std::size_t>(point.size()) * 24u);
    for (Eigen::Index i = 0; i < point.size(); ++i) {
        const std::int64_t value = static_cast<std::int64_t>(std::llround(point(i) * scale));
        key += std::to_string(value);
        key.push_back('|');
    }
    return key;
}

} // namespace detail

/**
 * @brief Build a one-dimensional stochastic quadrature rule with a fixed order
 * @param dimension Stochastic dimension descriptor
 * @param order Number of quadrature points
 * @param rule Quadrature rule family
 * @return Univariate quadrature rule
 */
inline UnivariateQuadratureRule
stochastic_quadrature_rule(const PolynomialChaosDimension &dimension, int order,
                           StochasticQuadratureRule rule = StochasticQuadratureRule::Gauss) {
    switch (rule) {
    case StochasticQuadratureRule::Gauss:
        return detail::gauss_rule(dimension, order, order);

    case StochasticQuadratureRule::ClenshawCurtis:
        return detail::clenshaw_curtis_rule(dimension, order, order);

    case StochasticQuadratureRule::GaussKronrod15:
        return detail::gauss_kronrod_rule(dimension, order, order);

    case StochasticQuadratureRule::AutoNested:
        if (detail::is_bounded_support(dimension)) {
            return detail::clenshaw_curtis_rule(dimension, order, order);
        }
        return detail::gauss_rule(dimension, order, order);
    }

    throw InvalidArgument("stochastic_quadrature_rule: unsupported rule");
}

/**
 * @brief Build a one-dimensional stochastic quadrature rule from a refinement level
 * @param dimension Stochastic dimension descriptor
 * @param level Refinement level (>= 1)
 * @param rule Quadrature rule family
 * @return Univariate quadrature rule
 */
inline UnivariateQuadratureRule
stochastic_quadrature_level(const PolynomialChaosDimension &dimension, int level,
                            StochasticQuadratureRule rule = StochasticQuadratureRule::AutoNested) {
    detail::validate_level(level, "stochastic_quadrature_level");

    switch (rule) {
    case StochasticQuadratureRule::Gauss:
        return detail::gauss_rule(dimension, detail::gauss_order_from_level(level), level);

    case StochasticQuadratureRule::ClenshawCurtis:
        return detail::clenshaw_curtis_rule(dimension,
                                            detail::clenshaw_curtis_order_from_level(level), level);

    case StochasticQuadratureRule::GaussKronrod15: {
        if (level > 2) {
            throw InvalidArgument("stochastic_quadrature_level: Gauss-Kronrod 15 only supports "
                                  "levels 1 (7-point) and 2 (15-point)");
        }
        return detail::gauss_kronrod_rule(dimension, level == 1 ? 7 : 15, level);
    }

    case StochasticQuadratureRule::AutoNested:
        if (detail::is_bounded_support(dimension)) {
            return detail::clenshaw_curtis_rule(
                dimension, detail::clenshaw_curtis_order_from_level(level), level);
        }
        return detail::gauss_rule(dimension, detail::gauss_order_from_level(level), level);
    }

    throw InvalidArgument("stochastic_quadrature_level: unsupported rule");
}

/**
 * @brief Build the tensor-product grid from a list of one-dimensional rules
 * @param rules Vector of univariate quadrature rules
 * @return Tensor-product grid
 */
inline StochasticQuadratureGrid
tensor_product_quadrature(const std::vector<UnivariateQuadratureRule> &rules) {
    if (rules.empty()) {
        throw InvalidArgument("tensor_product_quadrature: need at least one univariate rule");
    }

    Eigen::Index total_points = 1;
    bool nested = true;
    int level = 0;
    for (const auto &rule : rules) {
        if (rule.nodes.size() == 0 || rule.weights.size() == 0) {
            throw InvalidArgument("tensor_product_quadrature: rules must contain nodes and "
                                  "weights");
        }
        if (rule.nodes.size() != rule.weights.size()) {
            throw InvalidArgument("tensor_product_quadrature: node/weight size mismatch");
        }
        if (total_points > std::numeric_limits<Eigen::Index>::max() / rule.nodes.size()) {
            throw InvalidArgument("tensor_product_quadrature: grid is too large");
        }
        total_points *= rule.nodes.size();
        nested = nested && rule.nested;
        level = std::max(level, rule.level);
    }

    StochasticQuadratureGrid grid;
    grid.samples.resize(total_points, static_cast<Eigen::Index>(rules.size()));
    grid.weights.resize(total_points);
    grid.nested = nested;
    grid.level = level;

    for (Eigen::Index row = 0; row < total_points; ++row) {
        Eigen::Index cursor = row;
        double weight = 1.0;
        for (Eigen::Index axis = static_cast<Eigen::Index>(rules.size()) - 1; axis >= 0; --axis) {
            const auto &rule = rules[static_cast<std::size_t>(axis)];
            const Eigen::Index local = cursor % rule.nodes.size();
            cursor /= rule.nodes.size();
            grid.samples(row, axis) = rule.nodes(local);
            weight *= rule.weights(local);
            if (axis == 0) {
                break;
            }
        }
        grid.weights(row) = weight;
    }

    return grid;
}

/**
 * @brief Build a Smolyak sparse grid on probability measures
 *
 * The level follows the standard convention with one-dimensional indices
 * `i_j >= 1` and total-index band `level <= |i| <= level + d - 1`.
 *
 * @param dimensions Stochastic dimension descriptors
 * @param level Sparse grid level (>= 1)
 * @param options Smolyak construction options
 * @return Sparse quadrature grid
 */
inline StochasticQuadratureGrid
smolyak_sparse_grid(const std::vector<PolynomialChaosDimension> &dimensions, int level,
                    SmolyakQuadratureOptions options = {}) {
    if (dimensions.empty()) {
        throw InvalidArgument("smolyak_sparse_grid: need at least one stochastic dimension");
    }
    detail::validate_level(level, "smolyak_sparse_grid");
    if (options.merge_tolerance <= 0.0) {
        throw InvalidArgument("smolyak_sparse_grid: merge_tolerance must be positive");
    }
    if (options.zero_weight_tolerance < 0.0) {
        throw InvalidArgument("smolyak_sparse_grid: zero_weight_tolerance must be >= 0");
    }

    struct WeightedPoint {
        NumericVector point;
        double weight = 0.0;
    };

    std::map<std::string, WeightedPoint> merged;
    const int dim = static_cast<int>(dimensions.size());
    const int q = level + dim - 1;
    bool fully_nested = true;

    for (int sum = level; sum <= q; ++sum) {
        const double combination =
            (((q - sum) % 2) == 0 ? 1.0 : -1.0) * detail::binomial(dim - 1, q - sum);
        if (combination == 0.0) {
            continue;
        }

        for (const auto &index : detail::positive_compositions(dim, sum)) {
            std::vector<UnivariateQuadratureRule> rules;
            rules.reserve(dimensions.size());
            for (int axis = 0; axis < dim; ++axis) {
                UnivariateQuadratureRule rule = stochastic_quadrature_level(
                    dimensions[static_cast<std::size_t>(axis)],
                    index[static_cast<std::size_t>(axis)], options.rule);
                fully_nested = fully_nested && rule.nested;
                rules.push_back(std::move(rule));
            }

            StochasticQuadratureGrid tensor = tensor_product_quadrature(rules);
            for (Eigen::Index row = 0; row < tensor.samples.rows(); ++row) {
                NumericVector point = tensor.samples.row(row).transpose();
                const std::string key = detail::sample_key(point, options.merge_tolerance);
                auto &entry = merged[key];
                if (entry.point.size() == 0) {
                    entry.point = std::move(point);
                }
                entry.weight += combination * tensor.weights(row);
            }
        }
    }

    std::vector<WeightedPoint> points;
    points.reserve(merged.size());
    for (const auto &[key, value] : merged) {
        (void)key;
        if (std::abs(value.weight) <= options.zero_weight_tolerance) {
            continue;
        }
        points.push_back(value);
    }

    if (points.empty()) {
        throw RuntimeError("smolyak_sparse_grid: all points cancelled numerically");
    }

    StochasticQuadratureGrid grid;
    grid.samples.resize(static_cast<Eigen::Index>(points.size()), static_cast<Eigen::Index>(dim));
    grid.weights.resize(static_cast<Eigen::Index>(points.size()));
    grid.nested = fully_nested;
    grid.level = level;

    for (std::size_t i = 0; i < points.size(); ++i) {
        grid.samples.row(static_cast<Eigen::Index>(i)) = points[i].point.transpose();
        grid.weights(static_cast<Eigen::Index>(i)) = points[i].weight;
    }

    return grid;
}

/**
 * @brief Compute PCE projection coefficients using a univariate quadrature rule (vector)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param rule Univariate quadrature rule
 * @param sample_values Function values at quadrature nodes
 * @return PCE coefficient vector
 */
template <MetisScalar Scalar>
MetisVector<Scalar> pce_projection_coefficients(const PolynomialChaosBasis &basis,
                                                const UnivariateQuadratureRule &rule,
                                                const MetisVector<Scalar> &sample_values) {
    if (basis.dimension() != 1) {
        throw InvalidArgument("pce_projection_coefficients(rule): univariate rule requires a "
                              "one-dimensional PolynomialChaosBasis");
    }

    NumericMatrix samples(rule.nodes.size(), 1);
    samples.col(0) = rule.nodes;
    return pce_projection_coefficients(basis, samples, rule.weights, sample_values);
}

/**
 * @brief Compute PCE projection coefficients using a univariate quadrature rule (matrix)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param rule Univariate quadrature rule
 * @param sample_values Function values at quadrature nodes (matrix)
 * @return PCE coefficient matrix
 */
template <MetisScalar Scalar>
MetisMatrix<Scalar> pce_projection_coefficients(const PolynomialChaosBasis &basis,
                                                const UnivariateQuadratureRule &rule,
                                                const MetisMatrix<Scalar> &sample_values) {
    if (basis.dimension() != 1) {
        throw InvalidArgument("pce_projection_coefficients(rule): univariate rule requires a "
                              "one-dimensional PolynomialChaosBasis");
    }

    NumericMatrix samples(rule.nodes.size(), 1);
    samples.col(0) = rule.nodes;
    return pce_projection_coefficients(basis, samples, rule.weights, sample_values);
}

/**
 * @brief Compute PCE projection coefficients using a stochastic quadrature grid (vector)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param grid Stochastic quadrature grid
 * @param sample_values Function values at grid points
 * @return PCE coefficient vector
 */
template <MetisScalar Scalar>
MetisVector<Scalar> pce_projection_coefficients(const PolynomialChaosBasis &basis,
                                                const StochasticQuadratureGrid &grid,
                                                const MetisVector<Scalar> &sample_values) {
    return pce_projection_coefficients(basis, grid.samples, grid.weights, sample_values);
}

/**
 * @brief Compute PCE projection coefficients using a stochastic quadrature grid (matrix)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param grid Stochastic quadrature grid
 * @param sample_values Function values at grid points (matrix)
 * @return PCE coefficient matrix
 */
template <MetisScalar Scalar>
MetisMatrix<Scalar> pce_projection_coefficients(const PolynomialChaosBasis &basis,
                                                const StochasticQuadratureGrid &grid,
                                                const MetisMatrix<Scalar> &sample_values) {
    return pce_projection_coefficients(basis, grid.samples, grid.weights, sample_values);
}

} // namespace metis
