#pragma once
/**
 * @file PolynomialChaos.hpp
 * @brief Polynomial chaos expansion (PCE) basis, projection, and regression
 * @see Quadrature.hpp, OrthogonalPolynomials.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Arithmetic.hpp"
#include "metis/math/OrthogonalPolynomials.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace metis {

/**
 * @brief Univariate Askey-scheme family used for a PCE input dimension.
 */
enum class PolynomialChaosFamily {
    Hermite,  ///< Standard normal random variable via probabilists' Hermite polynomials
    Legendre, ///< Uniform random variable on [-1, 1]
    Jacobi,   ///< Beta-family random variable on [-1, 1]
    Laguerre, ///< Gamma / exponential-family random variable on [0, inf)
};

/**
 * @brief Multi-index truncation strategy for a multidimensional basis.
 */
enum class PolynomialChaosTruncation {
    TotalOrder,
    TensorProduct,
};

/**
 * @brief One stochastic dimension in a polynomial chaos basis.
 *
 * `alpha` and `beta` are used only by Jacobi and Laguerre families:
 * - Jacobi: alpha > -1, beta > -1
 * - Laguerre: alpha > -1, beta ignored
 */
struct PolynomialChaosDimension {
    PolynomialChaosFamily family = PolynomialChaosFamily::Legendre;
    double alpha = 0.0;
    double beta = 0.0;
};

/**
 * @brief Create a standard normal (Hermite) dimension
 * @return Hermite dimension descriptor
 */
inline PolynomialChaosDimension hermite_dimension() {
    return PolynomialChaosDimension{PolynomialChaosFamily::Hermite, 0.0, 0.0};
}

/**
 * @brief Create a uniform (Legendre) dimension on [-1, 1]
 * @return Legendre dimension descriptor
 */
inline PolynomialChaosDimension legendre_dimension() {
    return PolynomialChaosDimension{PolynomialChaosFamily::Legendre, 0.0, 0.0};
}

/**
 * @brief Create a Beta-family (Jacobi) dimension on [-1, 1]
 * @param alpha Jacobi alpha parameter (> -1)
 * @param beta Jacobi beta parameter (> -1)
 * @return Jacobi dimension descriptor
 */
inline PolynomialChaosDimension jacobi_dimension(double alpha, double beta) {
    return PolynomialChaosDimension{PolynomialChaosFamily::Jacobi, alpha, beta};
}

/**
 * @brief Create a Gamma-family (Laguerre) dimension on [0, inf)
 * @param alpha Laguerre alpha parameter (> -1)
 * @return Laguerre dimension descriptor
 */
inline PolynomialChaosDimension laguerre_dimension(double alpha = 0.0) {
    return PolynomialChaosDimension{PolynomialChaosFamily::Laguerre, alpha, 0.0};
}

/**
 * @brief Basis construction controls for multidimensional PCE.
 */
struct PolynomialChaosBasisOptions {
    PolynomialChaosTruncation truncation = PolynomialChaosTruncation::TotalOrder;
    bool normalized = true; ///< Use orthonormal basis functions when true
};

/**
 * @brief One multidimensional chaos basis term.
 */
struct PolynomialChaosTerm {
    std::vector<int> multi_index;
    double squared_norm = 1.0;
};

namespace detail {

inline void validate_degree(int degree, const std::string &context) {
    if (degree < 0) {
        throw InvalidArgument(context + ": degree must be >= 0");
    }
}

inline void validate_dimension(const PolynomialChaosDimension &dimension,
                               const std::string &context) {
    switch (dimension.family) {
    case PolynomialChaosFamily::Hermite:
    case PolynomialChaosFamily::Legendre:
        return;
    case PolynomialChaosFamily::Jacobi:
        if (dimension.alpha <= -1.0 || dimension.beta <= -1.0) {
            throw InvalidArgument(context + ": Jacobi parameters must satisfy alpha > -1 and "
                                            "beta > -1");
        }
        return;
    case PolynomialChaosFamily::Laguerre:
        if (dimension.alpha <= -1.0) {
            throw InvalidArgument(context + ": Laguerre parameter must satisfy alpha > -1");
        }
        return;
    }
}

template <MetisScalar Scalar> Scalar raw_hermite_polynomial(int degree, const Scalar &x) {
    validate_degree(degree, "raw_hermite_polynomial");

    if (degree == 0) {
        return Scalar(1.0);
    }
    if (degree == 1) {
        return x;
    }

    Scalar h_nm1 = Scalar(1.0);
    Scalar h_n = x;
    for (int n = 1; n < degree; ++n) {
        Scalar h_np1 = x * h_n - static_cast<double>(n) * h_nm1;
        h_nm1 = h_n;
        h_n = h_np1;
    }
    return h_n;
}

template <MetisScalar Scalar> Scalar raw_legendre_polynomial(int degree, const Scalar &x) {
    validate_degree(degree, "raw_legendre_polynomial");

    if constexpr (std::floating_point<Scalar>) {
        return legendre_poly(degree, x).first;
    } else {
        if (degree == 0) {
            return Scalar(1.0);
        }
        if (degree == 1) {
            return x;
        }

        Scalar p_nm1 = Scalar(1.0);
        Scalar p_n = x;
        for (int n = 1; n < degree; ++n) {
            const double nn = static_cast<double>(n);
            Scalar p_np1 = ((2.0 * nn + 1.0) * x * p_n - nn * p_nm1) / (nn + 1.0);
            p_nm1 = p_n;
            p_n = p_np1;
        }
        return p_n;
    }
}

template <MetisScalar Scalar>
Scalar raw_jacobi_polynomial(int degree, const Scalar &x, double alpha, double beta) {
    validate_degree(degree, "raw_jacobi_polynomial");
    validate_dimension(jacobi_dimension(alpha, beta), "raw_jacobi_polynomial");

    if (degree == 0) {
        return Scalar(1.0);
    }
    if (degree == 1) {
        return 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);
    }

    Scalar p_nm1 = Scalar(1.0);
    Scalar p_n = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);

    for (int n = 1; n < degree; ++n) {
        const double nn = static_cast<double>(n);
        const double two_n_ab = 2.0 * nn + alpha + beta;
        const double a_n = 2.0 * (nn + 1.0) * (nn + alpha + beta + 1.0) * two_n_ab;
        const Scalar b_n =
            (two_n_ab + 1.0) * (two_n_ab * (two_n_ab + 2.0) * x + alpha * alpha - beta * beta);
        const double c_n = 2.0 * (nn + alpha) * (nn + beta) * (two_n_ab + 2.0);
        Scalar p_np1 = (b_n * p_n - c_n * p_nm1) / a_n;
        p_nm1 = p_n;
        p_n = p_np1;
    }

    return p_n;
}

template <MetisScalar Scalar>
Scalar raw_laguerre_polynomial(int degree, const Scalar &x, double alpha) {
    validate_degree(degree, "raw_laguerre_polynomial");
    validate_dimension(laguerre_dimension(alpha), "raw_laguerre_polynomial");

    if (degree == 0) {
        return Scalar(1.0);
    }
    if (degree == 1) {
        return Scalar(1.0 + alpha) - x;
    }

    Scalar l_nm1 = Scalar(1.0);
    Scalar l_n = Scalar(1.0 + alpha) - x;

    for (int n = 1; n < degree; ++n) {
        const double nn = static_cast<double>(n);
        Scalar l_np1 = ((2.0 * nn + 1.0 + alpha - x) * l_n - (nn + alpha) * l_nm1) / (nn + 1.0);
        l_nm1 = l_n;
        l_n = l_np1;
    }

    return l_n;
}

inline double squared_norm_raw(const PolynomialChaosDimension &dimension, int degree) {
    validate_degree(degree, "squared_norm_raw");
    validate_dimension(dimension, "squared_norm_raw");

    switch (dimension.family) {
    case PolynomialChaosFamily::Hermite:
        return std::exp(std::lgamma(static_cast<double>(degree) + 1.0));

    case PolynomialChaosFamily::Legendre:
        return 2.0 / (2.0 * static_cast<double>(degree) + 1.0);

    case PolynomialChaosFamily::Jacobi: {
        const double n = static_cast<double>(degree);
        const double alpha = dimension.alpha;
        const double beta = dimension.beta;
        const double log_raw = (alpha + beta + 1.0) * std::log(2.0) -
                               std::log(2.0 * n + alpha + beta + 1.0) +
                               std::lgamma(n + alpha + 1.0) + std::lgamma(n + beta + 1.0) -
                               std::lgamma(n + 1.0) - std::lgamma(n + alpha + beta + 1.0);
        return std::exp(log_raw);
    }

    case PolynomialChaosFamily::Laguerre:
        return std::exp(std::lgamma(static_cast<double>(degree) + dimension.alpha + 1.0) -
                        std::lgamma(static_cast<double>(degree) + 1.0));
    }

    throw InvalidArgument("squared_norm_raw: unsupported family");
}

inline double squared_norm_probability(const PolynomialChaosDimension &dimension, int degree) {
    if (dimension.family == PolynomialChaosFamily::Hermite) {
        return squared_norm_raw(dimension, degree);
    }
    return squared_norm_raw(dimension, degree) / squared_norm_raw(dimension, 0);
}

inline bool multi_index_less(const std::vector<int> &lhs, const std::vector<int> &rhs) {
    const int lhs_sum = std::accumulate(lhs.begin(), lhs.end(), 0);
    const int rhs_sum = std::accumulate(rhs.begin(), rhs.end(), 0);
    if (lhs_sum != rhs_sum) {
        return lhs_sum < rhs_sum;
    }
    return lhs < rhs;
}

inline void total_order_recursive(int dim, int order, int axis, std::vector<int> &current,
                                  std::vector<std::vector<int>> &indices) {
    if (axis == dim) {
        indices.push_back(current);
        return;
    }

    const int used = std::accumulate(current.begin(), current.begin() + axis, 0);
    const int remaining = order - used;
    for (int degree = 0; degree <= remaining; ++degree) {
        current[axis] = degree;
        total_order_recursive(dim, order, axis + 1, current, indices);
    }
}

inline void tensor_product_recursive(int dim, int order, int axis, std::vector<int> &current,
                                     std::vector<std::vector<int>> &indices) {
    if (axis == dim) {
        indices.push_back(current);
        return;
    }

    for (int degree = 0; degree <= order; ++degree) {
        current[axis] = degree;
        tensor_product_recursive(dim, order, axis + 1, current, indices);
    }
}

inline std::vector<std::vector<int>> generate_multi_indices(int dim, int order,
                                                            PolynomialChaosTruncation truncation) {
    std::vector<std::vector<int>> indices;
    std::vector<int> current(static_cast<std::size_t>(dim), 0);

    switch (truncation) {
    case PolynomialChaosTruncation::TotalOrder:
        total_order_recursive(dim, order, 0, current, indices);
        break;
    case PolynomialChaosTruncation::TensorProduct:
        tensor_product_recursive(dim, order, 0, current, indices);
        break;
    }

    std::sort(indices.begin(), indices.end(), multi_index_less);
    return indices;
}

template <MetisScalar Scalar>
MetisVector<Scalar> apply_operator(const NumericMatrix &op, const MetisVector<Scalar> &values,
                                   const std::string &context) {
    if (values.rows() != op.cols()) {
        throw InvalidArgument(context + ": sample value size must match the number of rows in the "
                                        "design matrix");
    }

    MetisVector<Scalar> out(op.rows());
    for (Eigen::Index i = 0; i < op.rows(); ++i) {
        Scalar accum = Scalar(0.0);
        for (Eigen::Index j = 0; j < op.cols(); ++j) {
            accum += op(i, j) * values(j);
        }
        out(i) = accum;
    }
    return out;
}

template <MetisScalar Scalar>
MetisMatrix<Scalar> apply_operator(const NumericMatrix &op, const MetisMatrix<Scalar> &values,
                                   const std::string &context) {
    if (values.rows() != op.cols()) {
        throw InvalidArgument(context + ": sample value rows must match the number of rows in the "
                                        "design matrix");
    }

    MetisMatrix<Scalar> out(op.rows(), values.cols());
    for (Eigen::Index i = 0; i < op.rows(); ++i) {
        for (Eigen::Index col = 0; col < values.cols(); ++col) {
            Scalar accum = Scalar(0.0);
            for (Eigen::Index j = 0; j < op.cols(); ++j) {
                accum += op(i, j) * values(j, col);
            }
            out(i, col) = accum;
        }
    }
    return out;
}

inline void validate_samples(const NumericMatrix &samples, int dimension,
                             const std::string &context) {
    if (samples.rows() == 0) {
        throw InvalidArgument(context + ": sample matrix must contain at least one sample");
    }
    if (samples.cols() != dimension) {
        throw InvalidArgument(context + ": sample matrix column count must match the PCE "
                                        "dimension");
    }
}

inline NumericMatrix regression_operator(const NumericMatrix &design_matrix, double ridge,
                                         const std::string &context) {
    if (design_matrix.rows() < design_matrix.cols()) {
        throw InvalidArgument(context + ": need at least as many samples as basis terms for "
                                        "regression");
    }
    if (ridge < 0.0) {
        throw InvalidArgument(context + ": ridge regularization must be >= 0");
    }

    if (ridge == 0.0) {
        Eigen::JacobiSVD<NumericMatrix> svd(design_matrix,
                                            Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.rank() < design_matrix.cols()) {
            throw InvalidArgument(context + ": design matrix is rank deficient");
        }
        return svd.solve(NumericMatrix::Identity(design_matrix.rows(), design_matrix.rows()));
    }

    NumericMatrix gram = design_matrix.transpose() * design_matrix;
    if (ridge > 0.0) {
        gram.diagonal().array() += ridge;
    }
    return gram.ldlt().solve(design_matrix.transpose());
}

} // namespace detail

/**
 * @brief Evaluate a univariate chaos basis polynomial.
 *
 * The input `x` is assumed to already be expressed on the natural support of
 * the chosen family:
 * - Hermite: standard normal variable
 * - Legendre/Jacobi: support [-1, 1]
 * - Laguerre: support [0, inf)
 */
template <MetisScalar Scalar>
Scalar pce_polynomial(const PolynomialChaosDimension &dimension, int degree, const Scalar &x,
                      bool normalized = true) {
    detail::validate_degree(degree, "pce_polynomial");
    detail::validate_dimension(dimension, "pce_polynomial");

    Scalar value = Scalar(0.0);
    switch (dimension.family) {
    case PolynomialChaosFamily::Hermite:
        value = detail::raw_hermite_polynomial(degree, x);
        break;
    case PolynomialChaosFamily::Legendre:
        value = detail::raw_legendre_polynomial(degree, x);
        break;
    case PolynomialChaosFamily::Jacobi:
        value = detail::raw_jacobi_polynomial(degree, x, dimension.alpha, dimension.beta);
        break;
    case PolynomialChaosFamily::Laguerre:
        value = detail::raw_laguerre_polynomial(degree, x, dimension.alpha);
        break;
    }

    if (!normalized) {
        return value;
    }

    const double norm = detail::squared_norm_probability(dimension, degree);
    return value / metis::sqrt(Scalar(norm));
}

/**
 * @brief Return the probability-measure squared norm of a univariate basis term.
 */
inline double pce_squared_norm(const PolynomialChaosDimension &dimension, int degree,
                               bool normalized = true) {
    if (normalized) {
        return 1.0;
    }
    return detail::squared_norm_probability(dimension, degree);
}

/**
 * @brief Multidimensional polynomial chaos basis with fixed truncation/order.
 */
class PolynomialChaosBasis {
  public:
    PolynomialChaosBasis() = default;

    PolynomialChaosBasis(std::vector<PolynomialChaosDimension> dimensions, int order,
                         PolynomialChaosBasisOptions options = {})
        : dimensions_(std::move(dimensions)), order_(order), options_(options) {
        if (dimensions_.empty()) {
            throw InvalidArgument("PolynomialChaosBasis: need at least one stochastic dimension");
        }
        if (order_ < 0) {
            throw InvalidArgument("PolynomialChaosBasis: order must be >= 0");
        }
        for (const auto &dimension : dimensions_) {
            detail::validate_dimension(dimension, "PolynomialChaosBasis");
        }

        const std::vector<std::vector<int>> indices = detail::generate_multi_indices(
            static_cast<int>(dimensions_.size()), order_, options_.truncation);

        terms_.reserve(indices.size());
        squared_norms_.resize(static_cast<Eigen::Index>(indices.size()));
        for (std::size_t i = 0; i < indices.size(); ++i) {
            double norm = 1.0;
            for (std::size_t axis = 0; axis < dimensions_.size(); ++axis) {
                norm *= pce_squared_norm(dimensions_[axis], indices[i][axis], options_.normalized);
            }
            terms_.push_back(PolynomialChaosTerm{indices[i], norm});
            squared_norms_(static_cast<Eigen::Index>(i)) = norm;
        }
    }

    int dimension() const { return static_cast<int>(dimensions_.size()); }

    int order() const { return order_; }

    int size() const { return static_cast<int>(terms_.size()); }

    bool normalized() const { return options_.normalized; }

    PolynomialChaosTruncation truncation() const { return options_.truncation; }

    const std::vector<PolynomialChaosDimension> &dimensions() const { return dimensions_; }

    const std::vector<PolynomialChaosTerm> &terms() const { return terms_; }

    const NumericVector &squared_norms() const { return squared_norms_; }

    template <MetisScalar Scalar>
    MetisVector<Scalar> evaluate(const MetisVector<Scalar> &point) const {
        if (point.size() != dimension()) {
            throw InvalidArgument("PolynomialChaosBasis::evaluate(point): point dimension must "
                                  "match the basis dimension");
        }

        MetisVector<Scalar> values(size());
        for (int term_idx = 0; term_idx < size(); ++term_idx) {
            Scalar value = Scalar(1.0);
            for (int axis = 0; axis < dimension(); ++axis) {
                value *= pce_polynomial(dimensions_[static_cast<std::size_t>(axis)],
                                        terms_[static_cast<std::size_t>(term_idx)]
                                            .multi_index[static_cast<std::size_t>(axis)],
                                        point(axis), options_.normalized);
            }
            values(term_idx) = value;
        }
        return values;
    }

    NumericMatrix evaluate(const NumericMatrix &samples) const {
        detail::validate_samples(samples, dimension(), "PolynomialChaosBasis::evaluate(samples)");

        NumericMatrix design(samples.rows(), size());
        for (Eigen::Index row = 0; row < samples.rows(); ++row) {
            NumericVector point = samples.row(row).transpose();
            design.row(row) = evaluate(point).transpose();
        }
        return design;
    }

  private:
    std::vector<PolynomialChaosDimension> dimensions_;
    int order_ = 0;
    PolynomialChaosBasisOptions options_;
    std::vector<PolynomialChaosTerm> terms_;
    NumericVector squared_norms_;
};

/**
 * @brief Compute PCE projection coefficients from weighted samples (vector)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param samples Sample matrix (rows = samples, cols = dimensions)
 * @param weights Quadrature weights
 * @param sample_values Function values at sample points
 * @return PCE coefficient vector
 */
template <MetisScalar Scalar>
MetisVector<Scalar> pce_projection_coefficients(const PolynomialChaosBasis &basis,
                                                const NumericMatrix &samples,
                                                const NumericVector &weights,
                                                const MetisVector<Scalar> &sample_values) {
    detail::validate_samples(samples, basis.dimension(), "pce_projection_coefficients");
    if (weights.size() != samples.rows()) {
        throw InvalidArgument("pce_projection_coefficients: weights size must match the number of "
                              "samples");
    }
    if (sample_values.size() != samples.rows()) {
        throw InvalidArgument(
            "pce_projection_coefficients: sample value size must match the number of samples");
    }

    const NumericMatrix design = basis.evaluate(samples);
    MetisVector<Scalar> coeffs(basis.size());
    for (int term_idx = 0; term_idx < basis.size(); ++term_idx) {
        Scalar accum = Scalar(0.0);
        for (Eigen::Index sample_idx = 0; sample_idx < samples.rows(); ++sample_idx) {
            accum += weights(sample_idx) * design(sample_idx, term_idx) * sample_values(sample_idx);
        }
        coeffs(term_idx) = accum / basis.squared_norms()(term_idx);
    }
    return coeffs;
}

/**
 * @brief Compute PCE projection coefficients from weighted samples (matrix)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param samples Sample matrix (rows = samples, cols = dimensions)
 * @param weights Quadrature weights
 * @param sample_values Function values at sample points (matrix)
 * @return PCE coefficient matrix
 */
template <MetisScalar Scalar>
MetisMatrix<Scalar> pce_projection_coefficients(const PolynomialChaosBasis &basis,
                                                const NumericMatrix &samples,
                                                const NumericVector &weights,
                                                const MetisMatrix<Scalar> &sample_values) {
    detail::validate_samples(samples, basis.dimension(), "pce_projection_coefficients");
    if (weights.size() != samples.rows()) {
        throw InvalidArgument("pce_projection_coefficients: weights size must match the number of "
                              "samples");
    }
    if (sample_values.rows() != samples.rows()) {
        throw InvalidArgument("pce_projection_coefficients: sample value rows must match the "
                              "number of samples");
    }

    const NumericMatrix design = basis.evaluate(samples);
    MetisMatrix<Scalar> coeffs(basis.size(), sample_values.cols());
    for (int term_idx = 0; term_idx < basis.size(); ++term_idx) {
        for (Eigen::Index col = 0; col < sample_values.cols(); ++col) {
            Scalar accum = Scalar(0.0);
            for (Eigen::Index sample_idx = 0; sample_idx < samples.rows(); ++sample_idx) {
                accum += weights(sample_idx) * design(sample_idx, term_idx) *
                         sample_values(sample_idx, col);
            }
            coeffs(term_idx, col) = accum / basis.squared_norms()(term_idx);
        }
    }
    return coeffs;
}

/**
 * @brief Compute PCE coefficients via least-squares regression (vector)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param samples Sample matrix
 * @param sample_values Function values at sample points
 * @param ridge Ridge regularization parameter
 * @return PCE coefficient vector
 */
template <MetisScalar Scalar>
MetisVector<Scalar>
pce_regression_coefficients(const PolynomialChaosBasis &basis, const NumericMatrix &samples,
                            const MetisVector<Scalar> &sample_values, double ridge = 1e-12) {
    detail::validate_samples(samples, basis.dimension(), "pce_regression_coefficients");
    const NumericMatrix design = basis.evaluate(samples);
    const NumericMatrix op =
        detail::regression_operator(design, ridge, "pce_regression_coefficients");
    return detail::apply_operator(op, sample_values, "pce_regression_coefficients");
}

/**
 * @brief Compute PCE coefficients via least-squares regression (matrix)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param samples Sample matrix
 * @param sample_values Function values at sample points (matrix)
 * @param ridge Ridge regularization parameter
 * @return PCE coefficient matrix
 */
template <MetisScalar Scalar>
MetisMatrix<Scalar>
pce_regression_coefficients(const PolynomialChaosBasis &basis, const NumericMatrix &samples,
                            const MetisMatrix<Scalar> &sample_values, double ridge = 1e-12) {
    detail::validate_samples(samples, basis.dimension(), "pce_regression_coefficients");
    const NumericMatrix design = basis.evaluate(samples);
    const NumericMatrix op =
        detail::regression_operator(design, ridge, "pce_regression_coefficients");
    return detail::apply_operator(op, sample_values, "pce_regression_coefficients");
}

/**
 * @brief Extract PCE mean (zeroth coefficient)
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param coefficients PCE coefficient vector
 * @return Mean value
 */
template <MetisScalar Scalar> Scalar pce_mean(const MetisVector<Scalar> &coefficients) {
    if (coefficients.size() == 0) {
        throw InvalidArgument("pce_mean: coefficient vector must be non-empty");
    }
    return coefficients(0);
}

/**
 * @brief Compute PCE variance from coefficients
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param basis Polynomial chaos basis
 * @param coefficients PCE coefficient vector
 * @return Variance
 */
template <MetisScalar Scalar>
Scalar pce_variance(const PolynomialChaosBasis &basis, const MetisVector<Scalar> &coefficients) {
    if (coefficients.size() != basis.size()) {
        throw InvalidArgument("pce_variance: coefficient vector size must match the basis size");
    }

    Scalar variance = Scalar(0.0);
    for (int i = 1; i < coefficients.size(); ++i) {
        variance += basis.squared_norms()(i) * coefficients(i) * coefficients(i);
    }
    return variance;
}

} // namespace metis
