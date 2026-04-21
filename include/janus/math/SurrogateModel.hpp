#pragma once
/**
 * @file SurrogateModel.hpp
 * @brief Smooth surrogate functions (softmax, softmin, softplus, sigmoid, blend, KS aggregation)
 * @see Logic.hpp, Arithmetic.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/math/Arithmetic.hpp"
#include "metis/math/Logic.hpp"
#include "metis/math/Trig.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace metis {

// ======================================================================
// Softmax (Smooth Maximum / LogSumExp)
// ======================================================================

/**
 * @brief Computes the smooth maximum (LogSumExp) of a collection of values.
 *
 * Approximation of max(x1, x2, ...) that is differentiable.
 * out = softness * log(sum(exp(x_i / softness)))
 *
 * Implemented with shift for numerical stability:
 * out = max_val + softness * log(sum(exp((x_i - max_val) / softness)))
 *
 * @param args Vector of values (scalars or matrices)
 * @param softness Smoothing parameter (default 1.0). Higher = smoother (further from max).
 * @return Smooth maximum
 */
template <typename T> auto softmax(const std::vector<T> &args, double softness = 1.0) {
    if (args.empty()) {
        throw InvalidArgument("softmax: requires at least one value");
    }
    if (softness <= 0.0) {
        throw InvalidArgument("softmax: softness must be positive");
    }

    // 1. Find max element-wise
    auto max_val = args[0];
    for (size_t i = 1; i < args.size(); ++i) {
        max_val = metis::max(max_val, args[i]);
    }

    // 2. Compute sum of exponentials: sum(exp((x - max) / softness))
    // We need a way to initialize the sum with the correct type and dimensions.
    // We can compute the first term to initialize.

    // Using auto type deduction for the result of exp operation
    auto first_term = metis::exp((args[0] - max_val) / softness);
    auto sum_exp = first_term;

    for (size_t i = 1; i < args.size(); ++i) {
        sum_exp = sum_exp + metis::exp((args[i] - max_val) / softness);
    }

    // 3. Final computation
    return max_val + softness * metis::log(sum_exp);
}

// Convenience overload for 2 arguments
template <typename T1, typename T2> auto softmax(const T1 &a, const T2 &b, double softness = 1.0) {
    // We need to construct a vector, but T1 and T2 might be different types (double vs Matrix)
    // This is tricky if types are different.
    // Ideally we assume they are compatible or castable to a common type.
    // For simplicity in the generic case, let's implement the 2-arg logic directly to allow
    // standard promotions.

    auto max_val = metis::max(a, b);

    // Compute exp terms
    // exp((a - max) / softness) + exp((b - max) / softness)
    // Note: One of the exponents will include (max - max) = 0, so exp(0) = 1.
    // But we just compute blindly for simplicity and consistency.

    auto term1 = metis::exp((a - max_val) / softness);
    auto term2 = metis::exp((b - max_val) / softness);

    // Check if we need to broadcast scalar to matrix if one is matrix and other is scalar
    // metis::exp usually handles this if implemented correctly or if using Eigen arrays.
    // However, addition `term1 + term2` might fail if one is scalar and other is matrix in standard
    // Eigen without array(). We assume metis::exp returns compatible types (Eigen matrix or
    // scalar). If mixed scalar/matrix, standard Eigen requires array operation or broadcasting.

    // Let's rely on metis::exp returning something compatible with operator+
    return max_val + softness * metis::log(term1 + term2);
}

// ======================================================================
// Softmin (Smooth Minimum)
// ======================================================================

/**
 * @brief Computes the smooth minimum of a collection of values.
 * softmin(x) = -softmax(-x)
 *
 * @param args Vector of values
 * @param softness Smoothing parameter
 * @return Smooth minimum
 */
template <typename T> auto softmin(const std::vector<T> &args, double softness = 1.0) {
    std::vector<T> neg_args;
    neg_args.reserve(args.size());
    for (const auto &arg : args) {
        neg_args.push_back(-arg);
    }
    return -softmax(neg_args, softness);
}

// Convenience overload for 2 arguments
template <typename T1, typename T2> auto softmin(const T1 &a, const T2 &b, double softness = 1.0) {
    return -softmax(-a, -b, softness);
}

// ======================================================================
// Softplus (Smooth ReLU)
// ======================================================================

/**
 * @brief Smooth approximation of ReLU function: softplus(x) = (1/beta) * log(1 + exp(beta * x))
 *
 * @param x Input value
 * @param beta Sharpness parameter (default 1.0). Higher = closer to ReLU.
 * @param threshold Retained for backwards compatibility. Unused by the smooth implementation.
 * @return Softplus value
 */
namespace detail {
inline void validate_hardness(double hardness, const char *function_name) {
    if (hardness <= 0.0) {
        throw InvalidArgument(std::string(function_name) + ": hardness must be positive");
    }
}

inline void validate_rho(double rho) {
    if (rho <= 0.0) {
        throw InvalidArgument("ks_max: rho must be positive");
    }
}

template <std::floating_point T> auto softplus_scalar(const T &x, double beta) {
    const auto bx = beta * x;
    const auto shift = bx > 0.0 ? bx : 0.0;
    return (shift + std::log(std::exp(-shift) + std::exp(bx - shift))) / beta;
}

inline auto softplus_scalar(const SymbolicScalar &x, double beta) {
    const auto bx = beta * x;
    // CasADi's logsumexp applies max-shift stabilization internally:
    // logsumexp(a) = max(a) + log(sum(exp(a - max(a))))
    return SymbolicScalar::logsumexp(SymbolicScalar::vertcat({SymbolicScalar(0.0), bx})) / beta;
}
} // namespace detail

template <MetisScalar T> auto softplus(const T &x, double beta = 1.0, double /*threshold*/ = 20.0) {
    return detail::softplus_scalar(x, beta);
}

template <typename Derived>
auto softplus(const Eigen::MatrixBase<Derived> &x, double beta = 1.0, double threshold = 20.0) {
    using Scalar = typename Derived::Scalar;
    return x.derived().unaryExpr(
        [beta, threshold](const Scalar &v) { return metis::softplus(v, beta, threshold); });
}

// ======================================================================
// Smooth Approximation Suite
// ======================================================================

/**
 * @brief Smooth approximation of absolute value.
 *
 * Implemented as smooth_max(x, -x), which is C-infinity and approaches |x|
 * as hardness -> infinity.
 *
 * @param x Input value
 * @param hardness Sharpness parameter. Higher values approach abs(x).
 * @return Smooth approximation of |x|
 */
template <typename T> auto smooth_abs(const T &x, double hardness = 1.0) {
    detail::validate_hardness(hardness, "smooth_abs");
    return -x + softplus(2.0 * x, hardness);
}

/**
 * @brief Pairwise smooth maximum.
 *
 * Uses the identity max(a, b) = b + relu(a - b), replacing relu with softplus.
 * This remains C-infinity and approaches max(a, b) as hardness -> infinity.
 *
 * @param a First value
 * @param b Second value
 * @param hardness Sharpness parameter. Higher values approach max(a, b).
 * @return Smooth approximation of max(a, b)
 */
template <typename T1, typename T2>
auto smooth_max(const T1 &a, const T2 &b, double hardness = 1.0) {
    detail::validate_hardness(hardness, "smooth_max");
    return b + softplus(a - b, hardness);
}

/**
 * @brief Pairwise smooth minimum.
 *
 * Uses the identity min(a, b) = a - relu(a - b), replacing relu with softplus.
 * This remains C-infinity and approaches min(a, b) as hardness -> infinity.
 *
 * @param a First value
 * @param b Second value
 * @param hardness Sharpness parameter. Higher values approach min(a, b).
 * @return Smooth approximation of min(a, b)
 */
template <typename T1, typename T2>
auto smooth_min(const T1 &a, const T2 &b, double hardness = 1.0) {
    detail::validate_hardness(hardness, "smooth_min");
    return a - softplus(a - b, hardness);
}

/**
 * @brief Smooth approximation of clamp(x, low, high).
 *
 * Implemented by composing smooth_max and smooth_min.
 *
 * @param x Input value
 * @param low Lower bound
 * @param high Upper bound
 * @param hardness Sharpness parameter. Higher values approach hard clamping.
 * @return Smooth approximation of clamp(x, low, high)
 */
template <typename TX, typename TLow, typename THigh>
auto smooth_clamp(const TX &x, const TLow &low, const THigh &high, double hardness = 1.0) {
    detail::validate_hardness(hardness, "smooth_clamp");
    return smooth_min(smooth_max(x, low, hardness), high, hardness);
}

/**
 * @brief Kreisselmeier-Steinhauser smooth maximum aggregator.
 *
 * Commonly used to aggregate many constraints into one smooth surrogate:
 * ks_max(values, rho) = (1/rho) * log(sum(exp(rho * values_i)))
 *
 * @param values Values to aggregate
 * @param rho Aggregation sharpness. Higher values approach max(values).
 * @return Smooth maximum aggregate
 */
template <MetisScalar T> auto ks_max(const std::vector<T> &values, double rho = 1.0) {
    if (values.empty()) {
        throw InvalidArgument("ks_max: requires at least one value");
    }
    detail::validate_rho(rho);

    if constexpr (std::is_floating_point_v<T>) {
        const auto max_val = *std::max_element(values.begin(), values.end());
        double sum_exp = 0.0;
        for (const auto &value : values) {
            sum_exp += std::exp(rho * (value - max_val));
        }
        return max_val + std::log(sum_exp) / rho;
    } else {
        return SymbolicScalar::logsumexp(rho * SymbolicScalar::vertcat(values)) / rho;
    }
}

template <typename Derived>
auto ks_max(const Eigen::MatrixBase<Derived> &values, double rho = 1.0) {
    using Scalar = typename Derived::Scalar;
    std::vector<Scalar> flattened;
    flattened.reserve(static_cast<size_t>(values.size()));
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        flattened.push_back(values.derived().coeff(i));
    }
    return ks_max(flattened, rho);
}

// ======================================================================
// Sigmoid
// ======================================================================

/**
 * @brief Sigmoid shape selection
 */
enum class SigmoidType {
    Tanh,      ///< Hyperbolic tangent
    Logistic,  ///< Logistic / standard sigmoid
    Arctan,    ///< Arc tangent based
    Polynomial ///< Polynomial: x / sqrt(1 + x^2)
};

/**
 * @brief Sigmoid function with normalization capability.
 *
 * @param x Input value
 * @param type Type of sigmoid shape (Tanh, Arctan, Polynomial)
 * @param norm_min Minimum output value (asymptote at -inf)
 * @param norm_max Maximum output value (asymptote at +inf)
 * @return Sigmoid value scaled to [norm_min, norm_max]
 */
template <typename T>
auto sigmoid(const T &x, SigmoidType type = SigmoidType::Tanh, double norm_min = 0.0,
             double norm_max = 1.0) {

    // 1. Compute base sigmoid 's' in range [-1, 1] if possible, or standard form
    // The reference implementation normalizes based on the theoretical raw range of the function.
    // Tanh: range (-1, 1)
    // Arctan: scaled 2/pi * atan(pi/2 * x) -> range (-1, 1)
    // Polynomial: x / sqrt(1 + x^2) -> range (-1, 1)

    auto s = x; // placeholder type initialization

    switch (type) {
    case SigmoidType::Tanh:
        s = metis::tanh(x);
        break;
    case SigmoidType::Logistic:
        s = metis::tanh(0.5 * x);
        break;
    case SigmoidType::Arctan:
        s = (2.0 / M_PI) * metis::atan((M_PI / 2.0) * x);
        break;
    case SigmoidType::Polynomial:
        s = x / metis::sqrt(1.0 + x * x);
        break;
    }

    // 2. Normalize from [-1, 1] to [norm_min, norm_max]
    // s_normalized = s * (max - min) / 2 + (max + min) / 2
    double scale = (norm_max - norm_min) / 2.0;
    double offset = (norm_max + norm_min) / 2.0;

    return s * scale + offset;
}

// ======================================================================
// Swish
// ======================================================================

/**
 * @brief Swish activation function: x / (1 + exp(-beta * x))
 *
 * @param x Input
 * @param beta Beta parameter
 * @return Swish value
 */
template <typename T> auto swish(const T &x, double beta = 1.0) {
    return x / (1.0 + metis::exp(-beta * x));
}

// ======================================================================
// Blend
// ======================================================================

/**
 * @brief Smoothly blends between two values based on a switch parameter.
 *
 * @param switch_val Control value. 0 -> average, -inf -> low, +inf -> high.
 * @param val_high Value when switch is high
 * @param val_low Value when switch is low
 * @return Blended value
 */
template <typename TSwitch, typename THigh, typename TLow>
auto blend(const TSwitch &switch_val, const THigh &val_high, const TLow &val_low) {
    // Helper sigmoid (tanh type, 0 to 1 range)
    auto weights = sigmoid(switch_val, SigmoidType::Tanh, 0.0, 1.0);

    return val_high * weights + val_low * (1.0 - weights);
}

} // namespace metis
