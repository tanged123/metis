#pragma once
/**
 * @file Arithmetic.hpp
 * @brief Scalar and element-wise arithmetic functions (abs, sqrt, pow, exp, log, etc.)
 * @see Trig.hpp, Logic.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace metis {

// --- Absolute Value ---
/**
 * @brief Computes the absolute value of a scalar
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Absolute value of x
 */
template <MetisScalar T> T abs(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(x);
    } else {
        return fabs(x);
    }
}

/**
 * @brief Computes absolute value element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of absolute values
 */
template <typename Derived> auto abs(const Eigen::MatrixBase<Derived> &x) {
    return x.array().abs().matrix();
}

// --- Square Root ---
/**
 * @brief Computes the square root of a scalar
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Square root of x
 */
template <MetisScalar T> T sqrt(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sqrt(x);
    } else {
        return sqrt(x);
    }
}

/**
 * @brief Computes square root element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of square roots
 */
template <typename Derived> auto sqrt(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sqrt().matrix();
}

// --- Power ---
/**
 * @brief Computes the power function: base^exponent
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param base Base value
 * @param exponent Exponent value
 * @return base raised to the power of exponent
 */
template <MetisScalar T> T pow(const T &base, const T &exponent) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, exponent);
    } else {
        return pow(base, exponent);
    }
}

/**
 * @brief Computes power function base^exponent for scalars (mixed types)
 * @tparam T Non-double Metis scalar (e.g., SymbolicScalar)
 * @param base Base value
 * @param exponent Exponent value
 * @return base raised to exponent
 */
template <MetisScalar T>
    requires(!std::is_same_v<T, double>)
T pow(const T &base, double exponent) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::pow(base, static_cast<T>(exponent));
    } else {
        return metis::pow(base, static_cast<T>(exponent));
    }
}

/**
 * @brief Computes power function base^exponent for scalars (mixed types: double base)
 * @tparam T Non-double Metis scalar (e.g., SymbolicScalar)
 * @param base Base value
 * @param exponent Exponent value
 * @return base raised to exponent
 */
template <MetisScalar T>
    requires(!std::is_same_v<T, double>)
T pow(double base, const T &exponent) {
    // Cast base to T to match homogeneous overload
    return metis::pow(static_cast<T>(base), exponent);
}

/**
 * @brief Computes power function element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @tparam Scalar Exponent type
 * @param base Base matrix
 * @param exponent Exponent
 * @return Matrix of powers
 */
template <typename Derived, typename Scalar>
auto pow(const Eigen::MatrixBase<Derived> &base, const Scalar &exponent) {
    return base.array().pow(exponent).matrix();
}

// --- Exp ---
/**
 * @brief Computes the exponential function e^x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return e raised to the power of x
 */
template <MetisScalar T> T exp(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::exp(x);
    } else {
        return exp(x);
    }
}

/**
 * @brief Computes exponential function element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of exponentials
 */
template <typename Derived> auto exp(const Eigen::MatrixBase<Derived> &x) {
    return x.array().exp().matrix();
}

// --- Log ---
/**
 * @brief Computes the natural logarithm of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Natural logarithm of x
 */
template <MetisScalar T> T log(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log(x);
    } else {
        return log(x);
    }
}

/**
 * @brief Computes natural logarithm element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of logarithms
 */
template <typename Derived> auto log(const Eigen::MatrixBase<Derived> &x) {
    return x.array().log().matrix();
}

/**
 * @brief Computes the base-10 logarithm of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Base-10 logarithm of x
 */
template <MetisScalar T> T log10(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log10(x);
    } else {
        return log10(x);
    }
}

/**
 * @brief Computes base-10 logarithm element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of base-10 logarithms
 */
template <typename Derived> auto log10(const Eigen::MatrixBase<Derived> &x) {
    return x.array().log10().matrix();
}

// --- Hyperbolic Functions ---
/**
 * @brief Computes hyperbolic sine
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Hyperbolic sine of x
 */
template <MetisScalar T> T sinh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sinh(x);
    } else {
        return sinh(x);
    }
}

/**
 * @brief Computes hyperbolic sine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of hyperbolic sines
 */
template <typename Derived> auto sinh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sinh().matrix();
}

/**
 * @brief Computes hyperbolic cosine of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Hyperbolic cosine of x
 */
template <MetisScalar T> T cosh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::cosh(x);
    } else {
        return cosh(x);
    }
}

/**
 * @brief Computes hyperbolic cosine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of hyperbolic cosines
 */
template <typename Derived> auto cosh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().cosh().matrix();
}

/**
 * @brief Computes hyperbolic tangent of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Hyperbolic tangent of x
 */
template <MetisScalar T> T tanh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::tanh(x);
    } else {
        return tanh(x);
    }
}

/**
 * @brief Computes hyperbolic tangent element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of hyperbolic tangents
 */
template <typename Derived> auto tanh(const Eigen::MatrixBase<Derived> &x) {
    return x.array().tanh().matrix();
}

// --- Rounding and Sign ---
/**
 * @brief Computes floor of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Floor of x
 */
template <MetisScalar T> T floor(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::floor(x);
    } else {
        return floor(x);
    }
}

/**
 * @brief Computes floor element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of floors
 */
template <typename Derived> auto floor(const Eigen::MatrixBase<Derived> &x) {
    return x.array().floor().matrix();
}

/**
 * @brief Computes ceiling of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Ceiling of x
 */
template <MetisScalar T> T ceil(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::ceil(x);
    } else {
        return ceil(x);
    }
}

/**
 * @brief Computes ceiling element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of ceilings
 */
template <typename Derived> auto ceil(const Eigen::MatrixBase<Derived> &x) {
    return x.array().ceil().matrix();
}

/**
 * @brief Computes sign of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return 1.0 if x > 0, -1.0 if x < 0, 0.0 otherwise
 */
template <MetisScalar T> T sign(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        // Return 1.0, -1.0, or 0.0
        return (x > 0) ? T(1.0) : ((x < 0) ? T(-1.0) : T(0.0));
    } else {
        return sign(x);
    }
}

/**
 * @brief Computes sign element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of signs
 */
template <typename Derived> auto sign(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sign().matrix();
}

// --- Modulo ---
/**
 * @brief Computes floating-point remainder of x/y
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Numerator
 * @param y Denominator
 * @return Remainder of x/y
 */
template <MetisScalar T> T fmod(const T &x, const T &y) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fmod(x, y);
    } else {
        return fmod(x, y);
    }
}

// Note: Ensure strictly positive modulus if needed, std::fmod matches C++ behavior.
// There is no direct .fmod() in Eigen Array, need to map
/**
 * @brief Computes floating-point remainder element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @tparam Scalar Scalar type
 * @param x Numerator matrix
 * @param y Denominator scalar
 * @return Matrix of remainders
 */
template <typename Derived, typename Scalar>
auto fmod(const Eigen::MatrixBase<Derived> &x, const Scalar &y) {
    if constexpr (std::is_same_v<typename Derived::Scalar, casadi::MX> ||
                  std::is_same_v<Scalar, casadi::MX>) {
        // Element-wise fmod when either operand is symbolic
        using ResultScalar = casadi::MX;
        Eigen::Matrix<ResultScalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> result(
            x.rows(), x.cols());
        const ResultScalar y_mx = static_cast<ResultScalar>(y);
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            for (Eigen::Index j = 0; j < x.cols(); ++j) {
                result(i, j) = metis::fmod(static_cast<ResultScalar>(x(i, j)), y_mx);
            }
        }
        return result;
    } else {
        return x.binaryExpr(Eigen::MatrixBase<Derived>::Constant(x.rows(), x.cols(), y),
                            [](double a, double b) { return std::fmod(a, b); });
    }
}

// --- Log2 ---
/**
 * @brief Computes base-2 logarithm of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Base-2 logarithm of x
 */
template <MetisScalar T> T log2(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log2(x);
    } else {
        // CasADi: log2(x) = log(x) / log(2)
        return log(x) / log(T(2.0));
    }
}

/**
 * @brief Computes base-2 logarithm element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of base-2 logarithms
 */
template <typename Derived> auto log2(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return metis::log2(v); });
    } else {
        return (x.array().log() / std::log(2.0)).matrix();
    }
}

// --- Exp2 ---
/**
 * @brief Computes 2^x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return 2 raised to the power x
 */
template <MetisScalar T> T exp2(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::exp2(x);
    } else {
        // CasADi: 2^x = exp(x * log(2))
        return exp(x * log(T(2.0)));
    }
}

/**
 * @brief Computes 2^x element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of 2^x values
 */
template <typename Derived> auto exp2(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return metis::exp2(v); });
    } else {
        return (x.array() * std::log(2.0)).exp().matrix();
    }
}

// --- Cbrt ---
/**
 * @brief Computes cube root of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Cube root of x
 */
template <MetisScalar T> T cbrt(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::cbrt(x);
    } else {
        // CasADi: cbrt(x) = sign(x) * pow(abs(x), 1/3)
        // This handles negative values correctly
        return sign(x) * pow(fabs(x), T(1.0 / 3.0));
    }
}

/**
 * @brief Computes cube root element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of cube roots
 */
template <typename Derived> auto cbrt(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return metis::cbrt(v); });
    } else {
        return x.unaryExpr([](double v) { return std::cbrt(v); });
    }
}

// --- Round ---
/**
 * @brief Rounds x to the nearest integer
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Nearest integer to x
 */
template <MetisScalar T> T round(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::round(x);
    } else {
        // CasADi: implement round as floor(x + 0.5) for positive,
        // ceil(x - 0.5) for negative (round half away from zero)
        return floor(x + T(0.5));
    }
}

/**
 * @brief Rounds element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of rounded values
 */
template <typename Derived> auto round(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return metis::round(v); });
    } else {
        return x.array().round().matrix();
    }
}

// --- Trunc ---
/**
 * @brief Truncates x toward zero
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return x truncated toward zero
 */
template <MetisScalar T> T trunc(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::trunc(x);
    } else {
        // CasADi: trunc(x) = sign(x) * floor(abs(x))
        return sign(x) * floor(fabs(x));
    }
}

/**
 * @brief Truncates element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of truncated values
 */
template <typename Derived> auto trunc(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return metis::trunc(v); });
    } else {
        return x.unaryExpr([](double v) { return std::trunc(v); });
    }
}

// --- Hypot ---
/**
 * @brief Computes sqrt(x^2 + y^2) without undue overflow/underflow
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x First value
 * @param y Second value
 * @return Hypotenuse length
 */
template <MetisScalar T> T hypot(const T &x, const T &y) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::hypot(x, y);
    } else {
        // CasADi: use sqrt(x^2 + y^2) - CasADi handles this symbolically
        return sqrt(x * x + y * y);
    }
}

/**
 * @brief Computes hypot(x, y) with mixed types
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x First value
 * @param y Second value (double)
 * @return Hypotenuse length
 */
template <MetisScalar T>
    requires(!std::is_same_v<T, double>)
T hypot(const T &x, double y) {
    return metis::hypot(x, T(y));
}

/**
 * @brief Computes hypot(x, y) with mixed types (double first)
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x First value (double)
 * @param y Second value
 * @return Hypotenuse length
 */
template <MetisScalar T>
    requires(!std::is_same_v<T, double>)
T hypot(double x, const T &y) {
    return metis::hypot(T(x), y);
}

/**
 * @brief Computes hypot element-wise for matrices
 * @tparam Derived Eigen matrix type
 * @param x First matrix
 * @param y Second matrix
 * @return Matrix of hypotenuse values
 */
template <typename Derived>
auto hypot(const Eigen::MatrixBase<Derived> &x, const Eigen::MatrixBase<Derived> &y) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return (x.array().square() + y.array().square()).sqrt().matrix();
    } else {
        return x.binaryExpr(y, [](double a, double b) { return std::hypot(a, b); });
    }
}

// --- Expm1 ---
/**
 * @brief Computes exp(x) - 1, accurate for small x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return exp(x) - 1
 */
template <MetisScalar T> T expm1(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::expm1(x);
    } else {
        // CasADi: fall back to exp(x) - 1
        return exp(x) - T(1.0);
    }
}

/**
 * @brief Computes expm1 element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of expm1 values
 */
template <typename Derived> auto expm1(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return metis::expm1(v); });
    } else {
        return x.unaryExpr([](double v) { return std::expm1(v); });
    }
}

// --- Log1p ---
/**
 * @brief Computes log(1 + x), accurate for small x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return log(1 + x)
 */
template <MetisScalar T> T log1p(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log1p(x);
    } else {
        // CasADi: fall back to log(1 + x)
        return log(T(1.0) + x);
    }
}

/**
 * @brief Computes log1p element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of log1p values
 */
template <typename Derived> auto log1p(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return metis::log1p(v); });
    } else {
        return x.unaryExpr([](double v) { return std::log1p(v); });
    }
}

// --- Copysign ---
/**
 * @brief Returns magnitude of x with sign of y
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Magnitude source
 * @param y Sign source
 * @return |x| with sign of y
 */
template <MetisScalar T> T copysign(const T &x, const T &y) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::copysign(x, y);
    } else {
        // CasADi: copysign(x, y) = sign(y) * abs(x)
        return sign(y) * fabs(x);
    }
}

/**
 * @brief Copysign with mixed types
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Magnitude source
 * @param y Sign source (double)
 * @return |x| with sign of y
 */
template <MetisScalar T>
    requires(!std::is_same_v<T, double>)
T copysign(const T &x, double y) {
    return metis::copysign(x, T(y));
}

/**
 * @brief Copysign with mixed types (double magnitude)
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Magnitude source (double)
 * @param y Sign source
 * @return |x| with sign of y
 */
template <MetisScalar T>
    requires(!std::is_same_v<T, double>)
T copysign(double x, const T &y) {
    return metis::copysign(T(x), y);
}

/**
 * @brief Copysign element-wise for matrices
 * @tparam Derived Eigen matrix type
 * @param x Magnitude matrix
 * @param y Sign matrix
 * @return Matrix with magnitudes from x and signs from y
 */
template <typename Derived>
auto copysign(const Eigen::MatrixBase<Derived> &x, const Eigen::MatrixBase<Derived> &y) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.binaryExpr(y,
                            [](const Scalar &a, const Scalar &b) { return metis::copysign(a, b); });
    } else {
        return x.binaryExpr(y, [](double a, double b) { return std::copysign(a, b); });
    }
}

// --- Square ---
/**
 * @brief Computes x^2 (more efficient than pow(x, 2))
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return x squared
 */
template <MetisScalar T> T square(const T &x) { return x * x; }

/**
 * @brief Computes square element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of squared values
 */
template <typename Derived> auto square(const Eigen::MatrixBase<Derived> &x) {
    return x.array().square().matrix();
}

} // namespace metis
