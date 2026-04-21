#pragma once
/**
 * @file Trig.hpp
 * @brief Trigonometric and inverse trigonometric functions
 * @see Arithmetic.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace metis {

// --- Sin ---
/**
 * @brief Computes sine of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value (radians)
 * @return Sine of x
 */
template <MetisScalar T> T sin(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sin(x);
    } else {
        return sin(x);
    }
}

/**
 * @brief Computes sine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of sines
 */
template <typename Derived> auto sin(const Eigen::MatrixBase<Derived> &x) {
    return x.array().sin().matrix();
}

// --- Cos ---
/**
 * @brief Computes cosine of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value (radians)
 * @return Cosine of x
 */
template <MetisScalar T> T cos(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::cos(x);
    } else {
        return cos(x);
    }
}

/**
 * @brief Computes cosine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of cosines
 */
template <typename Derived> auto cos(const Eigen::MatrixBase<Derived> &x) {
    return x.array().cos().matrix();
}

// --- Tan ---
/**
 * @brief Computes tangent of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value (radians)
 * @return Tangent of x
 */
template <MetisScalar T> T tan(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::tan(x);
    } else {
        return tan(x);
    }
}

/**
 * @brief Computes tangent element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of tangents
 */
template <typename Derived> auto tan(const Eigen::MatrixBase<Derived> &x) {
    return x.array().tan().matrix();
}

// --- Arcsin ---
/**
 * @brief Computes arc sine of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Arc sine of x (radians)
 */
template <MetisScalar T> T asin(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::asin(x);
    } else {
        return asin(x);
    }
}

/**
 * @brief Computes arc sine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of arc sines
 */
template <typename Derived> auto asin(const Eigen::MatrixBase<Derived> &x) {
    return x.array().asin().matrix();
}

// --- Arccos ---
/**
 * @brief Computes arc cosine of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Arc cosine of x (radians)
 */
template <MetisScalar T> T acos(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::acos(x);
    } else {
        return acos(x);
    }
}

/**
 * @brief Computes arc cosine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of arc cosines
 */
template <typename Derived> auto acos(const Eigen::MatrixBase<Derived> &x) {
    return x.array().acos().matrix();
}

// --- Arctan ---
/**
 * @brief Computes arc tangent of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Arc tangent of x (radians)
 */
template <MetisScalar T> T atan(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::atan(x);
    } else {
        return atan(x);
    }
}

/**
 * @brief Computes arc tangent element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of arc tangents
 */
template <typename Derived> auto atan(const Eigen::MatrixBase<Derived> &x) {
    return x.array().atan().matrix();
}

// --- Atan2 ---
/**
 * @brief Computes arc tangent of y/x using signs of both arguments
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param y Numerator
 * @param x Denominator
 * @return Arc tangent of y/x (radians, included in [-pi, pi])
 */
template <MetisScalar T> T atan2(const T &y, const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::atan2(y, x);
    } else {
        return atan2(y, x);
    }
}

// --- Arcsinh ---
/**
 * @brief Computes inverse hyperbolic sine of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value
 * @return Inverse hyperbolic sine of x
 */
template <MetisScalar T> T asinh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::asinh(x);
    } else {
        return asinh(x);
    }
}

/**
 * @brief Computes inverse hyperbolic sine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of inverse hyperbolic sines
 */
template <typename Derived> auto asinh(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return asinh(v); });
    } else {
        return x.array().asinh().matrix();
    }
}

// --- Arccosh ---
/**
 * @brief Computes inverse hyperbolic cosine of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value (must be >= 1)
 * @return Inverse hyperbolic cosine of x
 */
template <MetisScalar T> T acosh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::acosh(x);
    } else {
        return acosh(x);
    }
}

/**
 * @brief Computes inverse hyperbolic cosine element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of inverse hyperbolic cosines
 */
template <typename Derived> auto acosh(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return acosh(v); });
    } else {
        return x.array().acosh().matrix();
    }
}

// --- Arctanh ---
/**
 * @brief Computes inverse hyperbolic tangent of x
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param x Input value (must be in (-1, 1))
 * @return Inverse hyperbolic tangent of x
 */
template <MetisScalar T> T atanh(const T &x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::atanh(x);
    } else {
        return atanh(x);
    }
}

/**
 * @brief Computes inverse hyperbolic tangent element-wise for a matrix
 * @tparam Derived Eigen matrix type
 * @param x Input matrix
 * @return Matrix of inverse hyperbolic tangents
 */
template <typename Derived> auto atanh(const Eigen::MatrixBase<Derived> &x) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, casadi::MX>) {
        return x.unaryExpr([](const Scalar &v) { return atanh(v); });
    } else {
        return x.array().atanh().matrix();
    }
}

} // namespace metis
