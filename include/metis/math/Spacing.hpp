#pragma once
/**
 * @file Spacing.hpp
 * @brief Point distribution generators (linspace, cosine, sine, log, geometric)
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Arithmetic.hpp"
#include "metis/math/Trig.hpp"
#include <Eigen/Dense>
#include <numbers>

namespace metis {

// --- linspace ---
/**
 * @brief Generates linearly spaced vector
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param start Start value
 * @param end End value
 * @param n Number of points
 * @return Vector of n linearly spaced points
 */
template <typename T> MetisVector<T> linspace(const T &start, const T &end, int n) {
    if (n < 1) {
        throw InvalidArgument("linspace: n must be >= 1");
    }
    if (n == 1) {
        MetisVector<T> ret(1);
        ret(0) = start;
        return ret;
    }

    MetisVector<T> ret(n);
    // Explicit loop to support symbolic types (cannot use .setLinSpaced)
    T step = (end - start) / static_cast<double>(n - 1);

    for (int i = 0; i < n; ++i) {
        ret(i) = start + static_cast<double>(i) * step;
    }
    // Ensure exact end point
    ret(n - 1) = end;
    return ret;
}

// --- cosine_spacing ---
/**
 * @brief Generates cosine spaced vector (denser at ends)
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param start Start value
 * @param end End value
 * @param n Number of points
 * @return Vector of n cosine-spaced points
 */
template <typename T> MetisVector<T> cosine_spacing(const T &start, const T &end, int n) {
    if (n < 1) {
        throw InvalidArgument("cosine_spacing: n must be >= 1");
    }
    if (n == 1) {
        MetisVector<T> ret(1);
        ret(0) = start;
        return ret;
    }

    MetisVector<T> ret(n);
    T center = 0.5 * (start + end);
    T radius = 0.5 * (end - start);
    double pi = std::numbers::pi_v<double>;

    for (int i = 0; i < n; ++i) {
        double angle = pi * static_cast<double>(i) / static_cast<double>(n - 1);
        ret(i) = center - radius * std::cos(angle);
    }
    return ret;
}

// --- sinspace ---
/**
 * @brief Generates sine spaced vector (denser at start by default)
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param start Start value
 * @param end End value
 * @param n Number of points
 * @param reverse_spacing If true, bunches points at the end instead of start
 * @return Vector of n sine-spaced points
 */
template <typename T>
MetisVector<T> sinspace(const T &start, const T &end, int n, bool reverse_spacing = false) {
    if (n < 1) {
        throw InvalidArgument("sinspace: n must be >= 1");
    }
    if (n == 1) {
        MetisVector<T> ret(1);
        ret(0) = start;
        return ret;
    }

    if (reverse_spacing) {
        auto ret = sinspace(end, start, n, false);
        return ret.reverse();
    }

    MetisVector<T> ret(n);
    double pi_half = std::numbers::pi_v<double> / 2.0;

    for (int i = 0; i < n; ++i) {
        // angle goes from 0 to pi/2
        double angle = pi_half * static_cast<double>(i) / static_cast<double>(n - 1);
        double factor = 1.0 - std::cos(angle); // 0 to 1
        ret(i) = start + (end - start) * factor;
    }
    // Ensure exact endpoints
    ret(0) = start;   // cos(0) = 1, factor=0
    ret(n - 1) = end; // cos(pi/2) = 0, factor=1
    return ret;
}

// --- logspace ---
/**
 * @brief Generates logarithmically spaced vector (base 10)
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param start Start exponent (10^start)
 * @param end End exponent (10^end)
 * @param n Number of points
 * @return Vector of n log-spaced points
 */
template <typename T> MetisVector<T> logspace(const T &start, const T &end, int n) {
    if (n < 1) {
        throw InvalidArgument("logspace: n must be >= 1");
    }
    if (n == 1) {
        MetisVector<T> ret(1);
        ret(0) = metis::pow(10.0, start);
        return ret;
    }

    MetisVector<T> ret(n);
    // Linear spacing in exponent
    for (int i = 0; i < n; ++i) {
        double fraction = static_cast<double>(i) / static_cast<double>(n - 1);
        T exponents = start + (end - start) * fraction;
        ret(i) = metis::pow(10.0, exponents);
    }
    // ret(0) is 10^start, ret(n-1) is 10^end
    return ret;
}

// --- geomspace ---
/**
 * @brief Generates geometrically spaced vector
 * @tparam T Scalar type (NumericScalar or SymbolicScalar)
 * @param start Start value
 * @param end End value
 * @param n Number of points
 * @return Vector of n geometrically spaced points
 */
template <typename T> MetisVector<T> geomspace(const T &start, const T &end, int n) {
    if (n < 1) {
        throw InvalidArgument("geomspace: n must be >= 1");
    }
    T log_start = metis::log10(start);
    T log_end = metis::log10(end);
    return metis::logspace(log_start, log_end, n);
}

} // namespace metis
