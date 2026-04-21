#pragma once
/**
 * @file Calculus.hpp
 * @brief Numerical differentiation and integration (gradient, trapz, cumtrapz, diff)
 * @see FiniteDifference.hpp, IntegrateDiscrete.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/math/Arithmetic.hpp"
#include <Eigen/Dense>
#include <optional>
#include <type_traits>
#include <utility>

namespace metis {

// --- diff(vector) ---
/**
 * @brief Computes adjacent differences of a vector
 * Returns a vector of size N-1 where out[i] = v[i+1] - v[i]
 *
 * @tparam Derived Eigen matrix type
 * @param v Input vector
 * @return Difference vector
 */
template <typename Derived> auto diff(const Eigen::MatrixBase<Derived> &v) {
    // Return expression template for efficiency
    return (v.tail(v.size() - 1) - v.head(v.size() - 1));
}

// --- trapz(y, x) ---
/**
 * @brief Computes trapezoidal integration
 * Approximation of integral of y(x) using trapezoidal rule
 *
 * @tparam DerivedY Eigen matrix type for Y
 * @tparam DerivedX Eigen matrix type for X
 * @param y Values of function at x points
 * @param x Grid points
 * @return Integrated value
 */
template <typename DerivedY, typename DerivedX>
auto trapz(const Eigen::MatrixBase<DerivedY> &y, const Eigen::MatrixBase<DerivedX> &x) {
    auto dx = (x.tail(x.size() - 1) - x.head(x.size() - 1));
    auto mean_y = 0.5 * (y.tail(y.size() - 1) + y.head(y.size() - 1));

    // Sum of element-wise product
    return (mean_y.array() * dx.array()).sum();
}

// --- cumtrapz(y, x) ---
/**
 * @brief Computes the cumulative trapezoidal integral.
 *
 * Returns a running integral with the same length as the inputs:
 * out(0) = 0 and out(i) = integral from x(0) to x(i) using trapezoids.
 *
 * @tparam DerivedY Eigen vector type for Y
 * @tparam DerivedX Eigen vector type for X
 * @param y Values of function at x points
 * @param x Grid points
 * @return Vector of cumulative trapezoidal integrals
 */
template <typename DerivedY, typename DerivedX>
auto cumtrapz(const Eigen::MatrixBase<DerivedY> &y, const Eigen::MatrixBase<DerivedX> &x) {
    using Scalar = std::decay_t<decltype(0.5 *
                                         (std::declval<typename DerivedY::Scalar>() +
                                          std::declval<typename DerivedY::Scalar>()) *
                                         (std::declval<typename DerivedX::Scalar>() -
                                          std::declval<typename DerivedX::Scalar>()))>;
    MetisVector<Scalar> result(y.size());

    if (y.size() != x.size()) {
        throw InvalidArgument("cumtrapz: y and x must have the same size");
    }

    if (y.size() == 0) {
        return result;
    }

    result(0) = Scalar(0.0);
    for (Eigen::Index i = 1; i < y.size(); ++i) {
        const auto interval = 0.5 * (y(i - 1) + y(i)) * (x(i) - x(i - 1));
        result(i) = result(i - 1) + interval;
    }

    return result;
}

/**
 * @brief Computes the cumulative trapezoidal integral with uniform spacing.
 *
 * Returns a running integral with the same length as the input:
 * out(0) = 0 and out(i) = integral over the first i intervals.
 *
 * @tparam DerivedY Eigen vector type for Y
 * @tparam Spacing Scalar spacing type
 * @param y Values of function samples
 * @param dx Uniform spacing between samples
 * @return Vector of cumulative trapezoidal integrals
 */
template <typename DerivedY, MetisScalar Spacing = double>
auto cumtrapz(const Eigen::MatrixBase<DerivedY> &y, const Spacing &dx = 1.0) {
    using Scalar = std::decay_t<decltype(0.5 *
                                         (std::declval<typename DerivedY::Scalar>() +
                                          std::declval<typename DerivedY::Scalar>()) *
                                         std::declval<Spacing>())>;
    MetisVector<Scalar> result(y.size());

    if (y.size() == 0) {
        return result;
    }

    result(0) = Scalar(0.0);
    for (Eigen::Index i = 1; i < y.size(); ++i) {
        const auto interval = 0.5 * (y(i - 1) + y(i)) * dx;
        result(i) = result(i - 1) + interval;
    }

    return result;
}

// --- gradient_1d(y, x) ---
/**
 * @brief Computes gradient of 1D data using central differences
 *
 * Simple gradient with first-order boundaries. For more control over
 * boundary accuracy and derivative order, use gradient() instead.
 *
 * Returns vector of same size as inputs.
 * Uses forward/backward difference at boundaries.
 *
 * @param y Values
 * @param x Grid points
 * @return Gradient vector
 */
template <typename DerivedY, typename DerivedX>
auto gradient_1d(const Eigen::MatrixBase<DerivedY> &y, const Eigen::MatrixBase<DerivedX> &x) {
    Eigen::Index n = y.size();
    typename DerivedY::PlainObject grad(n);

    if (n < 2) {
        if (n > 0)
            grad.setZero();
        return grad;
    }

    // Interior points: (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    if (n > 2) {
        grad.segment(1, n - 2) =
            (y.tail(n - 2) - y.head(n - 2)).array() / (x.tail(n - 2) - x.head(n - 2)).array();
    }

    // Boundaries (Forward/Backward difference)
    // Forward at 0
    grad(0) = (y(1) - y(0)) / (x(1) - x(0));
    // Backward at n-1
    grad(n - 1) = (y(n - 1) - y(n - 2)) / (x(n - 1) - x(n - 2));

    return grad;
}

/**
 * @brief Computes gradient using second-order accurate central differences
 *
 * Returns the gradient of a 1D array using second-order accurate central differences
 * in the interior and first or second-order accurate one-sided differences at boundaries.
 *
 * @tparam Derived Eigen vector type
 * @param f Function values (vector)
 * @param dx Spacing (scalar or vector). If scalar, uniform spacing is assumed.
 * @param edge_order Order of accuracy at boundaries (1 or 2)
 * @param n Derivative order (1 for first derivative, 2 for second derivative)
 * @return Gradient vector (same size as f)
 */
template <typename DerivedF, typename Spacing = double>
auto gradient(const Eigen::MatrixBase<DerivedF> &f, const Spacing &dx = 1.0, int edge_order = 1,
              int n = 1) {
    using Scalar = typename DerivedF::Scalar;
    using Vector = MetisVector<Scalar>;

    Eigen::Index N = f.size();
    Vector grad(N);

    if (N < 2) {
        if (N > 0)
            grad.setZero();
        return grad;
    }

    // Handle spacing - convert scalar to vector if needed
    Vector dx_vec(N - 1);
    if constexpr (std::is_arithmetic_v<Spacing>) {
        dx_vec.setConstant(dx);
    } else {
        // dx is already a vector - compute differences
        if (dx.size() == N) {
            // dx are grid points, compute spacing
            dx_vec = (dx.tail(N - 1) - dx.head(N - 1));
        } else if (dx.size() == N - 1) {
            // dx are already spacings
            dx_vec = dx;
        } else {
            throw InvalidArgument("gradient: dx must be scalar, size N, or size N-1");
        }
    }

    if (N == 2) {
        // Special case: only 2 points
        grad(0) = (f(1) - f(0)) / dx_vec(0);
        grad(1) = grad(0);
        return grad;
    }

    // Extract spacing arrays for interior calculation
    // hm[i] = spacing before point i+1
    // hp[i] = spacing after point i+1
    Vector hm = dx_vec.head(N - 2); // dx[0] to dx[N-3]
    Vector hp = dx_vec.tail(N - 2); // dx[1] to dx[N-2]

    // Extract function differences
    // dfp[i] = f[i+2] - f[i+1]
    // dfm[i] = f[i+1] - f[i]
    Vector dfp = f.tail(N - 2) - f.segment(1, N - 2); // f[2:] - f[1:-1]
    Vector dfm = f.segment(1, N - 2) - f.head(N - 2); // f[1:-1] - f[:-2]

    if (n == 1) {
        // First derivative using second-order accurate formula
        // Interior points: weighted average based on spacing
        grad.segment(1, N - 2) =
            (hm.array().square() * dfp.array() + hp.array().square() * dfm.array()) /
            (hm.array() * hp.array() * (hm.array() + hp.array()));

        if (edge_order == 1) {
            // First-order accurate boundaries (forward/backward difference)
            grad(0) = dfm(0) / hm(0);
            grad(N - 1) = dfp(N - 3) / hp(N - 3);

        } else if (edge_order == 2) {
            // Second-order accurate boundaries
            // First point
            Scalar dfm_0 = dfm(0);
            Scalar dfp_0 = dfp(0);
            Scalar hm_0 = hm(0);
            Scalar hp_0 = hp(0);
            grad(0) = (2.0 * dfm_0 * hm_0 * hp_0 + dfm_0 * hp_0 * hp_0 - dfp_0 * hm_0 * hm_0) /
                      (hm_0 * hp_0 * (hm_0 + hp_0));

            // Last point
            Scalar dfm_N = dfm(N - 3);
            Scalar dfp_N = dfp(N - 3);
            Scalar hm_N = hm(N - 3);
            Scalar hp_N = hp(N - 3);
            grad(N - 1) = (-dfm_N * hp_N * hp_N + dfp_N * hm_N * hm_N + 2.0 * dfp_N * hm_N * hp_N) /
                          (hm_N * hp_N * (hm_N + hp_N));
        } else {
            throw InvalidArgument("gradient: edge_order must be 1 or 2");
        }

    } else if (n == 2) {
        // Second derivative
        grad.segment(1, N - 2) =
            2.0 / (hm.array() + hp.array()) * (dfp.array() / hp.array() - dfm.array() / hm.array());

        // For second derivative, replicate edge values
        grad(0) = grad(1);
        grad(N - 1) = grad(N - 2);

    } else {
        throw InvalidArgument("gradient: derivative order (n) must be 1 or 2");
    }

    return grad;
}

/**
 * @brief Computes gradient with periodic boundary conditions
 *
 * For data defined on a periodic domain, this computes the gradient assuming
 * the final sample wraps around to the first sample. Samples should span one
 * cycle without repeating the first point at the end; the wrap interval is
 * inferred from `period - covered_span`.
 *
 * @tparam DerivedF Eigen vector type
 * @tparam Spacing Spacing type (scalar or vector)
 * @param f Function values (vector)
 * @param dx Spacing
 * @param period Period of the data (e.g., 360 for degrees)
 * @param edge_order Order of accuracy at boundaries (1 or 2)
 * @param n Derivative order (1 or 2)
 * @return Gradient vector
 */
template <typename DerivedF, typename Spacing = double, typename Period>
auto gradient_periodic(const Eigen::MatrixBase<DerivedF> &f, const Spacing &dx,
                       const Period &period, int edge_order = 1, int n = 1) {
    using Scalar = typename DerivedF::Scalar;
    using Vector = MetisVector<Scalar>;

    const Eigen::Index N = f.size();
    Vector grad(N);

    if (edge_order != 1 && edge_order != 2) {
        throw InvalidArgument("gradient_periodic: edge_order must be 1 or 2");
    }
    if (n != 1 && n != 2) {
        throw InvalidArgument("gradient_periodic: derivative order (n) must be 1 or 2");
    }

    if (N < 2) {
        if (N > 0)
            grad.setZero();
        return grad;
    }

    Eigen::VectorXd dx_vec(N - 1);
    if constexpr (std::is_arithmetic_v<Spacing>) {
        dx_vec.setConstant(static_cast<double>(dx));
    } else {
        if (dx.size() == N) {
            for (Eigen::Index i = 0; i < N - 1; ++i) {
                dx_vec(i) = static_cast<double>(dx(i + 1) - dx(i));
            }
        } else if (dx.size() == N - 1) {
            for (Eigen::Index i = 0; i < N - 1; ++i) {
                dx_vec(i) = static_cast<double>(dx(i));
            }
        } else {
            throw InvalidArgument("gradient_periodic: dx must be scalar, size N, or size N-1");
        }
    }

    if (N == 2) {
        if (n == 1) {
            grad(0) = (f(1) - f(0)) / dx_vec(0);
            grad(1) = grad(0);
        } else {
            grad.setZero();
        }
        return grad;
    }

    const double wrap_dx = static_cast<double>(period) - dx_vec.sum();
    if (wrap_dx <= 0.0) {
        throw InvalidArgument("gradient_periodic: period must exceed the covered span. Exclude "
                              "duplicate endpoint samples.");
    }

    for (Eigen::Index i = 0; i < N; ++i) {
        const Eigen::Index prev = (i == 0) ? N - 1 : i - 1;
        const Eigen::Index next = (i == N - 1) ? 0 : i + 1;

        const double hm = (i == 0) ? wrap_dx : dx_vec(i - 1);
        const double hp = (i == N - 1) ? wrap_dx : dx_vec(i);

        const auto dfm = f(i) - f(prev);
        const auto dfp = f(next) - f(i);

        if (n == 1) {
            grad(i) = (hm * hm * dfp + hp * hp * dfm) / (hm * hp * (hm + hp));
        } else {
            grad(i) = 2.0 / (hm + hp) * (dfp / hp - dfm / hm);
        }
    }

    return grad;
}

} // namespace metis
