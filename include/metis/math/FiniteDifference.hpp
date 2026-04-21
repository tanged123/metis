#pragma once
/**
 * @file FiniteDifference.hpp
 * @brief Finite difference weights, derivative approximations, and integration defects
 * @see Calculus.hpp, AutoDiff.hpp
 */

#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>

namespace metis {

// ============================================================================
// Integration Method Enum
// ============================================================================

/**
 * @brief Integration/differentiation method for trajectory optimization
 */
enum class IntegrationMethod {
    ForwardEuler,  ///< First-order forward difference: x[i+1] - x[i] = xdot[i] * dt
    BackwardEuler, ///< First-order backward difference: x[i+1] - x[i] = xdot[i+1] * dt
    Trapezoidal,   ///< Second-order trapezoidal: x[i+1] - x[i] = 0.5 * (xdot[i] + xdot[i+1]) * dt
    Midpoint       ///< Alias for Trapezoidal (same formula)
};

/**
 * @brief Parse integration method from string
 *
 * Accepted names (case-sensitive):
 * - "trapezoidal", "trapezoid", "midpoint" -> IntegrationMethod::Trapezoidal
 * - "forward_euler", "forward euler"       -> IntegrationMethod::ForwardEuler
 * - "backward_euler", "backward euler",
 *   "backwards_euler", "backwards euler"   -> IntegrationMethod::BackwardEuler
 *
 * @note "midpoint" is treated as an alias for Trapezoidal (both use the same
 *       averaging formula).
 *
 * @param method Method name string
 * @return Corresponding IntegrationMethod enum value
 * @throws InvalidArgument if the method name is not recognized
 */
inline IntegrationMethod parse_integration_method(const std::string &method) {
    if (method == "trapezoidal" || method == "trapezoid" || method == "midpoint") {
        return IntegrationMethod::Trapezoidal;
    } else if (method == "forward_euler" || method == "forward euler") {
        return IntegrationMethod::ForwardEuler;
    } else if (method == "backward_euler" || method == "backward euler" ||
               method == "backwards_euler" || method == "backwards euler") {
        return IntegrationMethod::BackwardEuler;
    }
    throw InvalidArgument("Unknown integration method: " + method);
}

// ============================================================================
// Low-Order Finite Difference Weights
// ============================================================================

/**
 * @brief Forward Euler (explicit) differentiation weights
 *
 * For approximating df/dx at point i: df/dx ≈ (f[i+1] - f[i]) / h
 * Returns [weight_i, weight_i+1] = [-1/h, 1/h]
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param h Step size
 * @return Pair of weights [w_i, w_{i+1}]
 */
template <typename Scalar> std::pair<Scalar, Scalar> forward_euler_weights(Scalar h) {
    return {-Scalar(1) / h, Scalar(1) / h};
}

/**
 * @brief Backward Euler (implicit) differentiation weights
 *
 * For approximating df/dx at point i+1: df/dx ≈ (f[i+1] - f[i]) / h
 * Same formula as forward, but used at i+1 instead of i
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param h Step size
 * @return Pair of weights [w_i, w_{i+1}]
 */
template <typename Scalar> std::pair<Scalar, Scalar> backward_euler_weights(Scalar h) {
    return {-Scalar(1) / h, Scalar(1) / h};
}

/**
 * @brief Central difference differentiation weights
 *
 * For approximating df/dx at point i: df/dx ≈ (f[i+1] - f[i-1]) / (2h)
 * Returns [weight_{i-1}, weight_{i+1}] = [-1/(2h), 1/(2h)]
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param h Step size
 * @return Pair of weights [w_{i-1}, w_{i+1}]
 */
template <typename Scalar> std::pair<Scalar, Scalar> central_difference_weights(Scalar h) {
    return {-Scalar(1) / (Scalar(2) * h), Scalar(1) / (Scalar(2) * h)};
}

/**
 * @brief Trapezoidal integration weights
 *
 * For integrating: integral(f, x[i], x[i+1]) ≈ 0.5 * (f[i] + f[i+1]) * h
 * Returns [weight_i, weight_{i+1}] for the two points
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param h Step size
 * @return Pair of weights [w_i, w_{i+1}]
 */
template <typename Scalar> std::pair<Scalar, Scalar> trapezoidal_weights(Scalar h) {
    return {Scalar(0.5) * h, Scalar(0.5) * h};
}

// ============================================================================
// Derivative Approximation Functions
// ============================================================================

/**
 * @brief Compute derivative using forward difference
 *
 * Returns vector of size N-1 where out[i] ≈ df/dx at point i
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param f Function values (size N)
 * @param x Grid points (size N)
 * @return MetisVector<Scalar> derivative approximation (size N-1)
 */
template <typename Scalar>
MetisVector<Scalar> forward_difference(const MetisVector<Scalar> &f, const MetisVector<Scalar> &x) {
    Eigen::Index n = f.size();

    if (n != x.size()) {
        throw InvalidArgument("forward_difference: f and x must have same size");
    }
    if (n < 2) {
        throw InvalidArgument("forward_difference: need at least 2 points");
    }

    MetisVector<Scalar> df = f.tail(n - 1) - f.head(n - 1);
    MetisVector<Scalar> dx = x.tail(n - 1) - x.head(n - 1);

    return (df.array() / dx.array()).matrix();
}

/**
 * @brief Compute derivative using backward difference
 *
 * Returns vector of size N-1 where out[i] ≈ df/dx at point i+1
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param f Function values (size N)
 * @param x Grid points (size N)
 * @return MetisVector<Scalar> derivative approximation (size N-1)
 */
template <typename Scalar>
MetisVector<Scalar> backward_difference(const MetisVector<Scalar> &f,
                                        const MetisVector<Scalar> &x) {
    return forward_difference(f, x);
}

/**
 * @brief Compute derivative using central difference
 *
 * Returns vector of size N-2 where out[i] ≈ df/dx at point i+1
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param f Function values (size N)
 * @param x Grid points (size N)
 * @return MetisVector<Scalar> derivative approximation (size N-2)
 */
template <typename Scalar>
MetisVector<Scalar> central_difference(const MetisVector<Scalar> &f, const MetisVector<Scalar> &x) {
    Eigen::Index n = f.size();

    if (n != x.size()) {
        throw InvalidArgument("central_difference: f and x must have same size");
    }
    if (n < 3) {
        throw InvalidArgument("central_difference: need at least 3 points");
    }

    MetisVector<Scalar> df = f.tail(n - 2) - f.head(n - 2);
    MetisVector<Scalar> dx = x.tail(n - 2) - x.head(n - 2);

    return (df.array() / dx.array()).matrix();
}

/**
 * @brief Compute integration defects for derivative constraints
 *
 * Returns the "defects" that should be zero if the derivative relationship holds:
 *   defect[i] = (x[i+1] - x[i]) - integral(xdot) over [t[i], t[i+1]]
 *
 * These defects are used as equality constraints in trajectory optimization.
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param x Variable values (size N)
 * @param xdot Derivative values (size N)
 * @param t Time grid (size N)
 * @param method Integration method
 * @return MetisVector<Scalar> defects (size N-1), should be zero when constraint is satisfied
 */
template <typename Scalar>
MetisVector<Scalar> integration_defects(const MetisVector<Scalar> &x,
                                        const MetisVector<Scalar> &xdot,
                                        const MetisVector<Scalar> &t,
                                        IntegrationMethod method = IntegrationMethod::Trapezoidal) {
    Eigen::Index n = x.size();

    if (n != xdot.size() || n != t.size()) {
        throw InvalidArgument("integration_defects: x, xdot, t must have same size");
    }
    if (n < 2) {
        throw InvalidArgument("integration_defects: need at least 2 points");
    }

    MetisVector<Scalar> dx = x.tail(n - 1) - x.head(n - 1);
    MetisVector<Scalar> dt = t.tail(n - 1) - t.head(n - 1);
    MetisVector<Scalar> defects(n - 1);

    switch (method) {
    case IntegrationMethod::ForwardEuler:
        defects = dx - (xdot.head(n - 1).array() * dt.array()).matrix();
        break;

    case IntegrationMethod::BackwardEuler:
        defects = dx - (xdot.tail(n - 1).array() * dt.array()).matrix();
        break;

    case IntegrationMethod::Trapezoidal:
    case IntegrationMethod::Midpoint:
        defects =
            dx -
            (Scalar(0.5) * (xdot.head(n - 1) + xdot.tail(n - 1)).array() * dt.array()).matrix();
        break;
    }

    return defects;
}

/**
 * @brief Computes finite difference coefficients for arbitrary grids
 *
 * Based on Fornberg 1988: "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids"
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param x Grid points (MetisVector<Scalar>)
 * @param x0 Evaluation point
 * @param derivative_degree Order of derivative to approximate
 * @return MetisVector<Scalar> of coefficients matching x size
 */
template <typename Scalar>
MetisVector<Scalar> finite_difference_coefficients(const MetisVector<Scalar> &x,
                                                   Scalar x0 = Scalar(0.0),
                                                   int derivative_degree = 1) {
    if (derivative_degree < 0) {
        throw InvalidArgument("finite_difference_coefficients: derivative_degree must be >= 0");
    }

    int n_points = static_cast<int>(x.size());
    if (n_points < derivative_degree + 1) {
        throw InvalidArgument(
            "finite_difference_coefficients: need at least (derivative_degree + 1) grid points");
    }

    int N = n_points - 1;
    int M = derivative_degree;

    auto get_idx = [&](int m, int n, int v) { return m * (N + 1) * (N + 1) + n * (N + 1) + v; };

    std::vector<Scalar> delta((M + 1) * (N + 1) * (N + 1), Scalar(0.0));
    delta[get_idx(0, 0, 0)] = Scalar(1.0);

    Scalar c1 = Scalar(1.0);

    for (int n = 1; n <= N; ++n) {
        Scalar c2 = Scalar(1.0);
        for (int v = 0; v < n; ++v) {
            Scalar c3 = x(n) - x(v);
            c2 = c2 * c3;

            for (int m = 0; m <= std::min(n, M); ++m) {
                Scalar term1 = (x(n) - x0) * delta[get_idx(m, n - 1, v)];
                Scalar term2 = Scalar(0.0);
                if (m > 0) {
                    term2 = static_cast<Scalar>(m) * delta[get_idx(m - 1, n - 1, v)];
                }
                delta[get_idx(m, n, v)] = (term1 - term2) / c3;
            }
        }

        for (int m = 0; m <= std::min(n, M); ++m) {
            Scalar term1 = Scalar(0.0);
            if (m > 0) {
                term1 = static_cast<Scalar>(m) * delta[get_idx(m - 1, n - 1, n - 1)];
            }
            Scalar term2 = (x(n - 1) - x0) * delta[get_idx(m, n - 1, n - 1)];

            delta[get_idx(m, n, n)] = (c1 / c2) * (term1 - term2);
        }
        c1 = c2;
    }

    MetisVector<Scalar> coeffs(n_points);
    for (int i = 0; i < n_points; ++i) {
        coeffs(i) = delta[get_idx(M, N, i)];
    }

    return coeffs;
}

} // namespace metis
