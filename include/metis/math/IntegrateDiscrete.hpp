#pragma once
/**
 * @file IntegrateDiscrete.hpp
 * @brief Discrete integration and squared-curvature computation from sampled data
 * @see Calculus.hpp, Integrate.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Arithmetic.hpp"
#include "metis/math/Calculus.hpp"
#include "metis/math/Logic.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <string>

namespace metis {

namespace detail {

/// Small epsilon to prevent division by zero in RMS fusion
constexpr double rms_fusion_epsilon = 1e-100;

// Helper to compute dx
template <typename DerivedX> auto compute_dx(const Eigen::MatrixBase<DerivedX> &x) {
    return (x.tail(x.size() - 1) - x.head(x.size() - 1)).eval();
}

// Slice helpers to avoid verbose .segment calls
template <typename Derived>
auto slice(const Eigen::MatrixBase<Derived> &m, Eigen::Index start, Eigen::Index end) {
    // Python style slice [start:end] (end exclusive)
    // Eigen block takes (start, size)
    // Handle negative indices python-style
    Eigen::Index n = m.size();
    Eigen::Index s = (start < 0) ? (n + start) : start;
    Eigen::Index e = (end < 0) ? (n + end) : end;
    if (e == 0 && end < 0)
        e = n + end; // Handle case like [:-1] -> 0 if n=1? check logic
    // Actually slice(0, -1) means 0 to n-1. Size n-1.
    // If indices are out of bounds, clamp or let Eigen assert.

    // Safety check for empty ranges
    if (e <= s)
        return m.head(0).eval();

    return m.segment(s, e - s).eval();
}

template <typename DerivedA, typename DerivedB>
auto concatenate(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    // Assume vectors
    MetisVector<Scalar> res(a.size() + b.size());
    res.head(a.size()) = a;
    res.tail(b.size()) = b;
    return res;
}

// --- Integration Methods ---

template <typename DerivedF, typename DerivedX>
auto integrate_forward_simpson(const Eigen::MatrixBase<DerivedF> &f,
                               const Eigen::MatrixBase<DerivedX> &x) {
    auto x1 = slice(x, 0, -2);
    auto x2 = slice(x, 1, -1);
    auto x3 = slice(x, 2, x.size());

    auto f1 = slice(f, 0, -2);
    auto f2 = slice(f, 1, -1);
    auto f3 = slice(f, 2, f.size());

    // h = x2 - x1, hp = x3 - x2
    auto h = (x2 - x1).eval();
    auto hp = (x3 - x2).eval();

    // q3 = 1 + hp / h
    auto q3 = (1.0 + hp.array() / h.array()).eval();

    // term = (f1 - f3 + 3*q3^2*(f1+f2) - 2*q3*(2*f1+f2)) / (6*q3*(q3-1))
    // Use arrays for element-wise
    auto num = f1.array() - f3.array() + 3.0 * q3.square() * (f1.array() + f2.array()) -
               2.0 * q3 * (2.0 * f1.array() + f2.array());
    auto den = 6.0 * q3 * (q3 - 1.0);

    return (num / den).matrix().eval();
}

template <typename DerivedF, typename DerivedX>
auto integrate_backward_simpson(const Eigen::MatrixBase<DerivedF> &f,
                                const Eigen::MatrixBase<DerivedX> &x) {
    auto x1 = slice(x, 0, -2);
    auto x2 = slice(x, 1, -1);
    auto x3 = slice(x, 2, x.size());

    auto f1 = slice(f, 0, -2);
    auto f2 = slice(f, 1, -1);
    auto f3 = slice(f, 2, f.size());

    auto h = (x3 - x2).eval();
    auto hm = (x2 - x1).eval();

    // q1 = -hm / h
    auto q1 = (-hm.array() / h.array()).eval();

    // term = (f2 - f1 + 3*q1^2*(f2+f3) - 2*q1*(2*f2+f3)) / (6*q1*(q1-1))
    auto num = f2.array() - f1.array() + 3.0 * q1.square() * (f2.array() + f3.array()) -
               2.0 * q1 * (2.0 * f2.array() + f3.array());
    auto den = 6.0 * q1 * (q1 - 1.0);

    return (num / den).matrix().eval();
}

template <typename DerivedF, typename DerivedX>
auto integrate_cubic(const Eigen::MatrixBase<DerivedF> &f, const Eigen::MatrixBase<DerivedX> &x) {
    // Uses 4 points to compute integral over interval [x2, x3]
    // Based on Lagrange polynomial through all 4 points
    auto x1 = slice(x, 0, -3);
    auto x2 = slice(x, 1, -2);
    auto x3 = slice(x, 2, -1);
    auto x4 = slice(x, 3, x.size());

    auto f1 = slice(f, 0, -3);
    auto f2 = slice(f, 1, -2);
    auto f3 = slice(f, 2, -1);
    auto f4 = slice(f, 3, f.size());

    auto h = (x3 - x2).eval();
    auto hm = (x2 - x1).eval();
    auto hp = (x4 - x3).eval();

    // q1 = -hm/h, q4 = 1 + hp/h
    auto q1 = (-hm.array() / h.array()).eval();
    auto q4 = (1.0 + hp.array() / h.array()).eval();

    // Cubic formula from AeroSandbox reference
    // avg_f = (
    //   6*q1^3*q4^2*(f2+f3) - 4*q1^3*q4*(2*f2+f3) + 2*q1^3*(f2-f4)
    // - 6*q1^2*q4^3*(f2+f3) + 3*q1^2*q4*(3*f2+f3) + 3*q1^2*(f4-f2)
    // + 4*q1*q4^3*(2*f2+f3) - 3*q1*q4^2*(3*f2+f3) + q1*(f2-f4)
    // + 2*q4^3*(f1-f2) + 3*q4^2*(f2-f1) + q4*(f1-f2)
    // ) / (12*q1*q4*(q1-1)*(q1-q4)*(q4-1))

    auto q1_2 = q1.square();
    auto q1_3 = q1_2 * q1;
    auto q4_2 = q4.square();
    auto q4_3 = q4_2 * q4;

    auto f2_arr = f2.array();
    auto f3_arr = f3.array();
    auto f1_arr = f1.array();
    auto f4_arr = f4.array();

    auto num = 6.0 * q1_3 * q4_2 * (f2_arr + f3_arr) - 4.0 * q1_3 * q4 * (2.0 * f2_arr + f3_arr) +
               2.0 * q1_3 * (f2_arr - f4_arr) - 6.0 * q1_2 * q4_3 * (f2_arr + f3_arr) +
               3.0 * q1_2 * q4 * (3.0 * f2_arr + f3_arr) + 3.0 * q1_2 * (f4_arr - f2_arr) +
               4.0 * q1 * q4_3 * (2.0 * f2_arr + f3_arr) -
               3.0 * q1 * q4_2 * (3.0 * f2_arr + f3_arr) + q1 * (f2_arr - f4_arr) +
               2.0 * q4_3 * (f1_arr - f2_arr) + 3.0 * q4_2 * (f2_arr - f1_arr) +
               q4 * (f1_arr - f2_arr);

    auto den = 12.0 * q1 * q4 * (q1 - 1.0) * (q1 - q4) * (q4 - 1.0);

    return (num / den).matrix().eval();
}

} // namespace detail

/**
 * @brief Integrates discrete samples using reconstruction methods
 *
 * @tparam DerivedF Eigen type for function values
 * @tparam DerivedX Eigen type for grid points
 * @param f Function values
 * @param x Grid points
 * @param multiply_by_dx If true, returns interval integrals; if false, returns average values
 * @param method "forward_euler", "backward_euler", "trapezoidal", "simpson", or "cubic"
 * @param method_endpoints "lower_order" or "ignore"
 * @return Vector of interval integrals or averages
 */
template <typename DerivedF, typename DerivedX>
MetisVector<typename DerivedF::Scalar>
integrate_discrete_intervals(const Eigen::MatrixBase<DerivedF> &f,
                             const Eigen::MatrixBase<DerivedX> &x, bool multiply_by_dx = true,
                             const std::string &method = "trapezoidal",
                             const std::string &method_endpoints = "lower_order") {
    using Scalar = typename DerivedF::Scalar;

    Eigen::Index n_points = f.size();
    if (n_points < 2) {
        // Return empty vector
        return MetisVector<Scalar>(0);
    }

    auto dx = detail::compute_dx(x);
    MetisVector<Scalar> avg_f;

    // Normalize method string
    std::string m = method;
    std::transform(m.begin(), m.end(), m.begin(), ::tolower);
    // Replace spaces with underscores if needed loops? Simple find/replace for now.
    std::replace(m.begin(), m.end(), ' ', '_');

    if (m == "forward_euler" || m == "forward" || m == "euler_forward") {
        avg_f = detail::slice(f, 0, -1);
    } else if (m == "backward_euler" || m == "backward" || m == "euler_backward") {
        avg_f = detail::slice(f, 1, n_points); // slice(1, n)
    } else if (m == "trapezoidal" || m == "trapezoid" || m == "trapz") {
        auto f_left = detail::slice(f, 0, -1);
        auto f_right = detail::slice(f, 1, n_points);
        avg_f = (f_left + f_right) * 0.5;
    } else if (m == "forward_simpson" || m == "simpson_forward") {
        avg_f = detail::integrate_forward_simpson(f, x);
        // Degree 2, leaves (0, 1) intervals remaining?
        // Forward simpson uses [0, 1, 2] to compute interval [0, 1].
        // So given N points, we get integrals for intervals 0..N-2.
        // We miss the last interval (N-2 to N-1).
        // remaining_endpoint_intervals = (0, 1)
        // Handled by endpoint logic below
    } else if (m == "backward_simpson" || m == "simpson_backward") {
        avg_f = detail::integrate_backward_simpson(f, x);
        // remaining_endpoint_intervals = (1, 0)
    } else if (m == "simpson") {
        // Fused simpson: combines forward and backward with RMS fusion
        if (n_points < 3) {
            return integrate_discrete_intervals(f, x, multiply_by_dx, "trapezoidal",
                                                method_endpoints);
        }
        auto res_fwd = detail::integrate_forward_simpson(f, x);
        auto res_bwd = detail::integrate_backward_simpson(f, x);

        auto first = detail::slice(res_fwd, 0, 1);
        auto last = detail::slice(res_bwd, -1, res_bwd.size());

        auto a = detail::slice(res_bwd, 0, -1);
        auto b = detail::slice(res_fwd, 1, res_fwd.size());

        // RMS fusion for middle intervals
        auto mid_sq = (a.array().square() + b.array().square()) * 0.5 + detail::rms_fusion_epsilon;
        MetisVector<Scalar> mid = metis::sqrt(mid_sq.matrix());

        auto tmp = detail::concatenate(first, mid);
        avg_f = detail::concatenate(tmp, last);
    } else if (m.find("simpson") != std::string::npos && m != "simpson") {
        // Generic simpson check, but specific variants matched above
        throw IntegrationError("unknown Simpson variant: " + method);
    } else if (m == "cubic" || m == "cubic_spline") {
        if (n_points < 4) {
            // Need at least 4 points for cubic, fallback to simpson
            return integrate_discrete_intervals(f, x, multiply_by_dx, "simpson", method_endpoints);
        }
        avg_f = detail::integrate_cubic(f, x);
        // remaining_endpoint_intervals = (1, 1) - misses first and last interval
    } else if (m != "forward_euler" && m != "backward_euler" && m != "trapezoidal" &&
               m != "forward" && m != "backward" && m != "euler_forward" && m != "euler_backward" &&
               m != "trapz" && m != "trapezoid") {
        throw IntegrationError(
            "unknown method: " + method +
            ". Use forward_euler, backward_euler, trapezoidal, simpson, or cubic.");
    }

    // --- Endpoint Handling ---

    int remaining_start = 0;
    int remaining_end = 0;
    int degree = 1;

    if (m.find("forward_simpson") != std::string::npos) {
        degree = 2;
        remaining_end = 1;
    } else if (m.find("backward_simpson") != std::string::npos) {
        degree = 2;
        remaining_start = 1;
    } else if (m.find("cubic") != std::string::npos) {
        degree = 3;
        remaining_start = 1;
        remaining_end = 1;
    } else if (m.find("euler") != std::string::npos) {
        degree = 0;
    }

    std::string eff_endpoints = method_endpoints;

    if (eff_endpoints == "lower_order") {
        // Fallback strategy
        std::string fallback_method = "trapezoidal";
        if (degree >= 3)
            fallback_method = "simpson"; // simplifies to simps recursion
        // For Simpson, fallback is trapezoidal?
        // Reference: if degree >= 3 endpoints="simpson". if endpoints="simpson", recurse with
        // forward/backward simpson.

        // Let's implement simpler: directly recurse with lower order for missing parts.

        if (remaining_start > 0) {
            // Need to fill beginning
            // Taking f[: 1 + remaining_start]?
            // Need enough points for trapezoidal (2 points).
            // slice(f, 0, 1 + remaining_start)
            // Call recursively with ignore endpoints
            auto left_res = integrate_discrete_intervals(
                detail::slice(f, 0, 1 + remaining_start), detail::slice(x, 0, 1 + remaining_start),
                false,         // Don't mul by dx yet
                "trapezoidal", // Using trapezoidal as universal fallback
                "ignore");
            // Concatenate
            avg_f = detail::concatenate(left_res, avg_f);
        }

        if (remaining_end > 0) {
            auto right_res = integrate_discrete_intervals(
                detail::slice(f, -(1 + remaining_end), x.size()),
                detail::slice(x, -(1 + remaining_end), x.size()), false, "trapezoidal", "ignore");
            avg_f = detail::concatenate(avg_f, right_res);
        }

    } else if (eff_endpoints == "ignore") {
        // Do nothing, but need to slice dx if we return integrals
        // dx is computed for full range?
        // avg_f is smaller.
    }

    // dx slicing if endpoints ignored
    Eigen::Index current_size = avg_f.size();
    if (current_size < dx.size()) {
        // Figure out offset based on remaining
        // If remaining_start=1, dx starts at 1.
        // But "ignore" might not tell us which side unless we know method.
        // Simpson forward: missing end. dx should be head.
        if (m.find("forward_simpson") != std::string::npos) {
            dx = detail::slice(dx, 0, current_size);
        } else if (m.find("backward_simpson") != std::string::npos) {
            dx = detail::slice(dx, dx.size() - current_size, dx.size());
        }
    }

    if (multiply_by_dx) {
        return (avg_f.array() * dx.array()).matrix();
    } else {
        return avg_f;
    }
}

/**
 * @brief Computes the integral of squared curvature over each interval
 *
 * Useful for regularization of smooth curves in optimization (penalizes high curvature).
 *
 * @tparam DerivedF Eigen type for function values
 * @tparam DerivedX Eigen type for grid points
 * @param f Function values
 * @param x Grid points
 * @param method "simpson" (default) or "hybrid_simpson_cubic"
 * @return Vector of squared-curvature integrals per interval
 */
template <typename DerivedF, typename DerivedX>
MetisVector<typename DerivedF::Scalar>
integrate_discrete_squared_curvature(const Eigen::MatrixBase<DerivedF> &f,
                                     const Eigen::MatrixBase<DerivedX> &x,
                                     const std::string &method = "simpson") {
    using Scalar = typename DerivedF::Scalar;

    Eigen::Index n_points = f.size();
    if (n_points < 3) {
        return MetisVector<Scalar>(0);
    }

    std::string m = method;
    std::transform(m.begin(), m.end(), m.begin(), ::tolower);
    std::replace(m.begin(), m.end(), ' ', '_');

    if (m == "simpson") {
        // Forward Simpson: uses points [i, i+1, i+2] for interval [i, i+1]
        auto x2 = detail::slice(x, 0, -2);
        auto x3 = detail::slice(x, 1, -1);
        auto x4 = detail::slice(x, 2, x.size());

        auto f2 = detail::slice(f, 0, -2);
        auto f3 = detail::slice(f, 1, -1);
        auto f4 = detail::slice(f, 2, f.size());

        auto h = (x3 - x2).eval();
        auto hp = (x4 - x3).eval();

        auto df = (f3 - f2).eval();
        auto dfp = (f4 - f3).eval();

        // res = 4 * (df*hp - dfp*h)^2 / (h * hp^2 * (h+hp)^2)
        auto numer = (df.array() * hp.array() - dfp.array() * h.array()).square();
        auto denom = h.array() * hp.array().square() * (h.array() + hp.array()).square();
        auto res_forward = (4.0 * numer / denom).eval();

        // Backward Simpson: uses points [i-1, i, i+1] for interval [i, i+1]
        auto x1_b = detail::slice(x, 0, -2);
        auto x2_b = detail::slice(x, 1, -1);
        auto x3_b = detail::slice(x, 2, x.size());

        auto f1_b = detail::slice(f, 0, -2);
        auto f2_b = detail::slice(f, 1, -1);
        auto f3_b = detail::slice(f, 2, f.size());

        auto h_b = (x3_b - x2_b).eval();
        auto hm_b = (x2_b - x1_b).eval();

        auto dfm_b = (f2_b - f1_b).eval();
        auto df_b = (f3_b - f2_b).eval();

        // res = 4 * (df*hm - dfm*h)^2 / (h * hm^2 * (h+hm)^2)
        auto numer_b = (df_b.array() * hm_b.array() - dfm_b.array() * h_b.array()).square();
        auto denom_b = h_b.array() * hm_b.array().square() * (h_b.array() + hm_b.array()).square();
        auto res_backward = (4.0 * numer_b / denom_b).eval();

        // Fuse: first from forward, middle RMS of both, last from backward
        auto first_interval = detail::slice(res_forward.matrix(), 0, 1);
        auto last_interval = detail::slice(res_backward.matrix(), -1, res_backward.size());

        auto a = detail::slice(res_backward.matrix(), 0, -1);
        auto b = detail::slice(res_forward.matrix(), 1, res_forward.size());

        // RMS fusion: sqrt((a^2 + b^2)/2 + eps)
        auto mid_sq = (a.array().square() + b.array().square()) * 0.5 + detail::rms_fusion_epsilon;
        MetisVector<Scalar> mid = metis::sqrt(mid_sq.matrix());

        auto tmp = detail::concatenate(first_interval, mid);
        return detail::concatenate(tmp, last_interval);
    } else if (m == "hybrid_simpson_cubic") {
        // Use gradient to estimate f' at each point
        auto dfdx = metis::gradient(f, x, 2); // edge_order=2

        auto h = detail::compute_dx(x);
        auto df = (detail::slice(f, 1, f.size()) - detail::slice(f, 0, -1)).eval();
        auto dfdx1 = detail::slice(dfdx, 0, -1);
        auto dfdx2 = detail::slice(dfdx, 1, dfdx.size());

        // res = 4*(dfdx1^2 + dfdx1*dfdx2 + dfdx2^2)/h + 12*df/h^2 * (df/h - dfdx1 - dfdx2)
        auto term1 =
            4.0 *
            (dfdx1.array().square() + dfdx1.array() * dfdx2.array() + dfdx2.array().square()) /
            h.array();
        auto term2 = 12.0 * df.array() / h.array().square() *
                     (df.array() / h.array() - dfdx1.array() - dfdx2.array());

        return (term1 + term2).matrix();
    } else {
        throw IntegrationError("unknown curvature method: " + method +
                               ". Use 'simpson' or 'hybrid_simpson_cubic'.");
    }
}

} // namespace metis
