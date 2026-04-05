#pragma once
/**
 * @file DiffTestHarness.hpp
 * @brief GTest-free core for differentiability testing of dual-mode functions
 *
 * Provides utilities to verify that dual-mode (numeric/symbolic) functions:
 *   (a) compile in symbolic mode
 *   (b) produce matching numeric and symbolic outputs
 *   (c) have AD Jacobians that match finite-difference Jacobians
 *
 * This header has NO GoogleTest dependency. See GTestDiffTest.hpp for GTest wrappers.
 *
 * @see GTestDiffTest.hpp, AutoDiff.hpp, FiniteDifference.hpp
 */

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <janus/core/Function.hpp>
#include <janus/core/JanusIO.hpp>
#include <janus/core/JanusTypes.hpp>
#include <janus/math/AutoDiff.hpp>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace janus::diff_test {

// ============================================================================
// Configuration
// ============================================================================

/**
 * @brief Options controlling differentiability test behavior
 */
struct DiffTestOptions {
    double fd_step = 1e-7;    ///< Finite difference perturbation size
    double jac_rtol = 1e-5;   ///< Relative tolerance for AD-vs-FD Jacobian comparison
    double jac_atol = 1e-8;   ///< Absolute tolerance for AD-vs-FD Jacobian comparison
    double value_tol = 1e-10; ///< Tolerance for numeric-vs-symbolic value comparison
};

// ============================================================================
// Result Types
// ============================================================================

/**
 * @brief Result of a dual-mode (symbolic compilation + value match) check
 */
struct DualModeResult {
    bool symbolic_compiles = false; ///< Did the lambda execute in symbolic mode?
    bool values_match = false;      ///< Does eval(symbolic_output) == numeric_output?
    double max_value_error = 0.0;   ///< Largest |symbolic - numeric| across outputs
    std::string failure_detail;     ///< Human-readable on failure, empty on success
};

/**
 * @brief Result of a full differentiability check (dual-mode + Jacobian)
 */
struct DiffTestResult : DualModeResult {
    bool jacobian_matches = false;   ///< Does AD Jacobian ≈ FD Jacobian?
    double max_jacobian_error = 0.0; ///< Largest element-wise |AD - FD|
    Eigen::MatrixXd ad_jacobian;     ///< AD Jacobian for diagnostics
    Eigen::MatrixXd fd_jacobian;     ///< FD Jacobian for diagnostics
};

// ============================================================================
// Detail: Arity Detection and Invocation Helpers
// ============================================================================

namespace detail {

/**
 * @brief Detect the arity of a callable when invoked with arguments of type T.
 *
 * Checks from 1 to 8 arguments and returns the first arity that compiles.
 * For generic lambdas like [](auto x, auto y) { ... }, std::is_invocable_v
 * correctly returns false for wrong arities.
 */
template <typename F, typename T> constexpr int detect_arity() {
    if constexpr (std::is_invocable_v<F, T>)
        return 1;
    else if constexpr (std::is_invocable_v<F, T, T>)
        return 2;
    else if constexpr (std::is_invocable_v<F, T, T, T>)
        return 3;
    else if constexpr (std::is_invocable_v<F, T, T, T, T>)
        return 4;
    else if constexpr (std::is_invocable_v<F, T, T, T, T, T>)
        return 5;
    else if constexpr (std::is_invocable_v<F, T, T, T, T, T, T>)
        return 6;
    else if constexpr (std::is_invocable_v<F, T, T, T, T, T, T, T>)
        return 7;
    else if constexpr (std::is_invocable_v<F, T, T, T, T, T, T, T, T>)
        return 8;
    else {
        static_assert(std::is_invocable_v<F, T>,
                      "Lambda must accept 1-8 scalar arguments of the given type");
        return 0;
    }
}

/**
 * @brief Get the detected arity of a callable for a given scalar type.
 */
template <typename Func, typename Scalar> constexpr int arity_of() {
    return detect_arity<std::decay_t<Func>, Scalar>();
}

/**
 * @brief Invoke a callable with N arguments unpacked from a vector.
 *
 * Validates that args.size() matches the callable's detected arity.
 * Mismatched sizes indicate a bug in the test (wrong number of values in a test point).
 */
template <typename Func, typename Scalar> auto invoke(Func &&f, const std::vector<Scalar> &args) {
    using F = std::decay_t<Func>;
    constexpr int N = detect_arity<F, Scalar>();

    if (static_cast<int>(args.size()) != N) {
        throw std::invalid_argument("DiffTestHarness: test point has " +
                                    std::to_string(args.size()) + " values but lambda accepts " +
                                    std::to_string(N) +
                                    " arguments. Check your test_points dimensions.");
    }

    if constexpr (N == 1)
        return f(args[0]);
    else if constexpr (N == 2)
        return f(args[0], args[1]);
    else if constexpr (N == 3)
        return f(args[0], args[1], args[2]);
    else if constexpr (N == 4)
        return f(args[0], args[1], args[2], args[3]);
    else if constexpr (N == 5)
        return f(args[0], args[1], args[2], args[3], args[4]);
    else if constexpr (N == 6)
        return f(args[0], args[1], args[2], args[3], args[4], args[5]);
    else if constexpr (N == 7)
        return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    else if constexpr (N == 8)
        return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
}

/**
 * @brief Normalize a scalar or vector return to Eigen::VectorXd
 */
template <typename T> Eigen::VectorXd to_numeric_vector(const T &val) {
    if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
        Eigen::VectorXd v(1);
        v(0) = static_cast<double>(val);
        return v;
    } else {
        // Assume Eigen vector/matrix — flatten to column
        Eigen::VectorXd v(val.size());
        for (Eigen::Index i = 0; i < val.size(); ++i) {
            v(i) = static_cast<double>(val(i));
        }
        return v;
    }
}

/**
 * @brief Normalize a symbolic scalar or vector return to a vector of SymbolicArg
 */
template <typename T> std::vector<SymbolicArg> to_symbolic_output_args(const T &val) {
    if constexpr (std::is_same_v<std::decay_t<T>, SymbolicScalar>) {
        return {val};
    } else {
        // Assume Eigen vector of MX elements
        std::vector<SymbolicArg> out;
        out.reserve(static_cast<size_t>(val.size()));
        for (Eigen::Index i = 0; i < val.size(); ++i) {
            out.emplace_back(val(i));
        }
        return out;
    }
}

/**
 * @brief Format a test point as a string for diagnostics
 */
inline std::string format_point(const std::vector<double> &point) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < point.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << point[i];
    }
    ss << "]";
    return ss.str();
}

/**
 * @brief Format an Eigen matrix as a string for diagnostics
 */
inline std::string format_matrix(const Eigen::MatrixXd &m) {
    std::ostringstream ss;
    ss << std::setprecision(8);
    if (m.rows() == 1 && m.cols() == 1) {
        ss << m(0, 0);
    } else {
        ss << "\n";
        for (Eigen::Index i = 0; i < m.rows(); ++i) {
            ss << "    [";
            for (Eigen::Index j = 0; j < m.cols(); ++j) {
                if (j > 0)
                    ss << ", ";
                ss << m(i, j);
            }
            ss << "]\n";
        }
    }
    return ss.str();
}

} // namespace detail

// ============================================================================
// Core Verification Functions
// ============================================================================

/**
 * @brief Verify that a dual-mode function compiles symbolically and that
 *        symbolic evaluation matches numeric evaluation at a single point.
 */
template <typename Func>
DualModeResult verify_dual_mode_at_point(Func &&f, const std::vector<double> &point,
                                         const DiffTestOptions &opts = {}) {
    DualModeResult result;
    // Derive n from the lambda's arity, not from point.size().
    // invoke() validates the two match and throws if they disagree.
    constexpr int n = detail::arity_of<Func, double>();

    // Step 1: Numeric evaluation
    Eigen::VectorXd numeric_output;
    try {
        numeric_output = detail::to_numeric_vector(detail::invoke(f, point));
    } catch (const std::exception &e) {
        result.failure_detail =
            "Numeric evaluation failed at point " + detail::format_point(point) + ": " + e.what();
        return result;
    }

    // Step 2: Symbolic evaluation
    std::vector<SymbolicScalar> sym_vars;
    sym_vars.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        sym_vars.push_back(janus::sym("x" + std::to_string(i)));
    }

    try {
        auto sym_result = detail::invoke(f, sym_vars);
        result.symbolic_compiles = true;

        // Build Function and evaluate at test point
        std::vector<SymbolicArg> input_args(sym_vars.begin(), sym_vars.end());
        std::vector<SymbolicArg> output_args = detail::to_symbolic_output_args(sym_result);

        // Vertcat outputs into single MX for evaluation
        std::vector<casadi::MX> mx_outputs;
        mx_outputs.reserve(output_args.size());
        for (const auto &o : output_args) {
            mx_outputs.push_back(o.get());
        }
        casadi::MX out_cat = casadi::MX::vertcat(mx_outputs);

        janus::Function func(input_args, {SymbolicArg(out_cat)});
        auto results = func(point);
        Eigen::VectorXd symbolic_output =
            Eigen::Map<Eigen::VectorXd>(results[0].data(), results[0].rows() * results[0].cols());

        // Step 3: Compare
        if (numeric_output.size() != symbolic_output.size()) {
            result.values_match = false;
            result.failure_detail =
                "Output dimension mismatch: numeric=" + std::to_string(numeric_output.size()) +
                " symbolic=" + std::to_string(symbolic_output.size()) + " at point " +
                detail::format_point(point);
            return result;
        }

        result.max_value_error = (numeric_output - symbolic_output).cwiseAbs().maxCoeff();
        result.values_match = result.max_value_error < opts.value_tol;

        if (!result.values_match) {
            std::ostringstream ss;
            ss << "Value mismatch at point " << detail::format_point(point)
               << ": max_error=" << result.max_value_error << " (tol=" << opts.value_tol << ")"
               << "\n  Numeric:  " << numeric_output.transpose()
               << "\n  Symbolic: " << symbolic_output.transpose();
            result.failure_detail = ss.str();
        }

    } catch (const std::exception &e) {
        result.symbolic_compiles = false;
        result.failure_detail =
            "Symbolic mode failed at point " + detail::format_point(point) + ": " + e.what();
    }

    return result;
}

/**
 * @brief Verify dual-mode across multiple test points. Returns on first failure.
 */
template <typename Func>
DualModeResult verify_dual_mode(Func &&f, const std::vector<std::vector<double>> &test_points,
                                const DiffTestOptions &opts = {}) {
    DualModeResult overall;

    if (test_points.empty()) {
        overall.failure_detail = "No test points supplied";
        return overall;
    }

    overall.symbolic_compiles = true;
    overall.values_match = true;
    overall.max_value_error = 0.0;

    for (const auto &point : test_points) {
        auto r = verify_dual_mode_at_point(f, point, opts);
        overall.max_value_error = std::max(overall.max_value_error, r.max_value_error);

        if (!r.symbolic_compiles) {
            overall.symbolic_compiles = false;
            overall.failure_detail = r.failure_detail;
            return overall;
        }
        if (!r.values_match) {
            overall.values_match = false;
            overall.failure_detail = r.failure_detail;
            return overall;
        }
    }

    return overall;
}

/**
 * @brief Verify differentiability at a single test point:
 *        dual-mode checks + AD Jacobian vs FD Jacobian.
 */
template <typename Func>
DiffTestResult verify_differentiable_at_point(Func &&f, const std::vector<double> &point,
                                              const DiffTestOptions &opts = {}) {
    DiffTestResult result;
    constexpr int n = detail::arity_of<Func, double>();

    // Step 1: Run dual-mode checks
    auto dm = verify_dual_mode_at_point(f, point, opts);
    result.symbolic_compiles = dm.symbolic_compiles;
    result.values_match = dm.values_match;
    result.max_value_error = dm.max_value_error;
    result.failure_detail = dm.failure_detail;

    if (!result.symbolic_compiles || !result.values_match) {
        return result;
    }

    // Step 2: Compute FD Jacobian (central differences)
    Eigen::VectorXd f0;
    try {
        f0 = detail::to_numeric_vector(detail::invoke(f, point));
    } catch (const std::exception &e) {
        result.jacobian_matches = false;
        result.failure_detail =
            "Numeric evaluation failed at point " + detail::format_point(point) + ": " + e.what();
        return result;
    }
    const int m = static_cast<int>(f0.size());

    Eigen::MatrixXd fd_jac(m, n);
    for (int j = 0; j < n; ++j) {
        std::vector<double> point_plus = point;
        std::vector<double> point_minus = point;
        point_plus[static_cast<size_t>(j)] += opts.fd_step;
        point_minus[static_cast<size_t>(j)] -= opts.fd_step;

        try {
            Eigen::VectorXd f_plus = detail::to_numeric_vector(detail::invoke(f, point_plus));
            Eigen::VectorXd f_minus = detail::to_numeric_vector(detail::invoke(f, point_minus));
            fd_jac.col(j) = (f_plus - f_minus) / (2.0 * opts.fd_step);
        } catch (const std::exception &e) {
            result.jacobian_matches = false;
            result.failure_detail = "FD probe failed at point " + detail::format_point(point) +
                                    " perturbing input " + std::to_string(j) + ": " + e.what();
            return result;
        }
    }
    result.fd_jacobian = fd_jac;

    // Step 3: Compute AD Jacobian
    std::vector<SymbolicScalar> sym_vars;
    sym_vars.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        sym_vars.push_back(janus::sym("x" + std::to_string(i)));
    }

    try {
        auto sym_result = detail::invoke(f, sym_vars);
        std::vector<SymbolicArg> output_args = detail::to_symbolic_output_args(sym_result);
        std::vector<SymbolicArg> input_args(sym_vars.begin(), sym_vars.end());

        // Compute symbolic Jacobian
        auto J_sym = janus::jacobian(output_args, input_args);

        // Wrap in Function and evaluate
        janus::Function J_func(input_args, {SymbolicArg(J_sym)});
        auto J_results = J_func(point);
        result.ad_jacobian = J_results[0];

    } catch (const std::exception &e) {
        result.jacobian_matches = false;
        result.failure_detail = "AD Jacobian computation failed at point " +
                                detail::format_point(point) + ": " + e.what();
        return result;
    }

    // Step 4: Compare AD vs FD Jacobian
    if (result.ad_jacobian.rows() != fd_jac.rows() || result.ad_jacobian.cols() != fd_jac.cols()) {
        result.jacobian_matches = false;
        std::ostringstream ss;
        ss << "Jacobian dimension mismatch at point " << detail::format_point(point)
           << ": AD=" << result.ad_jacobian.rows() << "x" << result.ad_jacobian.cols()
           << " FD=" << fd_jac.rows() << "x" << fd_jac.cols();
        result.failure_detail = ss.str();
        return result;
    }

    result.max_jacobian_error = 0.0;
    result.jacobian_matches = true;
    int worst_row = 0, worst_col = 0;
    for (int i = 0; i < result.ad_jacobian.rows() && result.jacobian_matches; ++i) {
        for (int j = 0; j < result.ad_jacobian.cols(); ++j) {
            double ad_val = result.ad_jacobian(i, j);
            double fd_val = fd_jac(i, j);

            if (!std::isfinite(ad_val) || !std::isfinite(fd_val)) {
                result.jacobian_matches = false;
                result.max_jacobian_error = std::numeric_limits<double>::infinity();
                worst_row = i;
                worst_col = j;
                break;
            }

            double error = std::abs(ad_val - fd_val);
            if (error > result.max_jacobian_error) {
                result.max_jacobian_error = error;
                worst_row = i;
                worst_col = j;
            }

            double tol = opts.jac_atol + opts.jac_rtol * std::abs(ad_val);
            if (error > tol) {
                result.jacobian_matches = false;
                break;
            }
        }
    }

    if (!result.jacobian_matches) {
        std::ostringstream ss;
        ss << "Jacobian mismatch at point " << detail::format_point(point)
           << ":\n  Max error: " << result.max_jacobian_error << " (rtol=" << opts.jac_rtol
           << ", atol=" << opts.jac_atol << ")"
           << "\n  Worst element: (" << worst_row << ", " << worst_col
           << "): AD=" << result.ad_jacobian(worst_row, worst_col)
           << ", FD=" << fd_jac(worst_row, worst_col)
           << "\n  AD Jacobian: " << detail::format_matrix(result.ad_jacobian)
           << "  FD Jacobian: " << detail::format_matrix(fd_jac);
        result.failure_detail = ss.str();
    }

    return result;
}

/**
 * @brief Verify differentiability across multiple test points. Returns on first failure.
 *
 * For each test point, verifies:
 *   1. Symbolic mode compiles
 *   2. Symbolic output matches numeric output within value_tol
 *   3. AD Jacobian matches FD Jacobian within jac_rtol/jac_atol
 *
 * @tparam Func Lambda type: auto f(Scalar x1, ..., Scalar xN) -> Scalar or JanusVector<Scalar>
 * @param f The dual-mode function to test
 * @param test_points Vector of test points, each a vector of doubles
 * @param opts Tolerance and step size options
 * @return DiffTestResult with pass/fail and diagnostics
 */
template <typename Func>
DiffTestResult verify_differentiable(Func &&f, const std::vector<std::vector<double>> &test_points,
                                     const DiffTestOptions &opts = {}) {
    DiffTestResult overall;

    if (test_points.empty()) {
        overall.failure_detail = "No test points supplied";
        return overall;
    }

    overall.symbolic_compiles = true;
    overall.values_match = true;
    overall.jacobian_matches = true;
    overall.max_value_error = 0.0;
    overall.max_jacobian_error = 0.0;

    for (const auto &point : test_points) {
        auto r = verify_differentiable_at_point(f, point, opts);
        overall.max_value_error = std::max(overall.max_value_error, r.max_value_error);
        overall.max_jacobian_error = std::max(overall.max_jacobian_error, r.max_jacobian_error);

        if (!r.symbolic_compiles) {
            overall.symbolic_compiles = false;
            overall.failure_detail = r.failure_detail;
            return overall;
        }
        if (!r.values_match) {
            overall.values_match = false;
            overall.failure_detail = r.failure_detail;
            return overall;
        }
        if (!r.jacobian_matches) {
            overall.jacobian_matches = false;
            overall.failure_detail = r.failure_detail;
            overall.ad_jacobian = r.ad_jacobian;
            overall.fd_jacobian = r.fd_jacobian;
            return overall;
        }
    }

    return overall;
}

} // namespace janus::diff_test
