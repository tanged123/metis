#pragma once
/**
 * @file Integrate.hpp
 * @brief ODE integration for Metis framework
 *
 * Provides `quad` (definite integration) and `solve_ivp` (initial value problem)
 * with dual-backend support: numeric fallback and CasADi CVODES for symbolic graphs.
 *
 * See also IntegratorStep.hpp for single-step integrators (euler_step, rk4_step, etc.)
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Arithmetic.hpp"
#include "metis/math/IntegratorStep.hpp"
#include "metis/math/Linalg.hpp"
#include "metis/math/Quadrature.hpp"
#include "metis/math/Spacing.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace metis {

namespace detail {
inline NumericVector to_numeric_vector(std::initializer_list<double> init) {
    NumericVector v(static_cast<Eigen::Index>(init.size()));
    Eigen::Index idx = 0;
    for (double val : init) {
        v(idx++) = val;
    }
    return v;
}
} // namespace detail

// ============================================================================
// ODE Result structure
// ============================================================================

/**
 * @brief Result structure for ODE solvers
 *
 * Contains time points, solution values, and solver metadata.
 *
 * @tparam Scalar numeric or symbolic scalar type
 */
template <typename Scalar> struct OdeResult {
    /// Time points where solution was computed
    MetisVector<Scalar> t;

    /// Solution values at each time point (each column is a state at time t[i])
    MetisMatrix<Scalar> y;

    /// Whether integration was successful
    bool success = true;

    /// Descriptive message
    std::string message = "";

    /// Status code (0 = success)
    int status = 0;
};

/**
 * @brief Stepper selection for second-order systems q'' = a(t, q).
 */
enum class SecondOrderIntegratorMethod {
    StormerVerlet,      ///< Symplectic Stormer-Verlet / velocity-Verlet
    RungeKuttaNystrom4, ///< Classical 4th-order Runge-Kutta-Nystrom
};

/**
 * @brief Stepper selection for mass-matrix systems M(t, y) y' = f(t, y).
 */
enum class MassMatrixIntegratorMethod {
    RosenbrockEuler, ///< One-stage linearly implicit Rosenbrock-Euler
    Bdf1,            ///< Backward Euler / first-order BDF with Newton iterations
};

/**
 * @brief Result structure for second-order IVP solvers.
 *
 * @tparam Scalar numeric or symbolic scalar type
 */
template <typename Scalar> struct SecondOrderOdeResult {
    /// Time points where solution was computed
    MetisVector<Scalar> t;

    /// Generalized coordinates at each time point (each column is q(t[i]))
    MetisMatrix<Scalar> q;

    /// Generalized velocities at each time point (each column is v(t[i]))
    MetisMatrix<Scalar> v;

    /// Whether integration was successful
    bool success = true;

    /// Descriptive message
    std::string message = "";

    /// Status code (0 = success)
    int status = 0;
};

/**
 * @brief Options for second-order structure-preserving trajectory integration.
 */
struct SecondOrderIvpOptions {
    SecondOrderIntegratorMethod method = SecondOrderIntegratorMethod::StormerVerlet;
    int substeps = 1;
};

/**
 * @brief Options for stiff mass-matrix integration.
 */
struct MassMatrixIvpOptions {
    MassMatrixIntegratorMethod method = MassMatrixIntegratorMethod::RosenbrockEuler;
    int substeps = 1;
    double abstol = 1e-8;
    double reltol = 1e-6;
    double finite_difference_epsilon = 1e-7;
    int max_newton_iterations = 10;
    double newton_tolerance = 1e-10;
    LinearSolvePolicy linear_solve_policy = LinearSolvePolicy();
    casadi::Dict symbolic_integrator_options;
};

// ============================================================================
// Quadrature Result structure
// ============================================================================

/**
 * @brief Result structure for quadrature (definite integration)
 *
 * @tparam Scalar numeric or symbolic scalar type
 */
template <typename Scalar> struct QuadResult {
    /// Integral value
    Scalar value;

    /// Estimated error (only meaningful for numeric backend)
    double error = 0.0;
};

namespace detail {

inline const char *method_name(SecondOrderIntegratorMethod method) {
    switch (method) {
    case SecondOrderIntegratorMethod::StormerVerlet:
        return "Stormer-Verlet";
    case SecondOrderIntegratorMethod::RungeKuttaNystrom4:
        return "RKN4";
    }
    throw InvalidArgument("solve_second_order_ivp: unsupported second-order method");
}

inline const char *method_name(MassMatrixIntegratorMethod method) {
    switch (method) {
    case MassMatrixIntegratorMethod::RosenbrockEuler:
        return "Rosenbrock-Euler";
    case MassMatrixIntegratorMethod::Bdf1:
        return "BDF1";
    }
    throw InvalidArgument("solve_ivp_mass_matrix: unsupported mass-matrix method");
}

inline void validate_eval_count(const std::string &context, int n_eval) {
    if (n_eval < 2) {
        throw IntegrationError(context + ": n_eval must be at least 2");
    }
}

inline void validate_second_order_options(const SecondOrderIvpOptions &opts,
                                          const std::string &context) {
    if (opts.substeps <= 0) {
        throw IntegrationError(context + ": substeps must be positive");
    }
}

inline void validate_mass_matrix_options(const MassMatrixIvpOptions &opts,
                                         const std::string &context) {
    if (opts.substeps <= 0) {
        throw IntegrationError(context + ": substeps must be positive");
    }
    if (opts.abstol <= 0.0) {
        throw IntegrationError(context + ": abstol must be positive");
    }
    if (opts.reltol <= 0.0) {
        throw IntegrationError(context + ": reltol must be positive");
    }
    if (opts.finite_difference_epsilon <= 0.0) {
        throw IntegrationError(context + ": finite_difference_epsilon must be positive");
    }
    if (opts.max_newton_iterations <= 0) {
        throw IntegrationError(context + ": max_newton_iterations must be positive");
    }
    if (opts.newton_tolerance <= 0.0) {
        throw IntegrationError(context + ": newton_tolerance must be positive");
    }
}

inline void validate_second_order_initial_state(const NumericVector &q0, const NumericVector &v0) {
    if (q0.size() == 0) {
        throw IntegrationError("solve_second_order_ivp: q0 must be non-empty");
    }
    if (q0.size() != v0.size()) {
        throw IntegrationError("solve_second_order_ivp: q0 and v0 must have the same size");
    }
}

inline double inf_norm(const NumericVector &x) { return x.lpNorm<Eigen::Infinity>(); }

inline bool is_constant_zero(const SymbolicScalar &expr) {
    if (expr.is_zero()) {
        return true;
    }

    try {
        casadi::Function probe("integrate_zero_probe", std::vector<casadi::MX>{},
                               std::vector<casadi::MX>{expr});
        std::vector<casadi::DM> res = probe(std::vector<casadi::DM>{});
        return std::abs(static_cast<double>(res.at(0))) <= 1e-14;
    } catch (...) {
        return false;
    }
}

template <typename VectorFunc>
NumericMatrix finite_difference_jacobian(VectorFunc &&func, const NumericVector &x,
                                         double epsilon) {
    NumericVector f0 = func(x);
    NumericMatrix J(f0.size(), x.size());
    NumericVector x_perturbed = x;

    for (Eigen::Index j = 0; j < x.size(); ++j) {
        const double xj = x(j);
        const double step = epsilon * std::max(1.0, std::abs(xj));
        x_perturbed(j) = xj + step;
        NumericVector f_plus = func(x_perturbed);
        x_perturbed(j) = xj - step;
        NumericVector f_minus = func(x_perturbed);
        x_perturbed(j) = xj;
        J.col(j) = ((f_plus - f_minus) / (2.0 * step)).eval();
    }

    return J;
}

template <typename MassFunc>
NumericMatrix evaluate_mass_matrix(MassFunc &&mass_matrix, double t, const NumericVector &y,
                                   const std::string &context) {
    NumericMatrix M = mass_matrix(t, y);
    if (M.rows() != y.size() || M.cols() != y.size()) {
        throw IntegrationError(context + ": mass matrix must be square with size matching y");
    }
    return M;
}

template <typename RhsFunc, typename MassFunc>
NumericVector rosenbrock_euler_step(RhsFunc &&rhs, MassFunc &&mass_matrix, const NumericVector &y,
                                    double t, double dt, const MassMatrixIvpOptions &opts) {
    NumericVector f = rhs(t, y);
    if (f.size() != y.size()) {
        throw IntegrationError("solve_ivp_mass_matrix: rhs dimension must match y");
    }

    NumericMatrix M = evaluate_mass_matrix(mass_matrix, t, y, "solve_ivp_mass_matrix");
    auto rhs_at_t = [&](const NumericVector &state) { return rhs(t, state); };
    NumericMatrix J = finite_difference_jacobian(rhs_at_t, y, opts.finite_difference_epsilon);
    NumericMatrix A = (M - dt * J).eval();
    NumericVector k = metis::solve(A, f, opts.linear_solve_policy);
    return (y + dt * k).eval();
}

template <typename RhsFunc, typename MassFunc>
NumericVector bdf1_step(RhsFunc &&rhs, MassFunc &&mass_matrix, const NumericVector &y, double t,
                        double dt, const MassMatrixIvpOptions &opts) {
    const double t_next = t + dt;

    auto residual = [&](const NumericVector &trial) {
        NumericVector f = rhs(t_next, trial);
        if (f.size() != y.size()) {
            throw IntegrationError("solve_ivp_mass_matrix: rhs dimension must match y");
        }
        NumericMatrix M = evaluate_mass_matrix(mass_matrix, t_next, trial, "solve_ivp_mass_matrix");
        return (M * ((trial - y) / dt) - f).eval();
    };

    NumericVector guess = y;
    try {
        NumericVector f0 = rhs(t, y);
        NumericMatrix M0 = evaluate_mass_matrix(mass_matrix, t, y, "solve_ivp_mass_matrix");
        guess = (y + dt * metis::solve(M0, f0, opts.linear_solve_policy)).eval();
    } catch (...) {
    }

    for (int iter = 0; iter < opts.max_newton_iterations; ++iter) {
        NumericVector r = residual(guess);
        if (inf_norm(r) <= opts.newton_tolerance) {
            return guess;
        }

        NumericMatrix J =
            finite_difference_jacobian(residual, guess, opts.finite_difference_epsilon);
        NumericVector delta = metis::solve(J, -r, opts.linear_solve_policy);
        guess += delta;
        if (inf_norm(delta) <= opts.newton_tolerance) {
            return guess;
        }
    }

    NumericVector r = residual(guess);
    throw IntegrationError("solve_ivp_mass_matrix: BDF1 Newton solve failed to converge; residual "
                           "inf-norm = " +
                           std::to_string(inf_norm(r)));
}

} // namespace detail

// ============================================================================
// quad: Definite Integral
// ============================================================================

/**
 * @brief Compute definite integral using adaptive quadrature (numeric) or CVODES (symbolic)
 *
 * Numeric backend uses Gauss-Kronrod adaptive quadrature.
 * Symbolic backend wraps CasADi's CVODES integrator.
 *
 * @param func Function to integrate (callable taking Scalar, returning Scalar)
 * @param a Lower bound
 * @param b Upper bound
 * @param abstol Absolute tolerance (default 1e-8)
 * @param reltol Relative tolerance (default 1e-6)
 * @return QuadResult with integral value and error estimate
 *
 * @code
 * // Numeric integration
 * auto result = metis::quad([](double x) { return x*x; }, 0.0, 1.0);
 * // result.value ≈ 1/3
 *
 * // Symbolic integration (generates CasADi graph)
 * auto x = metis::sym("x");
 * auto expr = x * x;
 * auto sym_result = metis::quad(expr, x, 0.0, 1.0);
 * @endcode
 */
template <typename Func, typename T>
QuadResult<T> quad(Func &&func, T a, T b, double abstol = 1e-8, double reltol = 1e-6) {
    // For numeric types, use Gauss-Kronrod quadrature
    if constexpr (std::is_floating_point_v<T>) {
        const auto &embedded = detail::gauss_kronrod_15_rule();

        // Transform to [a, b]
        T center = (a + b) / 2.0;
        T halfwidth = (b - a) / 2.0;

        T kronrod_sum = 0.0;
        T gauss_sum = 0.0;

        for (int i = 0; i < 15; ++i) {
            T x = center + halfwidth * embedded.nodes[static_cast<std::size_t>(i)];
            T fx = func(x);
            kronrod_sum += embedded.primary_weights[static_cast<std::size_t>(i)] * fx;
            gauss_sum += embedded.embedded_weights[static_cast<std::size_t>(i)] * fx;
        }

        kronrod_sum *= halfwidth;
        gauss_sum *= halfwidth;

        QuadResult<T> result;
        result.value = kronrod_sum;
        result.error = std::abs(kronrod_sum - gauss_sum);
        return result;
    } else {
        // Symbolic path: This won't work with a lambda directly
        // User should use the symbolic expression overload below
        throw IntegrationError("quad with callable is not supported for symbolic types. "
                               "Use quad(expr, variable, a, b) instead.");
    }
}

/**
 * @brief Compute definite integral of a symbolic expression using CasADi CVODES
 *
 * @param expr Symbolic expression to integrate
 * @param variable Variable of integration (MX symbol)
 * @param a Lower bound (numeric)
 * @param b Upper bound (numeric)
 * @param abstol Absolute tolerance
 * @param reltol Relative tolerance
 * @return QuadResult<SymbolicScalar> with symbolic integral value
 */
inline QuadResult<SymbolicScalar> quad(const SymbolicScalar &expr, const SymbolicScalar &variable,
                                       double a, double b, double abstol = 1e-8,
                                       double reltol = 1e-6) {
    // Find all variables in the expression
    std::vector<casadi::MX> all_vars = casadi::MX::symvar(expr);

    // Separate: variable of integration vs parameters
    std::vector<casadi::MX> parameters;
    for (const auto &var : all_vars) {
        if (!casadi::MX::is_equal(var, variable)) {
            parameters.push_back(var);
        }
    }

    // Build parameter vector
    SymbolicScalar p = SymbolicScalar::vertcat(parameters);

    // Create dummy state variable (integral accumulator)
    SymbolicScalar dummy_x = SymbolicScalar::sym("__quad_x");

    // Build integrator
    // The ODE is: dx/dt = expr, where t is the variable of integration
    casadi::MXDict dae = {{"x", dummy_x}, {"t", variable}, {"p", p}, {"ode", expr}};

    casadi::Dict opts = {{"abstol", abstol}, {"reltol", reltol}};

    casadi::Function integrator = casadi::integrator("quad_integrator", "cvodes", dae, a, b, opts);

    // Evaluate with x0 = 0
    casadi::DMDict arg = {{"x0", casadi::DM(0.0)}, {"p", casadi::DM::zeros(p.size1(), p.size2())}};

    // For symbolic output, we need to call with symbolic parameters
    casadi::MXDict sym_arg = {{"x0", casadi::MX(0.0)}, {"p", p}};

    casadi::MXDict res = integrator(sym_arg);

    QuadResult<SymbolicScalar> result;
    result.value = res.at("xf");
    result.error = abstol; // Nominal error bound
    return result;
}

/**
 * @brief Solve initial value problem: dy/dt = f(t, y), y(t0) = y0
 *
 * Numeric backend uses fixed-step RK4 integrator.
 * Symbolic backend uses CasADi CVODES for automatic differentiation through the ODE.
 *
 * @tparam Func Callable type f(t, y) -> dy/dt
 * @param fun Right-hand side function f(t, y) returning derivative
 * @param t_span Integration interval (t0, tf)
 * @param y0 Initial state vector (NumericVector or initializer list)
 * @param n_eval Number of output points (default 100)
 * @param abstol Absolute tolerance
 * @param reltol Relative tolerance
 * @return OdeResult<double> with time points and solution
 *
 * @code
 * // Simple exponential decay: dy/dt = -0.5*y
 * auto sol = metis::solve_ivp(
 *     [](double t, const metis::NumericVector& y) { return -0.5 * y; },
 *     {0.0, 10.0},  // t_span
 *     {2.5},        // y0 as initializer list
 *     100           // n_eval
 * );
 *
 * // Multi-state ODE (harmonic oscillator):
 * // dy/dt = v, dv/dt = -ω²y
 * double omega = 2.0;
 * auto sol = metis::solve_ivp(
 *     [omega](double t, const metis::NumericVector& state) {
 *         metis::NumericVector dydt(2);
 *         dydt << state(1), -omega * omega * state(0);
 *         return dydt;
 *     },
 *     {0.0, M_PI / omega},
 *     {1.0, 0.0},  // y=1, v=0
 *     100
 * );
 * @endcode
 */
template <typename Func>
OdeResult<double> solve_ivp(Func &&fun, std::pair<double, double> t_span, const NumericVector &y0,
                            int n_eval = 100, double abstol = 1e-8, double reltol = 1e-6) {
    double t0 = t_span.first;
    double tf = t_span.second;

    OdeResult<double> result;
    result.t = linspace(t0, tf, n_eval);
    result.y.resize(y0.size(), n_eval);
    result.y.col(0) = y0;

    // Use RK4 with adaptive stepping (simplified: fixed step based on tolerance)
    int n_state = y0.size();
    double dt_base = (tf - t0) / (n_eval - 1);

    // Substeps per output interval for accuracy
    int substeps = std::max(1, static_cast<int>(std::ceil(1.0 / std::sqrt(reltol))));

    NumericVector y = y0;

    for (int i = 1; i < n_eval; ++i) {
        double t = result.t(i - 1);
        double dt = dt_base / substeps;

        for (int s = 0; s < substeps; ++s) {
            // Use shared step implementation
            y = rk4_step(fun, y, t, dt);
            t += dt;
        }

        result.y.col(i) = y;
    }

    result.success = true;
    result.message = "Integration successful (RK4)";
    result.status = 0;

    return result;
}

/**
 * @brief Convenience overload: solve_ivp with initializer list for y0
 *
 * @code
 * auto sol = metis::solve_ivp(
 *     [](double t, const metis::NumericVector& y) { return -0.5 * y; },
 *     {0.0, 10.0},
 *     {2.5}  // Single-element initial state
 * );
 * @endcode
 */
template <typename Func>
OdeResult<double> solve_ivp(Func &&fun, std::pair<double, double> t_span,
                            std::initializer_list<double> y0_init, int n_eval = 100,
                            double abstol = 1e-8, double reltol = 1e-6) {
    return solve_ivp(std::forward<Func>(fun), t_span, detail::to_numeric_vector(y0_init), n_eval,
                     abstol, reltol);
}

/**
 * @brief Solve a second-order IVP q'' = a(t, q), q(t0) = q0, v(t0) = v0.
 *
 * This interface targets mechanical and orbital systems directly instead of
 * forcing them through a first-order state augmentation. `StormerVerlet`
 * preserves the geometric structure of separable Hamiltonian systems, while
 * `RungeKuttaNystrom4` provides a higher-order non-symplectic alternative.
 *
 * @tparam AccelFunc Callable type a(t, q) -> qdd
 * @param acceleration Acceleration function
 * @param t_span Integration interval
 * @param q0 Initial generalized coordinates
 * @param v0 Initial generalized velocities
 * @param n_eval Number of output points
 * @param opts Integration options
 * @return SecondOrderOdeResult with q(t) and v(t)
 */
template <typename AccelFunc>
SecondOrderOdeResult<double>
solve_second_order_ivp(AccelFunc &&acceleration, std::pair<double, double> t_span,
                       const NumericVector &q0, const NumericVector &v0, int n_eval = 100,
                       const SecondOrderIvpOptions &opts = {}) {
    detail::validate_eval_count("solve_second_order_ivp", n_eval);
    detail::validate_second_order_options(opts, "solve_second_order_ivp");
    detail::validate_second_order_initial_state(q0, v0);

    const double t0 = t_span.first;
    const double tf = t_span.second;

    SecondOrderOdeResult<double> result;
    result.t = linspace(t0, tf, n_eval);
    result.q.resize(q0.size(), n_eval);
    result.v.resize(v0.size(), n_eval);
    result.q.col(0) = q0;
    result.v.col(0) = v0;

    NumericVector q = q0;
    NumericVector v = v0;
    const double dt_base = (tf - t0) / static_cast<double>(n_eval - 1);

    for (int i = 1; i < n_eval; ++i) {
        double t = result.t(i - 1);
        const double dt = dt_base / static_cast<double>(opts.substeps);

        for (int s = 0; s < opts.substeps; ++s) {
            SecondOrderStepResult<double> step;
            if (opts.method == SecondOrderIntegratorMethod::StormerVerlet) {
                step = stormer_verlet_step(acceleration, q, v, t, dt);
            } else {
                step = rkn4_step(acceleration, q, v, t, dt);
            }
            q = step.q;
            v = step.v;
            t += dt;
        }

        result.q.col(i) = q;
        result.v.col(i) = v;
    }

    result.success = true;
    result.message =
        std::string("Integration successful (") + detail::method_name(opts.method) + ")";
    result.status = 0;
    return result;
}

/**
 * @brief Convenience overload for second-order IVPs using initializer lists.
 */
template <typename AccelFunc>
SecondOrderOdeResult<double>
solve_second_order_ivp(AccelFunc &&acceleration, std::pair<double, double> t_span,
                       std::initializer_list<double> q0_init, std::initializer_list<double> v0_init,
                       int n_eval = 100, const SecondOrderIvpOptions &opts = {}) {
    return solve_second_order_ivp(std::forward<AccelFunc>(acceleration), t_span,
                                  detail::to_numeric_vector(q0_init),
                                  detail::to_numeric_vector(v0_init), n_eval, opts);
}

/**
 * @brief Solve M(t, y) y' = f(t, y) with a native numeric stiff integrator.
 *
 * `RosenbrockEuler` is a one-stage linearly implicit method that handles stiff
 * ODEs well when the mass matrix is nonsingular. `Bdf1` works on the residual
 * equation directly and also supports singular mass matrices that encode
 * simple index-1 constrained systems.
 *
 * @tparam RhsFunc Callable type f(t, y) -> rhs
 * @tparam MassFunc Callable type M(t, y) -> mass matrix
 * @param rhs Right-hand side function
 * @param mass_matrix Mass matrix function
 * @param t_span Integration interval
 * @param y0 Initial state
 * @param n_eval Number of output points
 * @param opts Integration options
 * @return OdeResult with solution samples
 */
template <typename RhsFunc, typename MassFunc>
OdeResult<double> solve_ivp_mass_matrix(RhsFunc &&rhs, MassFunc &&mass_matrix,
                                        std::pair<double, double> t_span, const NumericVector &y0,
                                        int n_eval = 100, const MassMatrixIvpOptions &opts = {}) {
    detail::validate_eval_count("solve_ivp_mass_matrix", n_eval);
    detail::validate_mass_matrix_options(opts, "solve_ivp_mass_matrix");

    const double t0 = t_span.first;
    const double tf = t_span.second;
    if (t0 == tf) {
        throw IntegrationError("solve_ivp_mass_matrix: t_span endpoints must differ");
    }
    if (y0.size() == 0) {
        throw IntegrationError("solve_ivp_mass_matrix: y0 must be non-empty");
    }

    OdeResult<double> result;
    result.t = linspace(t0, tf, n_eval);
    result.y.resize(y0.size(), n_eval);
    result.y.col(0) = y0;

    NumericVector y = y0;
    const double dt_base = (tf - t0) / static_cast<double>(n_eval - 1);

    for (int i = 1; i < n_eval; ++i) {
        double t = result.t(i - 1);
        const double dt = dt_base / static_cast<double>(opts.substeps);

        for (int s = 0; s < opts.substeps; ++s) {
            if (opts.method == MassMatrixIntegratorMethod::RosenbrockEuler) {
                y = detail::rosenbrock_euler_step(rhs, mass_matrix, y, t, dt, opts);
            } else {
                y = detail::bdf1_step(rhs, mass_matrix, y, t, dt, opts);
            }
            t += dt;
        }

        result.y.col(i) = y;
    }

    result.success = true;
    result.message =
        std::string("Integration successful (") + detail::method_name(opts.method) + ")";
    result.status = 0;
    return result;
}

/**
 * @brief Convenience overload for mass-matrix IVPs using initializer lists.
 */
template <typename RhsFunc, typename MassFunc>
OdeResult<double> solve_ivp_mass_matrix(RhsFunc &&rhs, MassFunc &&mass_matrix,
                                        std::pair<double, double> t_span,
                                        std::initializer_list<double> y0_init, int n_eval = 100,
                                        const MassMatrixIvpOptions &opts = {}) {
    return solve_ivp_mass_matrix(std::forward<RhsFunc>(rhs), std::forward<MassFunc>(mass_matrix),
                                 t_span, detail::to_numeric_vector(y0_init), n_eval, opts);
}

/**
 * @brief Solve a symbolic mass-matrix IVP using CasADi IDAS.
 *
 * The original real-time system M(t, y) y' = f(t, y) is converted into the
 * semi-explicit DAE
 *
 * x' = z
 * 0  = M(t, x) z - f(t, x)
 *
 * after the same time normalization already used by `solve_ivp_expr`.
 *
 * @param rhs_expr Symbolic right-hand side f(t, y)
 * @param mass_expr Symbolic mass matrix M(t, y)
 * @param t_var Time variable symbol
 * @param y_var State variable symbol(s)
 * @param t_span Integration interval
 * @param y0 Initial state
 * @param n_eval Number of output points
 * @param opts Integration options
 * @return OdeResult with sampled numeric solution
 */
inline OdeResult<double>
solve_ivp_mass_matrix_expr(const SymbolicScalar &rhs_expr, const SymbolicScalar &mass_expr,
                           const SymbolicScalar &t_var, const SymbolicScalar &y_var,
                           std::pair<double, double> t_span, const NumericVector &y0,
                           int n_eval = 100, const MassMatrixIvpOptions &opts = {}) {
    detail::validate_eval_count("solve_ivp_mass_matrix_expr", n_eval);
    detail::validate_mass_matrix_options(opts, "solve_ivp_mass_matrix_expr");

    const double t0 = t_span.first;
    const double tf = t_span.second;
    if (t0 == tf) {
        throw IntegrationError("solve_ivp_mass_matrix_expr: t_span endpoints must differ");
    }
    const int n_state = static_cast<int>(y0.size());
    if (n_state == 0) {
        throw IntegrationError("solve_ivp_mass_matrix_expr: y0 must be non-empty");
    }

    SymbolicScalar t_normalized = SymbolicScalar::sym("t_norm");
    SymbolicScalar rhs_sub =
        SymbolicScalar::substitute(rhs_expr, t_var, t0 + (tf - t0) * t_normalized);
    SymbolicScalar mass_sub =
        SymbolicScalar::substitute(mass_expr, t_var, t0 + (tf - t0) * t_normalized);

    if (mass_sub.size1() != n_state || mass_sub.size2() != n_state) {
        throw IntegrationError(
            "solve_ivp_mass_matrix_expr: mass matrix must be square with size matching y0");
    }
    if (rhs_sub.size1() != n_state || rhs_sub.size2() != 1) {
        throw IntegrationError("solve_ivp_mass_matrix_expr: rhs must be a column vector matching "
                               "the size of y0");
    }

    casadi::DM y0_dm_full(n_state, 1);
    for (int i = 0; i < n_state; ++i) {
        y0_dm_full(i) = y0(i);
    }

    std::vector<double> t_eval_normalized(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        t_eval_normalized[i] = static_cast<double>(i) / static_cast<double>(n_eval - 1);
    }

    std::vector<bool> row_has_structural_nnz(static_cast<std::size_t>(n_state), false);
    for (int row = 0; row < n_state; ++row) {
        for (int col = 0; col < n_state; ++col) {
            if (!detail::is_constant_zero(mass_sub(row, col))) {
                row_has_structural_nnz[static_cast<std::size_t>(row)] = true;
                break;
            }
        }
    }

    std::vector<int> differential_indices;
    std::vector<int> algebraic_indices;
    for (int i = 0; i < n_state; ++i) {
        if (row_has_structural_nnz[static_cast<std::size_t>(i)]) {
            differential_indices.push_back(i);
        } else {
            algebraic_indices.push_back(i);
        }
    }

    auto is_component_of = [](const casadi::MX &var, const casadi::MX &vector_expr) {
        for (int i = 0; i < vector_expr.nnz(); ++i) {
            if (casadi::MX::is_equal(var, vector_expr(i))) {
                return true;
            }
        }
        return false;
    };

    casadi::Dict integrator_opts = {
        {"abstol", opts.abstol}, {"reltol", opts.reltol}, {"calc_ic", true}};
    for (const auto &kv : opts.symbolic_integrator_options) {
        integrator_opts[kv.first] = kv.second;
    }

    OdeResult<double> result;
    result.t.resize(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        result.t(i) = t0 + (tf - t0) * t_eval_normalized[i];
    }
    result.y.resize(n_state, n_eval);

    if (algebraic_indices.empty()) {
        SymbolicScalar z_var = SymbolicScalar::sym("y_dot_norm", n_state, 1);
        SymbolicScalar alg = casadi::MX::mtimes(mass_sub, z_var) - (tf - t0) * rhs_sub;

        std::vector<casadi::MX> all_vars = casadi::MX::symvar(casadi::MX::vertcat({rhs_sub, alg}));
        std::vector<casadi::MX> parameters;
        for (const auto &var : all_vars) {
            if (casadi::MX::is_equal(var, t_normalized) || casadi::MX::is_equal(var, y_var) ||
                casadi::MX::is_equal(var, z_var) || is_component_of(var, y_var) ||
                is_component_of(var, z_var)) {
                continue;
            }
            parameters.push_back(var);
        }

        casadi::MX p = casadi::MX::vertcat(parameters);
        casadi::MXDict dae = {{"x", y_var}, {"z", z_var},   {"t", t_normalized},
                              {"p", p},     {"ode", z_var}, {"alg", alg}};

        casadi::Function integrator = casadi::integrator("mass_matrix_ivp_integrator", "idas", dae,
                                                         0.0, t_eval_normalized, integrator_opts);

        casadi::DMDict integrator_args;
        integrator_args["x0"] = y0_dm_full;
        integrator_args["z0"] = casadi::DM::zeros(n_state, 1);
        integrator_args["p"] = casadi::DM::zeros(p.size1(), p.size2());
        casadi::DMDict res_dm = integrator(integrator_args);

        casadi::DM xf = res_dm.at("xf");
        for (int i = 0; i < n_state; ++i) {
            for (int j = 0; j < n_eval; ++j) {
                result.y(i, j) = static_cast<double>(xf(i, j));
            }
        }
    } else {
        if (differential_indices.empty()) {
            throw IntegrationError("solve_ivp_mass_matrix_expr: fully algebraic systems are not "
                                   "supported");
        }

        const int n_diff = static_cast<int>(differential_indices.size());
        const int n_alg = static_cast<int>(algebraic_indices.size());

        SymbolicScalar x_var = SymbolicScalar::sym("y_diff", n_diff, 1);
        SymbolicScalar z_var = SymbolicScalar::sym("y_alg", n_alg, 1);

        SymbolicScalar partitioned_y = SymbolicScalar(n_state, 1);
        int diff_cursor = 0;
        int alg_cursor = 0;
        for (int i = 0; i < n_state; ++i) {
            if (row_has_structural_nnz[static_cast<std::size_t>(i)]) {
                partitioned_y(i) = x_var(diff_cursor++);
            } else {
                partitioned_y(i) = z_var(alg_cursor++);
            }
        }

        SymbolicScalar rhs_partitioned = SymbolicScalar::substitute(rhs_sub, y_var, partitioned_y);
        SymbolicScalar mass_partitioned =
            SymbolicScalar::substitute(mass_sub, y_var, partitioned_y);

        SymbolicScalar rhs_diff(n_diff, 1);
        SymbolicScalar rhs_alg(n_alg, 1);
        for (int i = 0; i < n_diff; ++i) {
            rhs_diff(i) = rhs_partitioned(differential_indices[static_cast<std::size_t>(i)]);
        }
        for (int i = 0; i < n_alg; ++i) {
            rhs_alg(i) = rhs_partitioned(algebraic_indices[static_cast<std::size_t>(i)]);
        }

        SymbolicScalar mass_diff(n_diff, n_diff);
        for (int i = 0; i < n_diff; ++i) {
            for (int j = 0; j < n_diff; ++j) {
                mass_diff(i, j) =
                    mass_partitioned(differential_indices[static_cast<std::size_t>(i)],
                                     differential_indices[static_cast<std::size_t>(j)]);
            }
        }

        SymbolicScalar ode = casadi::MX::solve(mass_diff, (tf - t0) * rhs_diff);

        std::vector<casadi::MX> all_vars = casadi::MX::symvar(casadi::MX::vertcat(
            {rhs_partitioned, casadi::MX::reshape(mass_partitioned, n_state * n_state, 1)}));
        std::vector<casadi::MX> parameters;
        for (const auto &var : all_vars) {
            if (casadi::MX::is_equal(var, t_normalized) || casadi::MX::is_equal(var, x_var) ||
                casadi::MX::is_equal(var, z_var) || is_component_of(var, x_var) ||
                is_component_of(var, z_var)) {
                continue;
            }
            parameters.push_back(var);
        }

        casadi::MX p = casadi::MX::vertcat(parameters);
        casadi::MXDict dae = {{"x", x_var}, {"z", z_var}, {"t", t_normalized},
                              {"p", p},     {"ode", ode}, {"alg", rhs_alg}};

        casadi::Function integrator =
            casadi::integrator("mass_matrix_ivp_integrator_partitioned", "idas", dae, 0.0,
                               t_eval_normalized, integrator_opts);

        casadi::DM x0_dm(n_diff, 1);
        casadi::DM z0_dm(n_alg, 1);
        for (int i = 0; i < n_diff; ++i) {
            x0_dm(i) = y0(differential_indices[static_cast<std::size_t>(i)]);
        }
        for (int i = 0; i < n_alg; ++i) {
            z0_dm(i) = y0(algebraic_indices[static_cast<std::size_t>(i)]);
        }

        casadi::DMDict integrator_args;
        integrator_args["x0"] = x0_dm;
        integrator_args["z0"] = z0_dm;
        integrator_args["p"] = casadi::DM::zeros(p.size1(), p.size2());
        casadi::DMDict res_dm = integrator(integrator_args);

        casadi::DM xf = res_dm.at("xf");
        casadi::DM zf = res_dm.at("zf");
        for (int i = 0; i < n_eval; ++i) {
            int diff_out = 0;
            int alg_out = 0;
            for (int state = 0; state < n_state; ++state) {
                if (row_has_structural_nnz[static_cast<std::size_t>(state)]) {
                    result.y(state, i) = static_cast<double>(xf(diff_out++, i));
                } else {
                    result.y(state, i) = static_cast<double>(zf(alg_out++, i));
                }
            }
        }
    }

    result.success = true;
    result.message = "Integration successful (IDAS mass matrix)";
    result.status = 0;
    return result;
}

/**
 * @brief Solve IVP with symbolic ODE function (CasADi CVODES backend)
 *
 * This overload accepts a callable that takes symbolic arguments and returns
 * symbolic expressions, enabling automatic differentiation through the ODE.
 *
 * @tparam Func Callable type f(MX t, MX y) -> MX dy/dt
 * @param fun Symbolic ODE function
 * @param t_span Integration interval (t0, tf)
 * @param y0 Initial state (numeric, will be used for evaluation)
 * @param n_eval Number of output points
 * @param abstol Absolute tolerance
 * @param reltol Relative tolerance
 * @return OdeResult<SymbolicScalar> with symbolic solution
 */
template <typename Func>
OdeResult<SymbolicScalar> solve_ivp_symbolic(Func &&fun, std::pair<double, double> t_span,
                                             const Eigen::VectorXd &y0, int n_eval = 100,
                                             double abstol = 1e-8, double reltol = 1e-6) {
    double t0 = t_span.first;
    double tf = t_span.second;
    int n_state = y0.size();

    // Create symbolic variables for the ODE
    SymbolicScalar t_var = SymbolicScalar::sym("t");
    SymbolicScalar y_var = SymbolicScalar::sym("y", n_state, 1);

    // Evaluate the ODE function symbolically
    SymbolicScalar ode_expr;
    if constexpr (std::is_invocable_v<Func, casadi::MX, casadi::MX>) {
        ode_expr = fun(t_var, y_var);
    } else {
        // Try calling with Eigen-wrapped MX
        SymbolicVector y_sym(n_state);
        for (int i = 0; i < n_state; ++i) {
            y_sym(i) = y_var(i);
        }
        auto result = fun(t_var, y_sym);
        // Convert back to MX
        ode_expr = casadi::MX(n_state, 1);
        for (int i = 0; i < n_state; ++i) {
            ode_expr(i) = result(i);
        }
    }

    // Normalize time to [0, 1] for CVODES stability
    // t_real = t0 + (tf - t0) * t_normalized
    // ode_normalized = ode_real * (tf - t0)
    casadi::MX t_normalized = casadi::MX::sym("t_norm");
    casadi::MX ode_normalized =
        casadi::MX::substitute(ode_expr, t_var, t0 + (tf - t0) * t_normalized) * (tf - t0);

    // Create evaluation time grid (normalized)
    std::vector<double> t_eval_normalized(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        t_eval_normalized[i] = static_cast<double>(i) / (n_eval - 1);
    }

    // Build integrator
    casadi::MXDict dae = {{"x", y_var}, {"t", t_normalized}, {"ode", ode_normalized}};

    casadi::Dict opts = {{"abstol", abstol}, {"reltol", reltol}};

    casadi::Function integrator =
        casadi::integrator("ivp_integrator", "cvodes", dae, 0.0, t_eval_normalized, opts);

    // Convert y0 to DM
    casadi::DM y0_dm(n_state, 1);
    for (int i = 0; i < n_state; ++i) {
        y0_dm(i) = y0(i);
    }

    // Evaluate integrator - explicitly construct DMDict to avoid ambiguity
    casadi::DMDict integrator_args;
    integrator_args["x0"] = y0_dm;
    casadi::DMDict res_dm = integrator(integrator_args);

    OdeResult<SymbolicScalar> result;

    // Create symbolic time vector
    result.t.resize(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        result.t(i) = t0 + (tf - t0) * t_eval_normalized[i];
    }

    // Extract solution (xf contains all intermediate states)
    casadi::DM xf = res_dm.at("xf");
    result.y.resize(n_state, n_eval);
    for (int i = 0; i < n_state; ++i) {
        for (int j = 0; j < n_eval; ++j) {
            result.y(i, j) = static_cast<double>(xf(i, j));
        }
    }

    result.success = true;
    result.message = "Integration successful (CVODES)";
    result.status = 0;

    return result;
}

/**
 * @brief Solve IVP with a symbolic expression directly (expression-based API)
 *
 * This is the most flexible symbolic API, allowing the user to provide
 * the ODE as a CasADi expression with explicit variable specification.
 *
 * @param ode_expr Symbolic ODE expression (dy/dt)
 * @param t_var Time variable (MX symbol)
 * @param y_var State variable(s) (MX symbol vector)
 * @param t_span Integration interval
 * @param y0 Initial state
 * @param n_eval Number of output points
 * @param abstol Absolute tolerance
 * @param reltol Relative tolerance
 * @return OdeResult containing numeric time and solution values
 */
inline OdeResult<double> solve_ivp_expr(const SymbolicScalar &ode_expr, const SymbolicScalar &t_var,
                                        const SymbolicScalar &y_var,
                                        std::pair<double, double> t_span, const NumericVector &y0,
                                        int n_eval = 100, double abstol = 1e-8,
                                        double reltol = 1e-6) {
    double t0 = t_span.first;
    double tf = t_span.second;
    int n_state = y0.size();

    // Normalize time to [0, 1]
    SymbolicScalar t_normalized = SymbolicScalar::sym("t_norm");
    SymbolicScalar ode_normalized =
        SymbolicScalar::substitute(ode_expr, t_var, t0 + (tf - t0) * t_normalized) * (tf - t0);

    // Create evaluation time grid (normalized)
    std::vector<double> t_eval_normalized(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        t_eval_normalized[i] = static_cast<double>(i) / (n_eval - 1);
    }

    // Find parameters (variables in ode other than t and y)
    std::vector<casadi::MX> all_vars = casadi::MX::symvar(ode_normalized);
    std::vector<casadi::MX> parameters;
    for (const auto &var : all_vars) {
        if (!casadi::MX::is_equal(var, t_normalized) && !casadi::MX::is_equal(var, y_var)) {
            // Check each element of y_var
            bool is_y = false;
            for (int i = 0; i < y_var.size1() * y_var.size2(); ++i) {
                if (casadi::MX::is_equal(var, y_var(i))) {
                    is_y = true;
                    break;
                }
            }
            if (!is_y) {
                parameters.push_back(var);
            }
        }
    }

    casadi::MX p = casadi::MX::vertcat(parameters);

    // Build integrator
    casadi::MXDict dae = {{"x", y_var}, {"t", t_normalized}, {"p", p}, {"ode", ode_normalized}};

    casadi::Dict opts = {{"abstol", abstol}, {"reltol", reltol}};

    casadi::Function integrator =
        casadi::integrator("ivp_integrator", "cvodes", dae, 0.0, t_eval_normalized, opts);

    // Convert y0 to DM
    casadi::DM y0_dm(n_state, 1);
    for (int i = 0; i < n_state; ++i) {
        y0_dm(i) = y0(i);
    }

    // Empty parameters for now (user would need to provide values)
    casadi::DM p_dm = casadi::DM::zeros(p.size1(), p.size2());

    // Evaluate integrator - explicitly construct DMDict to avoid ambiguity
    casadi::DMDict integrator_args;
    integrator_args["x0"] = y0_dm;
    integrator_args["p"] = p_dm;
    casadi::DMDict res_dm = integrator(integrator_args);

    OdeResult<double> result;

    // Create time vector
    result.t.resize(n_eval);
    for (int i = 0; i < n_eval; ++i) {
        result.t(i) = t0 + (tf - t0) * t_eval_normalized[i];
    }

    // Extract solution
    casadi::DM xf = res_dm.at("xf");
    result.y.resize(n_state, n_eval);
    for (int i = 0; i < n_state; ++i) {
        for (int j = 0; j < n_eval; ++j) {
            result.y(i, j) = static_cast<double>(xf(i, j));
        }
    }

    result.success = true;
    result.message = "Integration successful (CVODES)";
    result.status = 0;

    return result;
}

} // namespace metis
