/**
 * @file Opti.hpp
 * @brief High-level nonlinear programming interface
 */

#pragma once

#include "OptiOptions.hpp"
#include "OptiSol.hpp"
#include "OptiSweep.hpp"
#include "Scaling.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Calculus.hpp"
#include "metis/math/FiniteDifference.hpp"
#include <algorithm>
#include <casadi/casadi.hpp>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace metis {

namespace detail {

inline double validate_positive_scale(double scale, const std::string &context) {
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        throw InvalidArgument(context + ": scale must be positive and finite");
    }
    return scale;
}

inline double max_finite_abs(std::optional<double> lower_bound, std::optional<double> upper_bound) {
    double bound_mag = 0.0;
    if (lower_bound.has_value() && std::isfinite(lower_bound.value())) {
        bound_mag = std::max(bound_mag, std::abs(lower_bound.value()));
    }
    if (upper_bound.has_value() && std::isfinite(upper_bound.value())) {
        bound_mag = std::max(bound_mag, std::abs(upper_bound.value()));
    }
    return bound_mag;
}

inline double suggest_scalar_scale(double init_guess, std::optional<double> lower_bound,
                                   std::optional<double> upper_bound) {
    return std::max({std::abs(init_guess), max_finite_abs(lower_bound, upper_bound), 1.0});
}

inline double suggest_vector_scale(const NumericVector &init_guess,
                                   std::optional<double> lower_bound,
                                   std::optional<double> upper_bound) {
    double init_mag = init_guess.size() > 0 ? init_guess.cwiseAbs().maxCoeff() : 0.0;
    return std::max({init_mag, max_finite_abs(lower_bound, upper_bound), 1.0});
}

inline std::vector<double> dm_to_std_vector(const casadi::DM &value) {
    return static_cast<std::vector<double>>(value);
}

inline double dm_to_scalar(const casadi::DM &value) {
    std::vector<double> elements = dm_to_std_vector(value);
    return elements.empty() ? 0.0 : elements.front();
}

inline std::vector<casadi::MX> current_assignments(const casadi::Opti &opti) {
    std::vector<casadi::MX> assignments = opti.initial();
    std::vector<casadi::MX> params = opti.value_parameters();
    assignments.insert(assignments.end(), params.begin(), params.end());
    return assignments;
}

inline double normalized_magnitude(double magnitude, double scale) {
    return std::abs(magnitude) / validate_positive_scale(scale, "normalized_magnitude");
}

inline bool nearly_equal(double a, double b) {
    double tol = 1e-12 * std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tol;
}

inline double constraint_violation(double value, double lower_bound, bool has_lower_bound,
                                   double upper_bound, bool has_upper_bound) {
    if (has_lower_bound && value < lower_bound) {
        return lower_bound - value;
    }
    if (has_upper_bound && value > upper_bound) {
        return value - upper_bound;
    }
    return 0.0;
}

inline double constraint_magnitude(double value, double lower_bound, bool has_lower_bound,
                                   double upper_bound, bool has_upper_bound) {
    double magnitude = std::abs(value);
    if (has_lower_bound) {
        magnitude = std::max(magnitude, std::abs(lower_bound));
    }
    if (has_upper_bound) {
        magnitude = std::max(magnitude, std::abs(upper_bound));
    }
    return std::max(magnitude, 1.0);
}

inline ScalingIssueLevel issue_level(double normalized, const ScalingAnalysisOptions &opts) {
    return normalized > opts.normalized_high_critical ? ScalingIssueLevel::Critical
                                                      : ScalingIssueLevel::Warning;
}

} // namespace detail

/**
 * @brief Options for variable creation
 *
 * Allows specifying category and freeze status for optimization variables.
 *
 * Example:
 * @code
 *   auto x = opti.variable(1.0, {.category = "Wing", .freeze = false});
 *   auto y = opti.variable(2.0, {.category = "Wing", .freeze = true});  // Frozen at 2.0
 * @endcode
 */
struct VariableOptions {
    std::string category = "Uncategorized"; ///< Category for grouping variables
    bool freeze = false;                    ///< If true, variable is fixed at init_guess
};

/**
 * @brief Main optimization environment class
 *
 * Wraps CasADi's Opti interface to provide Metis-native types
 * and a clean C++ API for nonlinear programming with IPOPT backend.
 *
 * Example:
 * @code
 *   metis::Opti opti;
 *   auto x = opti.variable(0.0);  // scalar, init_guess=0
 *   auto y = opti.variable(0.0);
 *   opti.minimize((1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x));
 *   auto sol = opti.solve();
 *   double x_opt = sol.value(x);  // ~1.0
 * @endcode
 *
 * Rosenbrock Benchmark:
 * @code
 *   metis::Opti opti;
 *   auto x = opti.variable(10, 0.0);  // 10 variables
 *   opti.subject_to(x >= 0);
 *   // minimize sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
 *   SymbolicScalar obj = 0;
 *   for (int i = 0; i < 9; ++i) {
 *       obj = obj + 100 * metis::pow(x(i+1) - x(i)*x(i), 2)
 *                 + metis::pow(1 - x(i), 2);
 *   }
 *   opti.minimize(obj);
 *   auto sol = opti.solve();  // All x[i] ~= 1.0
 * @endcode
 *
 * @see OptiSol for extracting results
 * @see OptiOptions for solver configuration
 * @see SweepResult for parametric sweep results
 * @see ScalingReport for scaling diagnostics
 */
class Opti {
  public:
    /**
     * @brief Construct a new optimization environment
     */
    Opti() : opti_(), categories_to_freeze_(), categories_() {}

    /**
     * @brief Construct with frozen categories
     *
     * Variables created with a category in `categories_to_freeze` will be
     * automatically frozen at their initial guess values.
     *
     * @param categories_to_freeze List of category names to freeze
     */
    explicit Opti(const std::vector<std::string> &categories_to_freeze)
        : opti_(), categories_to_freeze_(categories_to_freeze), categories_() {}

    ~Opti() = default;

    // =========================================================================
    // Decision Variables
    // =========================================================================

    /**
     * @brief Create a scalar decision variable
     *
     * @param init_guess Initial guess value (where optimizer starts)
     * @param scale Optional scale for numerical conditioning (default: |init_guess| or 1)
     * @param lower_bound Optional lower bound constraint
     * @param upper_bound Optional upper bound constraint
     * @return Symbolic scalar representing the optimization variable
     */
    SymbolicScalar variable(double init_guess = 0.0, std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt) {
        return variable(init_guess, VariableOptions{}, scale, lower_bound, upper_bound);
    }

    /**
     * @brief Create a scalar decision variable with options
     *
     * @param init_guess Initial guess value (where optimizer starts)
     * @param opts Variable options (category, freeze)
     * @param scale Optional scale for numerical conditioning (default: |init_guess| or 1)
     * @param lower_bound Optional lower bound constraint
     * @param upper_bound Optional upper bound constraint
     * @return Symbolic scalar representing the variable (or frozen parameter)
     */
    SymbolicScalar variable(double init_guess, const VariableOptions &opts,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt) {
        // Check if variable should be frozen (explicit or via category)
        bool should_freeze = opts.freeze || is_category_frozen(opts.category);

        // Determine scale
        double s = scale.has_value()
                       ? detail::validate_positive_scale(scale.value(), "Opti::variable")
                       : detail::suggest_scalar_scale(init_guess, lower_bound, upper_bound);

        SymbolicScalar scaled_var;

        if (should_freeze) {
            // Create as parameter (not optimized)
            SymbolicScalar param = opti_.parameter();
            opti_.set_value(param, init_guess);
            scaled_var = param;
        } else {
            // Create scaled variable
            SymbolicScalar raw_var = opti_.variable();
            scaled_var = s * raw_var;

            // Set initial guess
            opti_.set_initial(raw_var, init_guess / s);

            // Apply bounds if specified
            if (lower_bound.has_value()) {
                opti_.subject_to(scaled_var >= lower_bound.value());
            }
            if (upper_bound.has_value()) {
                opti_.subject_to(scaled_var <= upper_bound.value());
            }

            NumericVector init_vector(1);
            init_vector(0) = init_guess;
            register_variable_block(init_vector, opts.category, s, scale.has_value(), lower_bound,
                                    upper_bound);
        }

        // Track in category
        categories_[opts.category].push_back(scaled_var);

        return scaled_var;
    }

    /**
     * @brief Create a vector of decision variables with scalar init guess
     *
     * @param n_vars Number of variables
     * @param init_guess Initial guess applied to all elements
     * @param scale Optional scale for numerical conditioning
     * @param lower_bound Optional lower bound for all elements
     * @param upper_bound Optional upper bound for all elements
     * @return Symbolic vector (column vector) of optimization variables
     */
    SymbolicVector variable(int n_vars, double init_guess = 0.0,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt) {
        // Determine scale
        double s = scale.has_value()
                       ? detail::validate_positive_scale(scale.value(), "Opti::variable")
                       : detail::suggest_scalar_scale(init_guess, lower_bound, upper_bound);

        // Create scaled variable
        SymbolicScalar raw_var = opti_.variable(n_vars, 1);
        SymbolicScalar scaled_var = s * raw_var;

        // Set initial guess (constant vector)
        opti_.set_initial(raw_var, init_guess / s);

        // Apply bounds if specified
        if (lower_bound.has_value()) {
            opti_.subject_to(scaled_var >= lower_bound.value());
        }
        if (upper_bound.has_value()) {
            opti_.subject_to(scaled_var <= upper_bound.value());
        }

        register_variable_block(NumericVector::Constant(n_vars, init_guess), "Uncategorized", s,
                                scale.has_value(), lower_bound, upper_bound);

        return metis::to_eigen(scaled_var);
    }

    /**
     * @brief Create a vector of decision variables with per-element init guess
     *
     * @param init_guess Vector of initial guesses (size determines n_vars)
     * @param scale Optional scale for numerical conditioning
     * @param lower_bound Optional lower bound for all elements
     * @param upper_bound Optional upper bound for all elements
     * @return Symbolic vector of optimization variables
     */
    SymbolicVector variable(const NumericVector &init_guess,
                            std::optional<double> scale = std::nullopt,
                            std::optional<double> lower_bound = std::nullopt,
                            std::optional<double> upper_bound = std::nullopt) {
        int n_vars = static_cast<int>(init_guess.size());

        // Determine scale from initial guess and finite bounds
        double s = scale.has_value()
                       ? detail::validate_positive_scale(scale.value(), "Opti::variable")
                       : detail::suggest_vector_scale(init_guess, lower_bound, upper_bound);

        // Create scaled variable
        SymbolicScalar raw_var = opti_.variable(n_vars, 1);
        SymbolicScalar scaled_var = s * raw_var;

        // Set initial guess (convert Eigen to std::vector)
        std::vector<double> init_vec(init_guess.data(), init_guess.data() + init_guess.size());
        for (auto &v : init_vec) {
            v /= s;
        }
        opti_.set_initial(raw_var, init_vec);

        // Apply bounds if specified
        if (lower_bound.has_value()) {
            opti_.subject_to(scaled_var >= lower_bound.value());
        }
        if (upper_bound.has_value()) {
            opti_.subject_to(scaled_var <= upper_bound.value());
        }

        register_variable_block(init_guess, "Uncategorized", s, scale.has_value(), lower_bound,
                                upper_bound);

        return metis::to_eigen(scaled_var);
    }

    // =========================================================================
    // Parameters (fixed values during optimization)
    // =========================================================================

    /**
     * @brief Create a scalar parameter
     *
     * Parameters are fixed values that can be changed between solves.
     *
     * @param value Parameter value
     * @return Symbolic scalar representing the parameter
     */
    SymbolicScalar parameter(double value) {
        SymbolicScalar param = opti_.parameter();
        opti_.set_value(param, value);
        return param;
    }

    /**
     * @brief Create a vector parameter
     *
     * @param value Vector of parameter values
     * @return Symbolic vector representing the parameters
     */
    SymbolicVector parameter(const NumericVector &value) {
        int n = static_cast<int>(value.size());
        SymbolicScalar param = opti_.parameter(n, 1);
        std::vector<double> vals(value.data(), value.data() + value.size());
        opti_.set_value(param, vals);
        return metis::to_eigen(param);
    }

    // =========================================================================
    // Constraints
    // =========================================================================

    /**
     * @brief Add a scalar constraint
     *
     * Supports both equality and inequality constraints:
     * @code
     *   opti.subject_to(x >= 0);      // inequality
     *   opti.subject_to(x == 1);      // equality
     *   opti.subject_to(x * x + y * y <= 1);  // nonlinear
     * @endcode
     *
     * @param constraint Symbolic inequality/equality expression
     */
    void subject_to(const SymbolicScalar &constraint) { opti_.subject_to(constraint); }

    /**
     * @brief Add a scaled scalar constraint
     *
     * The provided scale is forwarded to CasADi so the solver sees
     * `constraint / linear_scale`.
     *
     * @param constraint symbolic inequality/equality expression
     * @param linear_scale positive scale factor
     */
    void subject_to(const SymbolicScalar &constraint, double linear_scale) {
        opti_.subject_to(constraint, casadi::DM(detail::validate_positive_scale(
                                         linear_scale, "Opti::subject_to")));
    }

    /**
     * @brief Add multiple constraints
     *
     * @param constraints Vector of constraint expressions
     */
    void subject_to(const std::vector<SymbolicScalar> &constraints) {
        for (const auto &c : constraints) {
            opti_.subject_to(c);
        }
    }

    /**
     * @brief Add multiple constraints with one shared linear scale
     * @param constraints vector of constraint expressions
     * @param linear_scale positive scale factor applied to all
     */
    void subject_to(const std::vector<SymbolicScalar> &constraints, double linear_scale) {
        double s = detail::validate_positive_scale(linear_scale, "Opti::subject_to");
        for (const auto &c : constraints) {
            opti_.subject_to(c, casadi::DM(s));
        }
    }

    /**
     * @brief Add constraints from initializer list
     *
     * Example:
     * @code
     *   opti.subject_to({
     *       x >= 0,
     *       y >= 0,
     *       x + y <= 10
     *   });
     * @endcode
     */
    void subject_to(std::initializer_list<SymbolicScalar> constraints) {
        for (const auto &c : constraints) {
            opti_.subject_to(c);
        }
    }

    /**
     * @brief Add an initializer-list of constraints with one shared linear scale
     * @param constraints initializer list of constraint expressions
     * @param linear_scale positive scale factor applied to all
     */
    void subject_to(std::initializer_list<SymbolicScalar> constraints, double linear_scale) {
        double s = detail::validate_positive_scale(linear_scale, "Opti::subject_to");
        for (const auto &c : constraints) {
            opti_.subject_to(c, casadi::DM(s));
        }
    }

    // ---- Scalar Constraint Helpers ----

    /**
     * @brief Apply lower bound to a scalar variable
     * @param scalar Symbolic scalar
     * @param lower_bound Lower bound
     */
    void subject_to_lower(const SymbolicScalar &scalar, double lower_bound) {
        opti_.subject_to(scalar >= lower_bound);
    }

    /**
     * @brief Apply upper bound to a scalar variable
     * @param scalar Symbolic scalar
     * @param upper_bound Upper bound
     */
    void subject_to_upper(const SymbolicScalar &scalar, double upper_bound) {
        opti_.subject_to(scalar <= upper_bound);
    }

    /**
     * @brief Apply both lower and upper bounds to a scalar variable
     * @param scalar Symbolic scalar
     * @param lower_bound Lower bound
     * @param upper_bound Upper bound
     */
    void subject_to_bounds(const SymbolicScalar &scalar, double lower_bound, double upper_bound) {
        subject_to_lower(scalar, lower_bound);
        subject_to_upper(scalar, upper_bound);
    }

    // ---- Vector Constraint Helpers ----

    /**
     * @brief Apply lower bound to all elements of a vector
     *
     * Example:
     * @code
     *   auto x = opti.variable(10, 0.0);
     *   opti.subject_to_lower(x, 0.0);  // All x[i] >= 0
     * @endcode
     *
     * @param vec Symbolic vector
     * @param lower_bound Lower bound for all elements
     */
    void subject_to_lower(const SymbolicVector &vec, double lower_bound) {
        for (Eigen::Index i = 0; i < vec.size(); ++i) {
            opti_.subject_to(vec(i) >= lower_bound);
        }
    }

    /**
     * @brief Apply upper bound to all elements of a vector
     *
     * @param vec Symbolic vector
     * @param upper_bound Upper bound for all elements
     */
    void subject_to_upper(const SymbolicVector &vec, double upper_bound) {
        for (Eigen::Index i = 0; i < vec.size(); ++i) {
            opti_.subject_to(vec(i) <= upper_bound);
        }
    }

    /**
     * @brief Apply both lower and upper bounds to all elements
     *
     * Example:
     * @code
     *   auto x = opti.variable(10, 0.5);
     *   opti.subject_to_bounds(x, 0.0, 1.0);  // 0 <= x[i] <= 1
     * @endcode
     *
     * @param vec Symbolic vector
     * @param lower_bound Lower bound for all elements
     * @param upper_bound Upper bound for all elements
     */
    void subject_to_bounds(const SymbolicVector &vec, double lower_bound, double upper_bound) {
        subject_to_lower(vec, lower_bound);
        subject_to_upper(vec, upper_bound);
    }

    // =========================================================================
    // Objective
    // =========================================================================

    /**
     * @brief Set objective to minimize
     *
     * @param objective Symbolic expression to minimize
     */
    void minimize(const SymbolicScalar &objective) { set_objective(objective, 1.0, false, false); }

    /**
     * @brief Set objective to minimize with explicit objective scaling
     *
     * The solver sees `objective / objective_scale`.
     *
     * @param objective symbolic expression to minimize
     * @param objective_scale positive scale factor
     */
    void minimize(const SymbolicScalar &objective, double objective_scale) {
        set_objective(objective, detail::validate_positive_scale(objective_scale, "Opti::minimize"),
                      true, false);
    }

    /**
     * @brief Set objective to maximize
     *
     * @param objective Symbolic expression to maximize
     */
    void maximize(const SymbolicScalar &objective) { set_objective(objective, 1.0, false, true); }

    /**
     * @brief Set objective to maximize with explicit objective scaling
     *
     * The solver sees `-objective / objective_scale`.
     *
     * @param objective symbolic expression to maximize
     * @param objective_scale positive scale factor
     */
    void maximize(const SymbolicScalar &objective, double objective_scale) {
        set_objective(objective, detail::validate_positive_scale(objective_scale, "Opti::maximize"),
                      true, true);
    }

    // =========================================================================
    // Solve
    // =========================================================================

    /**
     * @brief Solve the optimization problem
     *
     * Uses the solver specified in options (default: IPOPT).
     *
     * @param options Solver configuration (solver, max_iter, verbose, etc.)
     * @return OptiSol containing optimized values
     * @throws RuntimeError if selected solver unavailable or fails
     */
    OptiSol solve(const OptiOptions &options = {}) {
        // Verify solver availability
        if (!solver_available(options.solver)) {
            throw RuntimeError(std::string("Solver '") + solver_name(options.solver) +
                               "' is not available in this CasADi build");
        }

        casadi::Dict solver_opts;

        // Configure solver-specific options
        switch (options.solver) {
        case Solver::Snopt:
            configure_snopt_opts(solver_opts, options);
            break;
        case Solver::QpOases:
            configure_qpoases_opts(solver_opts, options);
            break;
        case Solver::Ipopt:
        default:
            configure_ipopt_opts(solver_opts, options);
            break;
        }

        // Common options
        if (options.jit) {
            solver_opts["jit"] = true;
            solver_opts["jit_options"] = casadi::Dict{{"flags", std::vector<std::string>{"-O3"}}};
        }

        // Set solver and solve
        opti_.solver(solver_name(options.solver), solver_opts);
        return OptiSol(opti_.solve());
    }

    /**
     * @brief Perform parametric sweep over a parameter
     *
     * Solves the optimization problem for each value in the sweep range,
     * automatically warm-starting subsequent solves from the previous solution.
     *
     * Example:
     * @code
     *   auto rho = opti.parameter(1.225);  // Air density
     *   // ... define problem using rho ...
     *
     *   auto result = opti.solve_sweep(rho, {1.0, 1.1, 1.2, 1.3});
     *   for (size_t i = 0; i < result.size(); ++i) {
     *       if (result.converged[i]) {
     *           std::cout << "rho=" << result.param_values[i]
     *                     << " x*=" << result.solutions[i]->value(x) << "\n";
     *       }
     *   }
     * @endcode
     *
     * @param param Parameter to sweep (from opti.parameter())
     * @param values Vector of parameter values to try
     * @param options Solver options (applied to all solves)
     * @return SweepResult containing all solutions
     */
    SweepResult solve_sweep(const SymbolicScalar &param, const std::vector<double> &values,
                            const OptiOptions &options = {}) {
        // Verify solver availability
        if (!solver_available(options.solver)) {
            throw RuntimeError(std::string("Solver '") + solver_name(options.solver) +
                               "' is not available in this CasADi build");
        }

        SweepResult result;
        result.param_values = values;
        result.all_converged = true;

        // Configure solver once
        casadi::Dict solver_opts;

        switch (options.solver) {
        case Solver::Snopt:
            configure_snopt_opts(solver_opts, options);
            break;
        case Solver::QpOases:
            configure_qpoases_opts(solver_opts, options);
            break;
        case Solver::Ipopt:
        default:
            configure_ipopt_opts(solver_opts, options);
            // Enable warm-start for parameter sweeps
            solver_opts["ipopt.warm_start_init_point"] = "yes";
            break;
        }

        opti_.solver(solver_name(options.solver), solver_opts);

        for (size_t idx = 0; idx < values.size(); ++idx) {
            double val = values[idx];
            opti_.set_value(param, val);

            try {
                auto sol = opti_.solve();
                result.iterations.push_back(sol.stats().count("iter_count")
                                                ? static_cast<int>(sol.stats().at("iter_count"))
                                                : -1);
                result.solutions.emplace_back(sol);
                result.converged.push_back(true);
                result.errors.emplace_back();
            } catch (const std::exception &e) {
                result.all_converged = false;
                result.solutions.emplace_back(std::nullopt);
                result.converged.push_back(false);
                result.errors.push_back(e.what());
                result.iterations.push_back(-1);
            }
        }

        return result;
    }
    // =========================================================================
    // Derivative Helpers (for trajectory optimization)
    // =========================================================================

    /**
     * @brief Create a derivative variable constrained by integration
     *
     * Returns a new variable that is constrained to be the derivative of
     * `variable` with respect to `with_respect_to`. This is the core
     * mechanism for trajectory optimization via direct collocation.
     *
     * Example:
     * @code
     *   NumericVector time = metis::linspace(0, 1, 100);
     *   auto position = opti.variable(100, 0.0);
     *   auto velocity = opti.derivative_of(position, time, 0.0);
     *   // velocity is now constrained: d(position)/dt = velocity
     * @endcode
     *
     * @param var The quantity to differentiate
     * @param with_respect_to Independent variable (e.g., time array)
     * @param derivative_init_guess Initial guess for derivative values
     * @param method Integration method: "trapezoidal", "forward_euler", "backward_euler"
     * @return Symbolic vector representing the derivative
     */
    SymbolicVector derivative_of(const SymbolicVector &var, const NumericVector &with_respect_to,
                                 double derivative_init_guess,
                                 const std::string &method = "trapezoidal") {
        int n = static_cast<int>(var.size());
        if (n != static_cast<int>(with_respect_to.size())) {
            throw InvalidArgument(
                "derivative_of: variable and with_respect_to must have same size");
        }

        // Create derivative variable
        auto deriv = variable(n, derivative_init_guess);

        // Add integration constraints
        constrain_derivative(deriv, var, with_respect_to, method);

        return deriv;
    }

    /**
     * @brief Constrain an existing variable to be a derivative
     *
     * Adds constraints: d(variable)/d(with_respect_to) == derivative
     *
     * @param derivative The derivative variable
     * @param var The variable being differentiated
     * @param with_respect_to Independent variable (e.g., time)
     * @param method Integration method (string or IntegrationMethod enum)
     */
    void constrain_derivative(const SymbolicVector &derivative, const SymbolicVector &var,
                              const NumericVector &with_respect_to,
                              const std::string &method = "trapezoidal") {
        constrain_derivative(derivative, var, with_respect_to, parse_integration_method(method));
    }

    /**
     * @brief Constrain an existing variable to be a derivative (enum version)
     */
    void constrain_derivative(const SymbolicVector &derivative, const SymbolicVector &var,
                              const NumericVector &with_respect_to, IntegrationMethod method) {
        int n = static_cast<int>(var.size());

        // Use metis::diff from Calculus.hpp
        NumericVector dt = metis::diff(with_respect_to);

        switch (method) {
        case IntegrationMethod::Trapezoidal:
        case IntegrationMethod::Midpoint:
            // Second-order: x[i+1] - x[i] = 0.5 * (xdot[i] + xdot[i+1]) * dt[i]
            for (int i = 0; i < n - 1; ++i) {
                SymbolicScalar lhs = var(i + 1) - var(i);
                SymbolicScalar rhs = 0.5 * (derivative(i) + derivative(i + 1)) * dt(i);
                opti_.subject_to(lhs == rhs);
            }
            break;

        case IntegrationMethod::ForwardEuler:
            // First-order forward: x[i+1] - x[i] = xdot[i] * dt[i]
            for (int i = 0; i < n - 1; ++i) {
                SymbolicScalar lhs = var(i + 1) - var(i);
                SymbolicScalar rhs = derivative(i) * dt(i);
                opti_.subject_to(lhs == rhs);
            }
            break;

        case IntegrationMethod::BackwardEuler:
            // First-order backward: x[i+1] - x[i] = xdot[i+1] * dt[i]
            for (int i = 0; i < n - 1; ++i) {
                SymbolicScalar lhs = var(i + 1) - var(i);
                SymbolicScalar rhs = derivative(i + 1) * dt(i);
                opti_.subject_to(lhs == rhs);
            }
            break;
        }
    }

    // =========================================================================
    // Access
    // =========================================================================

    /**
     * @brief Diagnose variable, objective, and constraint scaling at the current initial point
     * @param opts thresholds controlling warning/critical classification
     * @return scaling report with per-variable and per-constraint metadata
     * @see ScalingReport for the returned structure
     */
    ScalingReport analyze_scaling(const ScalingAnalysisOptions &opts = {}) const {
        ScalingReport report;
        report.summary.variable_blocks = static_cast<int>(variable_records_.size());
        report.summary.scalar_constraints = static_cast<int>(opti_.ng());

        if (!variable_records_.empty()) {
            report.summary.min_variable_scale = variable_records_.front().scale;
            report.summary.max_variable_scale = variable_records_.front().scale;
        }

        for (size_t i = 0; i < variable_records_.size(); ++i) {
            const auto &record = variable_records_[i];
            VariableScalingInfo info;
            info.block_index = static_cast<int>(i);
            info.size = record.size;
            info.category = record.category;
            info.frozen = false;
            info.user_supplied_scale = record.user_supplied_scale;
            info.scale = record.scale;
            info.init_abs_mean = record.init_abs_mean;
            info.init_abs_max = record.init_abs_max;
            info.normalized_init_abs_mean = record.init_abs_mean / record.scale;
            info.normalized_init_abs_max = record.init_abs_max / record.scale;
            info.lower_bound = record.lower_bound;
            info.upper_bound = record.upper_bound;
            info.suggested_scale = record.suggested_scale;
            report.variables.push_back(info);

            report.summary.min_variable_scale =
                std::min(report.summary.min_variable_scale, record.scale);
            report.summary.max_variable_scale =
                std::max(report.summary.max_variable_scale, record.scale);

            if (record.init_abs_max > 0.0 &&
                info.normalized_init_abs_max < opts.normalized_low_warn) {
                report.issues.push_back(
                    {ScalingIssueLevel::Warning, ScalingIssueKind::Variable, static_cast<int>(i),
                     "variable[" + std::to_string(i) + "]",
                     "initial guess is much smaller than the applied variable scale",
                     record.init_abs_max, record.scale, info.normalized_init_abs_max,
                     record.suggested_scale});
            } else if (info.normalized_init_abs_max > opts.normalized_high_warn) {
                report.issues.push_back(
                    {detail::issue_level(info.normalized_init_abs_max, opts),
                     ScalingIssueKind::Variable, static_cast<int>(i),
                     "variable[" + std::to_string(i) + "]",
                     "initial guess is much larger than the applied variable scale",
                     record.init_abs_max, record.scale, info.normalized_init_abs_max,
                     record.suggested_scale});
            }
        }

        if (!variable_records_.empty()) {
            report.summary.variable_scale_ratio =
                report.summary.max_variable_scale / report.summary.min_variable_scale;
            if (report.summary.variable_scale_ratio > opts.variable_scale_ratio_warn) {
                report.issues.push_back(
                    {ScalingIssueLevel::Warning, ScalingIssueKind::Summary, -1, "variables",
                     "decision-variable scales span many orders of magnitude",
                     report.summary.max_variable_scale, report.summary.min_variable_scale,
                     report.summary.variable_scale_ratio, report.summary.max_variable_scale});
            }
        }

        std::vector<casadi::MX> assignments = detail::current_assignments(opti_);

        if (objective_expr_.has_value()) {
            report.objective.configured = true;
            report.objective.maximize = objective_is_maximize_;
            report.objective.user_supplied_scale = objective_scale_user_supplied_;
            report.objective.scale = objective_scale_;
            report.objective.value_at_initial =
                detail::dm_to_scalar(opti_.value(objective_expr_.value(), assignments));
            report.objective.normalized_value =
                std::abs(report.objective.value_at_initial) / objective_scale_;
            report.objective.suggested_scale =
                std::max(std::abs(report.objective.value_at_initial), 1.0);

            if (report.objective.normalized_value > opts.normalized_high_warn) {
                report.issues.push_back(
                    {detail::issue_level(report.objective.normalized_value, opts),
                     ScalingIssueKind::Objective, 0, "objective",
                     "objective magnitude is large relative to the applied objective scale",
                     std::abs(report.objective.value_at_initial), objective_scale_,
                     report.objective.normalized_value, report.objective.suggested_scale});
            }
        }

        if (opti_.ng() > 0) {
            std::vector<double> g_values =
                detail::dm_to_std_vector(opti_.value(opti_.g(), assignments));
            std::vector<double> lbg_values =
                detail::dm_to_std_vector(opti_.value(opti_.lbg(), assignments));
            std::vector<double> ubg_values =
                detail::dm_to_std_vector(opti_.value(opti_.ubg(), assignments));
            std::vector<double> g_scales = detail::dm_to_std_vector(opti_.g_linear_scale());

            for (int i = 0; i < static_cast<int>(g_values.size()); ++i) {
                ConstraintScalingInfo info;
                info.row = i;
                info.value_at_initial = g_values.at(i);
                info.scale = i < static_cast<int>(g_scales.size()) ? g_scales.at(i) : 1.0;
                info.lower_bound = i < static_cast<int>(lbg_values.size()) ? lbg_values.at(i) : 0.0;
                info.upper_bound = i < static_cast<int>(ubg_values.size()) ? ubg_values.at(i) : 0.0;
                info.has_lower_bound =
                    i < static_cast<int>(lbg_values.size()) && std::isfinite(info.lower_bound);
                info.has_upper_bound =
                    i < static_cast<int>(ubg_values.size()) && std::isfinite(info.upper_bound);
                info.equality = info.has_lower_bound && info.has_upper_bound &&
                                detail::nearly_equal(info.lower_bound, info.upper_bound);
                info.normalized_magnitude =
                    detail::constraint_magnitude(info.value_at_initial, info.lower_bound,
                                                 info.has_lower_bound, info.upper_bound,
                                                 info.has_upper_bound) /
                    info.scale;
                info.normalized_violation =
                    detail::constraint_violation(info.value_at_initial, info.lower_bound,
                                                 info.has_lower_bound, info.upper_bound,
                                                 info.has_upper_bound) /
                    info.scale;
                info.suggested_scale = detail::constraint_magnitude(
                    info.value_at_initial, info.lower_bound, info.has_lower_bound, info.upper_bound,
                    info.has_upper_bound);
                report.constraints.push_back(info);

                if (info.normalized_magnitude > opts.normalized_high_warn) {
                    report.issues.push_back(
                        {detail::issue_level(info.normalized_magnitude, opts),
                         ScalingIssueKind::Constraint, i, "constraint[" + std::to_string(i) + "]",
                         "constraint magnitude or bound magnitude is large relative to its scale",
                         info.suggested_scale, info.scale, info.normalized_magnitude,
                         info.suggested_scale});
                }
            }
        }

        if (opti_.nx() > 0 && opti_.ng() > 0) {
            casadi::Sparsity jac_sp = casadi::MX::jacobian_sparsity(opti_.g(), opti_.x());
            report.summary.jacobian_nnz = jac_sp.nnz();
            report.summary.jacobian_density =
                static_cast<double>(jac_sp.nnz()) /
                static_cast<double>(std::max<casadi_int>(1, opti_.nx() * opti_.ng()));
        }

        return report;
    }

    /**
     * @brief Access the underlying CasADi Opti object
     *
     * For advanced usage when CasADi-specific features are needed.
     *
     * @return reference to the CasADi Opti object
     */
    const casadi::Opti &casadi_opti() const { return opti_; }

    /**
     * @brief Set initial guess for a symbolic expression
     *
     * Forwards to CasADi's set_initial for element-level initial guesses
     * (e.g. matrix slices from transcription methods).
     *
     * @param x Symbolic expression
     * @param v Initial value
     */
    void set_initial(const casadi::MX &x, const casadi::DM &v) { opti_.set_initial(x, v); }

    // =========================================================================
    // Category & Freezing
    // =========================================================================

    /**
     * @brief Get all variables in a category
     *
     * @param category Category name
     * @return Vector of symbolic scalars in that category (empty if not found)
     */
    std::vector<SymbolicScalar> get_category(const std::string &category) const {
        auto it = categories_.find(category);
        if (it != categories_.end()) {
            return it->second;
        }
        return {};
    }

    /**
     * @brief Get all registered category names
     * @return Vector of category names
     */
    std::vector<std::string> get_category_names() const {
        std::vector<std::string> names;
        names.reserve(categories_.size());
        for (const auto &[name, _] : categories_) {
            names.push_back(name);
        }
        return names;
    }

    /**
     * @brief Check if a category is marked for freezing
     * @param category Category name
     * @return True if the category was specified in constructor to be frozen
     */
    bool is_category_frozen(const std::string &category) const {
        return std::find(categories_to_freeze_.begin(), categories_to_freeze_.end(), category) !=
               categories_to_freeze_.end();
    }

  private:
    struct VariableRecord {
        int size = 0;
        std::string category = "Uncategorized";
        bool user_supplied_scale = false;
        double scale = 1.0;
        double init_abs_mean = 0.0;
        double init_abs_max = 0.0;
        std::optional<double> lower_bound;
        std::optional<double> upper_bound;
        double suggested_scale = 1.0;
    };

    casadi::Opti opti_;
    std::vector<std::string> categories_to_freeze_;
    std::map<std::string, std::vector<SymbolicScalar>> categories_;
    std::vector<VariableRecord> variable_records_;
    std::optional<SymbolicScalar> objective_expr_;
    double objective_scale_ = 1.0;
    bool objective_scale_user_supplied_ = false;
    bool objective_is_maximize_ = false;

    // =========================================================================
    // Solver Configuration Helpers
    // =========================================================================

    void configure_ipopt_opts(casadi::Dict &solver_opts, const OptiOptions &options) {
        solver_opts["ipopt.max_iter"] = options.max_iter;
        solver_opts["ipopt.max_cpu_time"] = options.max_cpu_time;
        solver_opts["ipopt.tol"] = options.tol;
        solver_opts["ipopt.mu_strategy"] = options.mu_strategy;
        solver_opts["ipopt.sb"] = "yes"; // Suppress banner

        if (options.verbose) {
            solver_opts["ipopt.print_level"] = 5;
        } else {
            solver_opts["print_time"] = false;
            solver_opts["ipopt.print_level"] = 0;
        }

        if (options.detect_simple_bounds) {
            solver_opts["detect_simple_bounds"] = true;
        }
    }

    void configure_snopt_opts(casadi::Dict &solver_opts, const OptiOptions &options) {
        const auto &snopt = options.snopt_opts;

        // SNOPT option names in CasADi use underscores
        solver_opts["snopt.Major_iterations_limit"] = snopt.major_iterations_limit;
        solver_opts["snopt.Minor_iterations_limit"] = snopt.minor_iterations_limit;
        solver_opts["snopt.Major_optimality_tolerance"] = snopt.major_optimality_tolerance;
        solver_opts["snopt.Major_feasibility_tolerance"] = snopt.major_feasibility_tolerance;

        if (options.verbose) {
            solver_opts["snopt.Major_print_level"] = 1;
            solver_opts["snopt.Minor_print_level"] = 1;
        } else {
            solver_opts["print_time"] = false;
            solver_opts["snopt.Major_print_level"] = snopt.print_level;
            solver_opts["snopt.Minor_print_level"] = 0;
        }
    }

    void configure_qpoases_opts(casadi::Dict &solver_opts, const OptiOptions &options) {
        solver_opts["qpoases.nWSR"] = options.max_iter;
        solver_opts["qpoases.CPUtime"] = options.max_cpu_time;

        if (!options.verbose) {
            solver_opts["print_time"] = false;
            solver_opts["qpoases.printLevel"] = "none";
        }
    }

    void register_variable_block(const NumericVector &init_guess, const std::string &category,
                                 double scale, bool user_supplied_scale,
                                 std::optional<double> lower_bound,
                                 std::optional<double> upper_bound) {
        VariableRecord record;
        record.size = static_cast<int>(init_guess.size());
        record.category = category;
        record.user_supplied_scale = user_supplied_scale;
        record.scale = detail::validate_positive_scale(scale, "Opti::variable");
        record.init_abs_mean = init_guess.size() > 0 ? init_guess.cwiseAbs().mean() : 0.0;
        record.init_abs_max = init_guess.size() > 0 ? init_guess.cwiseAbs().maxCoeff() : 0.0;
        record.lower_bound = lower_bound;
        record.upper_bound = upper_bound;
        record.suggested_scale =
            init_guess.size() == 1
                ? detail::suggest_scalar_scale(init_guess(0), lower_bound, upper_bound)
                : detail::suggest_vector_scale(init_guess, lower_bound, upper_bound);
        variable_records_.push_back(record);
    }

    void set_objective(const SymbolicScalar &objective, double scale, bool user_supplied_scale,
                       bool maximize) {
        objective_expr_ = objective;
        objective_scale_ = detail::validate_positive_scale(scale, "Opti objective");
        objective_scale_user_supplied_ = user_supplied_scale;
        objective_is_maximize_ = maximize;
        opti_.minimize((maximize ? -objective : objective) / objective_scale_);
    }
};

} // namespace metis
