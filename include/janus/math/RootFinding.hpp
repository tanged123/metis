#pragma once
/**
 * @file RootFinding.hpp
 * @brief Nonlinear root finding and implicit function solvers
 * @see AutoDiff.hpp
 */

#include "metis/core/Function.hpp"
#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include <casadi/casadi.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace metis {

/**
 * @brief Numeric nonlinear solver strategy selection.
 */
enum class RootSolveStrategy {
    Auto,               ///< Trust-region LM, then line-search Newton, Broyden, pseudo-transient
    TrustRegionNewton,  ///< Levenberg-Marquardt style trust-region Newton
    LineSearchNewton,   ///< Exact-Jacobian Newton with backtracking line search
    QuasiNewtonBroyden, ///< Broyden updates after one exact Jacobian evaluation
    PseudoTransientContinuation, ///< Backward-Euler pseudo-transient continuation
};

/**
 * @brief Numeric nonlinear solver method actually used.
 */
enum class RootSolveMethod {
    None,                        ///< No method selected yet
    TrustRegionNewton,           ///< Levenberg-Marquardt style trust-region Newton
    LineSearchNewton,            ///< Exact-Jacobian Newton with backtracking line search
    QuasiNewtonBroyden,          ///< Broyden updates after one exact Jacobian evaluation
    PseudoTransientContinuation, ///< Backward-Euler pseudo-transient continuation
};

/**
 * @brief Options for root finding algorithms.
 *
 * Numeric `rootfinder<double>()` uses Metis' own globalization stack. Symbolic
 * `rootfinder<SymbolicScalar>()` and `create_implicit_function()` still rely on
 * CasADi's differentiable `newton` rootfinder, so only the CasADi-compatible
 * subset of options affects those paths.
 */
struct RootFinderOptions {
    double abstol = 1e-10;              // Absolute tolerance on residual infinity norm
    double abstolStep = 1e-10;          // Stall threshold on step infinity norm
    int max_iter = 50;                  // Total numeric iteration budget across all stages
    bool line_search = true;            // Include line-search Newton in Auto fallback stack
    bool verbose = false;               // Print numeric solver stage progress
    std::string linear_solver = "qr";   // Linear solver used by CasADi rootfinder
    casadi::Dict linear_solver_options; // Options forwarded to the linear solver
    RootSolveStrategy strategy = RootSolveStrategy::Auto;
    double trust_region_initial_damping = 1e-3;
    double trust_region_damping_increase = 4.0;
    double trust_region_damping_decrease = 0.25;
    double line_search_contraction = 0.5;
    double line_search_sufficient_decrease = 1e-4;
    int max_backtracking_steps = 12;
    int broyden_jacobian_refresh = 8; // 0 disables exact Jacobian refreshes
    double pseudo_transient_dt0 = 1e-2;
    double pseudo_transient_dt_growth = 2.0;
    double pseudo_transient_dt_max = 1e6;
};

/**
 * @brief Options for building differentiable implicit solve wrappers.
 */
struct ImplicitFunctionOptions {
    int implicit_input_index = 0;  // Input slot containing the unknown state
    int implicit_output_index = 0; // Output slot containing the residual equation
};

/**
 * @brief Result of a root finding operation.
 */
template <typename Scalar> struct RootResult {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x; // Solution
    int iterations = 0;                         // Total numeric iterations used
    bool converged = false;                     // Whether solution converged
    RootSolveMethod method = RootSolveMethod::None;
    double residual_norm = std::numeric_limits<double>::infinity();
    double step_norm = 0.0;
    std::string message = "";
};

namespace detail {

inline std::string method_name(RootSolveMethod method) {
    switch (method) {
    case RootSolveMethod::None:
        return "none";
    case RootSolveMethod::TrustRegionNewton:
        return "trust-region Newton";
    case RootSolveMethod::LineSearchNewton:
        return "line-search Newton";
    case RootSolveMethod::QuasiNewtonBroyden:
        return "quasi-Newton Broyden";
    case RootSolveMethod::PseudoTransientContinuation:
        return "pseudo-transient continuation";
    }
    throw InvalidArgument("rootfinder: unsupported solve method");
}

inline RootSolveMethod strategy_to_method(RootSolveStrategy strategy) {
    switch (strategy) {
    case RootSolveStrategy::Auto:
        return RootSolveMethod::None;
    case RootSolveStrategy::TrustRegionNewton:
        return RootSolveMethod::TrustRegionNewton;
    case RootSolveStrategy::LineSearchNewton:
        return RootSolveMethod::LineSearchNewton;
    case RootSolveStrategy::QuasiNewtonBroyden:
        return RootSolveMethod::QuasiNewtonBroyden;
    case RootSolveStrategy::PseudoTransientContinuation:
        return RootSolveMethod::PseudoTransientContinuation;
    }
    throw InvalidArgument("rootfinder: unsupported solve strategy");
}

inline std::string unique_name(const std::string &prefix) {
    static std::atomic<std::uint64_t> counter{0};
    return prefix + "_" + std::to_string(counter.fetch_add(1));
}

inline void validate_root_options(const RootFinderOptions &opts, const std::string &context) {
    if (opts.abstol <= 0.0) {
        throw InvalidArgument(context + ": abstol must be positive");
    }
    if (opts.abstolStep <= 0.0) {
        throw InvalidArgument(context + ": abstolStep must be positive");
    }
    if (opts.max_iter <= 0) {
        throw InvalidArgument(context + ": max_iter must be positive");
    }
    if (opts.trust_region_initial_damping <= 0.0) {
        throw InvalidArgument(context + ": trust_region_initial_damping must be positive");
    }
    if (opts.trust_region_damping_increase <= 1.0) {
        throw InvalidArgument(context + ": trust_region_damping_increase must exceed 1");
    }
    if (opts.trust_region_damping_decrease <= 0.0 || opts.trust_region_damping_decrease >= 1.0) {
        throw InvalidArgument(context + ": trust_region_damping_decrease must lie in (0, 1)");
    }
    if (opts.line_search_contraction <= 0.0 || opts.line_search_contraction >= 1.0) {
        throw InvalidArgument(context + ": line_search_contraction must lie in (0, 1)");
    }
    if (opts.line_search_sufficient_decrease <= 0.0 ||
        opts.line_search_sufficient_decrease >= 1.0) {
        throw InvalidArgument(context + ": line_search_sufficient_decrease must lie in (0, 1)");
    }
    if (opts.max_backtracking_steps <= 0) {
        throw InvalidArgument(context + ": max_backtracking_steps must be positive");
    }
    if (opts.broyden_jacobian_refresh < 0) {
        throw InvalidArgument(context + ": broyden_jacobian_refresh cannot be negative");
    }
    if (opts.pseudo_transient_dt0 <= 0.0) {
        throw InvalidArgument(context + ": pseudo_transient_dt0 must be positive");
    }
    if (opts.pseudo_transient_dt_growth <= 1.0) {
        throw InvalidArgument(context + ": pseudo_transient_dt_growth must exceed 1");
    }
    if (opts.pseudo_transient_dt_max < opts.pseudo_transient_dt0) {
        throw InvalidArgument(context +
                              ": pseudo_transient_dt_max must be at least pseudo_transient_dt0");
    }
}

inline void validate_root_problem(const casadi::Function &f_casadi, const std::string &context) {
    if (f_casadi.n_in() != 1 || f_casadi.n_out() != 1) {
        throw InvalidArgument(context + ": Function F must have exactly 1 input and 1 output");
    }

    const auto input_sparsity = f_casadi.sparsity_in(0);
    const auto output_sparsity = f_casadi.sparsity_out(0);
    if (!input_sparsity.is_dense() || !input_sparsity.is_column()) {
        throw InvalidArgument(context + ": input must be a dense column vector");
    }
    if (!output_sparsity.is_dense() || !output_sparsity.is_column()) {
        throw InvalidArgument(context + ": output must be a dense column vector residual");
    }
    if (f_casadi.nnz_in(0) != f_casadi.nnz_out(0)) {
        throw InvalidArgument(context + ": input and output dimensions must match");
    }
}

// Helper to convert options to CasADi dictionary
inline casadi::Dict opts_to_dict(const RootFinderOptions &opts) {
    casadi::Dict d;
    d["abstol"] = opts.abstol;
    d["abstolStep"] = opts.abstolStep;
    d["max_iter"] = opts.max_iter;
    d["linear_solver"] = opts.linear_solver;
    if (!opts.linear_solver_options.empty()) {
        d["linear_solver_options"] = opts.linear_solver_options;
    }
    if (opts.verbose) {
        d["verbose"] = true;
        d["print_in"] = true;
        d["print_out"] = true;
    }
    return d;
}

inline casadi::DM vector_to_dm(const Eigen::VectorXd &x) {
    casadi::DM out(x.size(), 1);
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        out(static_cast<int>(i), 0) = x(i);
    }
    return out;
}

inline Eigen::VectorXd dm_to_vector(const casadi::DM &x) {
    const std::vector<double> elements = static_cast<std::vector<double>>(x);
    return Eigen::Map<const Eigen::VectorXd>(elements.data(),
                                             static_cast<Eigen::Index>(elements.size()));
}

inline Eigen::MatrixXd dm_to_matrix(const casadi::DM &x) {
    const std::vector<double> elements = static_cast<std::vector<double>>(x);
    return Eigen::Map<const Eigen::MatrixXd>(elements.data(), x.size1(), x.size2());
}

inline std::string implicit_function_name(const casadi::Function &g_casadi) {
    return g_casadi.name() + "_implicit";
}

inline void validate_implicit_problem(const casadi::Function &g_casadi,
                                      const Eigen::VectorXd &x_guess,
                                      const ImplicitFunctionOptions &implicit_opts) {
    if (g_casadi.n_in() == 0 || g_casadi.n_out() == 0) {
        throw InvalidArgument(
            "create_implicit_function: G must have at least one input and one output");
    }

    if (implicit_opts.implicit_input_index < 0 ||
        implicit_opts.implicit_input_index >= g_casadi.n_in()) {
        throw InvalidArgument("create_implicit_function: implicit_input_index out of range");
    }
    if (implicit_opts.implicit_output_index < 0 ||
        implicit_opts.implicit_output_index >= g_casadi.n_out()) {
        throw InvalidArgument("create_implicit_function: implicit_output_index out of range");
    }

    const auto unknown_sparsity = g_casadi.sparsity_in(implicit_opts.implicit_input_index);
    if (!unknown_sparsity.is_dense() || !unknown_sparsity.is_column()) {
        throw InvalidArgument(
            "create_implicit_function: implicit input must be a dense column vector");
    }

    const auto residual_sparsity = g_casadi.sparsity_out(implicit_opts.implicit_output_index);
    if (!residual_sparsity.is_dense() || !residual_sparsity.is_column()) {
        throw InvalidArgument(
            "create_implicit_function: implicit output must be a dense column vector residual");
    }

    const auto n_unknown =
        static_cast<Eigen::Index>(g_casadi.nnz_in(implicit_opts.implicit_input_index));
    const auto n_residual =
        static_cast<Eigen::Index>(g_casadi.nnz_out(implicit_opts.implicit_output_index));
    if (n_unknown != n_residual) {
        throw InvalidArgument(
            "create_implicit_function: implicit input and output dimensions must match");
    }
    if (x_guess.size() != n_unknown) {
        throw InvalidArgument("create_implicit_function: x_guess size must match the implicit "
                              "input dimension");
    }
}

inline Eigen::VectorXd solve_linear_system(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
    return A.colPivHouseholderQr().solve(b);
}

inline bool all_finite(const Eigen::VectorXd &x) { return x.array().isFinite().all(); }

inline bool all_finite(const Eigen::MatrixXd &x) { return x.array().isFinite().all(); }

struct NumericState {
    Eigen::VectorXd x;
    Eigen::VectorXd residual;
    Eigen::MatrixXd jacobian;
    double residual_norm = std::numeric_limits<double>::infinity();
    double merit = std::numeric_limits<double>::infinity();
};

struct StageOutcome {
    RootSolveMethod method = RootSolveMethod::None;
    NumericState state;
    int iterations = 0;
    bool converged = false;
    double step_norm = 0.0;
    std::string message;
};

inline Eigen::VectorXd evaluate_residual_only(const casadi::Function &residual_fn,
                                              const Eigen::VectorXd &x) {
    const std::vector<casadi::DM> residual_dm =
        residual_fn(std::vector<casadi::DM>{vector_to_dm(x)});
    if (residual_dm.size() != 1u) {
        throw MetisError("rootfinder: residual function must return exactly one output");
    }
    return dm_to_vector(residual_dm.front());
}

inline NumericState evaluate_state(const casadi::Function &residual_fn,
                                   const casadi::Function &jacobian_fn, const Eigen::VectorXd &x,
                                   const std::string &context) {
    NumericState state;
    state.x = x;
    state.residual = evaluate_residual_only(residual_fn, x);

    const std::vector<casadi::DM> jacobian_dm =
        jacobian_fn(std::vector<casadi::DM>{vector_to_dm(x)});
    if (jacobian_dm.size() != 1u) {
        throw MetisError(context + ": Jacobian function must return exactly one output");
    }
    state.jacobian = dm_to_matrix(jacobian_dm.front());
    state.residual_norm = state.residual.lpNorm<Eigen::Infinity>();
    state.merit = 0.5 * state.residual.squaredNorm();

    if (!all_finite(state.residual) || !all_finite(state.jacobian) ||
        !std::isfinite(state.residual_norm) || !std::isfinite(state.merit)) {
        throw MetisError(context + ": residual/Jacobian evaluation produced non-finite values");
    }
    return state;
}

inline void maybe_log(const RootFinderOptions &opts, const std::string &message) {
    if (opts.verbose) {
        std::cout << "[rootfinder] " << message << "\n";
    }
}

inline bool is_converged(const NumericState &state, const RootFinderOptions &opts) {
    return state.residual_norm <= opts.abstol;
}

inline StageOutcome solve_trust_region(const casadi::Function &residual_fn,
                                       const casadi::Function &jacobian_fn,
                                       const NumericState &start, const RootFinderOptions &opts,
                                       int max_iterations) {
    StageOutcome out;
    out.method = RootSolveMethod::TrustRegionNewton;
    out.state = start;
    if (is_converged(start, opts) || max_iterations <= 0) {
        out.converged = is_converged(start, opts);
        out.message = out.converged ? "initial iterate satisfies tolerance"
                                    : "no trust-region iterations remaining";
        return out;
    }

    NumericState current = start;
    double lambda = opts.trust_region_initial_damping;
    constexpr double min_lambda = 1e-12;
    constexpr double max_lambda = 1e12;

    for (int iter = 0; iter < max_iterations; ++iter) {
        const Eigen::VectorXd gradient = current.jacobian.transpose() * current.residual;
        Eigen::MatrixXd normal_matrix = current.jacobian.transpose() * current.jacobian;
        normal_matrix.diagonal().array() += lambda;

        const Eigen::VectorXd step = solve_linear_system(normal_matrix, -gradient);
        out.iterations += 1;
        out.step_norm = step.lpNorm<Eigen::Infinity>();

        if (!all_finite(step)) {
            out.message = "trust-region Newton produced a non-finite step";
            break;
        }
        if (out.step_norm <= opts.abstolStep) {
            out.message = "trust-region Newton stalled on a tiny step";
            break;
        }

        const double predicted = -gradient.dot(step) - 0.5 * step.dot(normal_matrix * step);
        if (!(predicted > 0.0) || !std::isfinite(predicted)) {
            lambda = std::min(max_lambda, lambda * opts.trust_region_damping_increase);
            out.message = "trust-region Newton could not predict a decrease";
            continue;
        }

        try {
            NumericState candidate =
                evaluate_state(residual_fn, jacobian_fn, current.x + step, "rootfinder");
            const double actual = current.merit - candidate.merit;
            const double rho = actual / predicted;

            if (actual > 0.0 && rho > 1e-4) {
                current = std::move(candidate);
                out.state = current;
                out.message = "trust-region Newton accepted a step";

                if (rho > 0.75) {
                    lambda = std::max(min_lambda, lambda * opts.trust_region_damping_decrease);
                } else if (rho < 0.25) {
                    lambda = std::min(max_lambda, lambda * opts.trust_region_damping_increase);
                }

                if (is_converged(current, opts)) {
                    out.converged = true;
                    out.message = "converged with trust-region Newton";
                    return out;
                }
            } else {
                lambda = std::min(max_lambda, lambda * opts.trust_region_damping_increase);
                out.message = "trust-region Newton rejected a step";
            }
        } catch (const std::exception &) {
            lambda = std::min(max_lambda, lambda * opts.trust_region_damping_increase);
            out.message = "trust-region Newton rejected an invalid trial step";
        }
    }

    if (out.message.empty()) {
        out.message = "trust-region Newton exhausted its iteration budget";
    }
    return out;
}

inline StageOutcome solve_line_search(const casadi::Function &residual_fn,
                                      const casadi::Function &jacobian_fn,
                                      const NumericState &start, const RootFinderOptions &opts,
                                      int max_iterations) {
    StageOutcome out;
    out.method = RootSolveMethod::LineSearchNewton;
    out.state = start;
    if (is_converged(start, opts) || max_iterations <= 0) {
        out.converged = is_converged(start, opts);
        out.message = out.converged ? "initial iterate satisfies tolerance"
                                    : "no line-search iterations remaining";
        return out;
    }

    NumericState current = start;
    for (int iter = 0; iter < max_iterations; ++iter) {
        const Eigen::VectorXd gradient = current.jacobian.transpose() * current.residual;
        Eigen::VectorXd step = solve_linear_system(current.jacobian, -current.residual);
        if (gradient.dot(step) >= 0.0) {
            step = -gradient;
        }

        out.iterations += 1;
        if (!all_finite(step)) {
            out.message = "line-search Newton produced a non-finite step";
            break;
        }
        const double full_step_norm = step.lpNorm<Eigen::Infinity>();
        if (full_step_norm <= opts.abstolStep) {
            out.step_norm = full_step_norm;
            out.message = "line-search Newton stalled on a tiny step";
            break;
        }

        double alpha = 1.0;
        bool accepted = false;
        const double directional_derivative = gradient.dot(step);
        for (int bt = 0; bt < opts.max_backtracking_steps; ++bt) {
            try {
                NumericState candidate = evaluate_state(residual_fn, jacobian_fn,
                                                        current.x + alpha * step, "rootfinder");
                if (candidate.merit <= current.merit + opts.line_search_sufficient_decrease *
                                                           alpha * directional_derivative) {
                    current = std::move(candidate);
                    out.state = current;
                    out.step_norm = (alpha * step).lpNorm<Eigen::Infinity>();
                    accepted = true;
                    break;
                }
            } catch (const std::exception &) {
            }
            alpha *= opts.line_search_contraction;
        }

        if (!accepted) {
            out.message = "line-search Newton could not find an acceptable step";
            break;
        }

        if (is_converged(current, opts)) {
            out.converged = true;
            out.message = "converged with line-search Newton";
            return out;
        }
    }

    if (out.message.empty()) {
        out.message = "line-search Newton exhausted its iteration budget";
    }
    return out;
}

inline StageOutcome solve_broyden(const casadi::Function &residual_fn,
                                  const casadi::Function &jacobian_fn, const NumericState &start,
                                  const RootFinderOptions &opts, int max_iterations) {
    StageOutcome out;
    out.method = RootSolveMethod::QuasiNewtonBroyden;
    out.state = start;
    if (is_converged(start, opts) || max_iterations <= 0) {
        out.converged = is_converged(start, opts);
        out.message = out.converged ? "initial iterate satisfies tolerance"
                                    : "no Broyden iterations remaining";
        return out;
    }

    Eigen::VectorXd x = start.x;
    Eigen::VectorXd residual = start.residual;
    double merit = start.merit;
    Eigen::MatrixXd B = start.jacobian;
    int accepted_steps = 0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::VectorXd step = solve_linear_system(B, -residual);
        if (!all_finite(step)) {
            out.message = "Broyden update produced a non-finite step";
            break;
        }

        const double full_step_norm = step.lpNorm<Eigen::Infinity>();
        out.iterations += 1;
        if (full_step_norm <= opts.abstolStep) {
            out.step_norm = full_step_norm;
            out.message = "Broyden update stalled on a tiny step";
            break;
        }

        double alpha = 1.0;
        bool accepted = false;
        Eigen::VectorXd candidate_x;
        Eigen::VectorXd candidate_residual;
        for (int bt = 0; bt < opts.max_backtracking_steps; ++bt) {
            candidate_x = x + alpha * step;
            try {
                candidate_residual = evaluate_residual_only(residual_fn, candidate_x);
                const double candidate_merit = 0.5 * candidate_residual.squaredNorm();
                if (candidate_merit < merit) {
                    merit = candidate_merit;
                    out.step_norm = (alpha * step).lpNorm<Eigen::Infinity>();
                    accepted = true;
                    break;
                }
            } catch (const std::exception &) {
            }
            alpha *= opts.line_search_contraction;
        }

        if (!accepted) {
            out.message = "Broyden update could not decrease the residual merit";
            break;
        }

        const Eigen::VectorXd s = candidate_x - x;
        const Eigen::VectorXd y = candidate_residual - residual;
        const double denom = s.squaredNorm();

        x = candidate_x;
        residual = candidate_residual;
        accepted_steps += 1;

        if (residual.lpNorm<Eigen::Infinity>() <= opts.abstol) {
            out.state = evaluate_state(residual_fn, jacobian_fn, x, "rootfinder");
            out.converged = true;
            out.message = "converged with Broyden updates";
            return out;
        }

        const bool refresh_exact = opts.broyden_jacobian_refresh > 0 &&
                                   accepted_steps % opts.broyden_jacobian_refresh == 0;
        if (refresh_exact) {
            out.state = evaluate_state(residual_fn, jacobian_fn, x, "rootfinder");
            B = out.state.jacobian;
        } else if (denom > std::numeric_limits<double>::epsilon()) {
            B.noalias() += ((y - B * s) / denom) * s.transpose();
        } else {
            out.message = "Broyden update encountered a degenerate secant step";
            break;
        }
    }

    out.state = evaluate_state(residual_fn, jacobian_fn, x, "rootfinder");
    if (out.message.empty()) {
        out.message = "Broyden update exhausted its iteration budget";
    }
    return out;
}

inline StageOutcome solve_pseudo_transient(const casadi::Function &residual_fn,
                                           const casadi::Function &jacobian_fn,
                                           const NumericState &start, const RootFinderOptions &opts,
                                           int max_iterations) {
    StageOutcome out;
    out.method = RootSolveMethod::PseudoTransientContinuation;
    out.state = start;
    if (is_converged(start, opts) || max_iterations <= 0) {
        out.converged = is_converged(start, opts);
        out.message = out.converged ? "initial iterate satisfies tolerance"
                                    : "no pseudo-transient iterations remaining";
        return out;
    }

    NumericState current = start;
    double dt = opts.pseudo_transient_dt0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        out.iterations += 1;
        bool accepted = false;

        for (int bt = 0; bt < opts.max_backtracking_steps; ++bt) {
            Eigen::MatrixXd shifted = current.jacobian;
            shifted.diagonal().array() += 1.0 / dt;

            const Eigen::VectorXd step = solve_linear_system(shifted, -current.residual);
            out.step_norm = step.lpNorm<Eigen::Infinity>();
            if (!all_finite(step)) {
                out.message = "pseudo-transient continuation produced a non-finite step";
                return out;
            }
            if (out.step_norm <= opts.abstolStep) {
                out.message = "pseudo-transient continuation stalled on a tiny step";
                return out;
            }

            try {
                NumericState candidate =
                    evaluate_state(residual_fn, jacobian_fn, current.x + step, "rootfinder");
                if (candidate.merit < current.merit) {
                    current = std::move(candidate);
                    out.state = current;
                    dt = std::min(opts.pseudo_transient_dt_max,
                                  dt * opts.pseudo_transient_dt_growth);
                    accepted = true;
                    break;
                }
            } catch (const std::exception &) {
            }

            dt *= opts.line_search_contraction;
        }

        if (!accepted) {
            out.message = "pseudo-transient continuation could not decrease the residual merit";
            break;
        }

        if (is_converged(current, opts)) {
            out.converged = true;
            out.message = "converged with pseudo-transient continuation";
            return out;
        }
    }

    if (out.message.empty()) {
        out.message = "pseudo-transient continuation exhausted its iteration budget";
    }
    return out;
}

} // namespace detail

/**
 * @brief Persistent nonlinear root solver.
 *
 * Numeric solves use a globalization stack with trust-region Newton, line
 * search Newton, Broyden, and pseudo-transient continuation. Symbolic solves
 * still embed CasADi's differentiable `newton` rootfinder in the graph.
 */
class NewtonSolver {
  public:
    /**
     * @brief Construct a new persistent nonlinear solver.
     *
     * @param F Function F(x) = 0 to solve
     * @param opts Solver options
     */
    NewtonSolver(const metis::Function &F, const RootFinderOptions &opts = {}) : opts_(opts) {
        detail::validate_root_options(opts_, "NewtonSolver");

        casadi::Function f_casadi = F.casadi_function();
        detail::validate_root_problem(f_casadi, "NewtonSolver");

        residual_fn_ = f_casadi;
        n_x_ = f_casadi.nnz_in(0);

        casadi::MX x = metis::sym(f_casadi.name_in(0), f_casadi.size1_in(0), f_casadi.size2_in(0));
        casadi::MX residual = f_casadi(std::vector<casadi::MX>{x}).at(0);
        jacobian_fn_ = casadi::Function(detail::unique_name(f_casadi.name() + "_jacobian"),
                                        std::vector<casadi::MX>{x},
                                        std::vector<casadi::MX>{casadi::MX::jacobian(residual, x)});

        try {
            symbolic_solver_ = casadi::rootfinder(detail::unique_name("rf_solver"), "newton",
                                                  f_casadi, detail::opts_to_dict(opts_));
        } catch (const std::exception &e) {
            throw MetisError(std::string("NewtonSolver creation failed: ") + e.what());
        }
    }

    /**
     * @brief Solve F(x) = 0 from initial guess x0.
     */
    template <typename Scalar>
    RootResult<Scalar> solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x0) const {
        RootResult<Scalar> result;
        const int n_x = static_cast<int>(x0.size());
        if (n_x != n_x_) {
            throw InvalidArgument("NewtonSolver::solve: x0 size must match the residual dimension");
        }

        if constexpr (std::is_floating_point_v<Scalar>) {
            const Eigen::VectorXd x0_numeric = x0.template cast<double>();
            const auto initial = detail::evaluate_state(residual_fn_, jacobian_fn_, x0_numeric,
                                                        "NewtonSolver::solve");
            detail::maybe_log(opts_, "initial residual inf-norm = " +
                                         std::to_string(initial.residual_norm));

            detail::NumericState current = initial;
            detail::NumericState best = initial;

            result.x = initial.x.template cast<Scalar>();
            result.residual_norm = initial.residual_norm;
            if (detail::is_converged(initial, opts_)) {
                result.converged = true;
                result.message = "Initial guess satisfies the residual tolerance";
                return result;
            }

            int iterations_used = 0;
            RootSolveMethod last_method = RootSolveMethod::None;
            std::string last_message = "No numeric solver stage was executed";
            std::vector<RootSolveMethod> order;
            if (opts_.strategy == RootSolveStrategy::Auto) {
                order.push_back(RootSolveMethod::TrustRegionNewton);
                if (opts_.line_search) {
                    order.push_back(RootSolveMethod::LineSearchNewton);
                }
                order.push_back(RootSolveMethod::QuasiNewtonBroyden);
                order.push_back(RootSolveMethod::PseudoTransientContinuation);
            } else {
                order.push_back(detail::strategy_to_method(opts_.strategy));
            }

            for (RootSolveMethod method : order) {
                const int remaining = opts_.max_iter - iterations_used;
                if (remaining <= 0) {
                    break;
                }

                detail::maybe_log(opts_, "starting " + detail::method_name(method));
                detail::StageOutcome stage;
                switch (method) {
                case RootSolveMethod::TrustRegionNewton:
                    stage = detail::solve_trust_region(residual_fn_, jacobian_fn_, current, opts_,
                                                       remaining);
                    break;
                case RootSolveMethod::LineSearchNewton:
                    stage = detail::solve_line_search(residual_fn_, jacobian_fn_, current, opts_,
                                                      remaining);
                    break;
                case RootSolveMethod::QuasiNewtonBroyden:
                    stage = detail::solve_broyden(residual_fn_, jacobian_fn_, current, opts_,
                                                  remaining);
                    break;
                case RootSolveMethod::PseudoTransientContinuation:
                    stage = detail::solve_pseudo_transient(residual_fn_, jacobian_fn_, current,
                                                           opts_, remaining);
                    break;
                case RootSolveMethod::None:
                    continue;
                }

                iterations_used += stage.iterations;
                current = stage.state;
                if (current.merit < best.merit) {
                    best = current;
                }

                last_method = method;
                last_message = stage.message;
                result.step_norm = stage.step_norm;
                detail::maybe_log(opts_, detail::method_name(method) + ": residual inf-norm = " +
                                             std::to_string(current.residual_norm) +
                                             ", iterations = " + std::to_string(iterations_used));

                if (stage.converged) {
                    result.x = current.x.template cast<Scalar>();
                    result.iterations = iterations_used;
                    result.converged = true;
                    result.method = method;
                    result.residual_norm = current.residual_norm;
                    result.message = stage.message;
                    return result;
                }
            }

            result.x = best.x.template cast<Scalar>();
            result.iterations = iterations_used;
            result.converged = false;
            result.method = last_method;
            result.residual_norm = best.residual_norm;
            result.message = "Failed to converge: " + last_message;
        } else {
            SymbolicScalar x0_mx = metis::to_mx(x0);

            std::vector<SymbolicScalar> args;
            args.push_back(x0_mx);
            if (symbolic_solver_.n_in() > 1) {
                args.push_back(SymbolicScalar(0, 0));
            }
            std::vector<SymbolicScalar> res = symbolic_solver_(args);

            result.x = metis::to_eigen(res[0]);
            result.converged = true;
            result.message = "Symbolic graph generated with CasADi newton rootfinder";
        }
        return result;
    }

  private:
    RootFinderOptions opts_;
    int n_x_ = 0;
    casadi::Function residual_fn_;
    casadi::Function jacobian_fn_;
    casadi::Function symbolic_solver_;
};

/**
 * @brief Solve F(x) = 0 for x given an initial guess.
 *
 * Numeric mode uses Metis' globalization stack. Symbolic mode embeds CasADi's
 * differentiable rootfinder.
 *
 * @tparam Scalar double (numeric) or SymbolicScalar (symbolic)
 * @param F Function mapping x -> residual
 * @param x0 Initial guess
 * @param opts Solver options
 * @return RootResult containing solution and diagnostics
 */
template <typename Scalar>
RootResult<Scalar> rootfinder(const metis::Function &F,
                              const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x0,
                              const RootFinderOptions &opts = {}) {
    NewtonSolver solver(F, opts);
    return solver.solve(x0);
}

/**
 * @brief Create a differentiable implicit solve wrapper for G(...) = 0.
 *
 * All non-implicit inputs of @p G become inputs of the returned function, in
 * their original order and original shapes. CasADi's rootfinder provides the
 * implicit sensitivities, so the returned function remains differentiable with
 * respect to every remaining input.
 *
 * @param G Implicit function containing the unknown input and residual output
 * @param x_guess Fixed initial guess for the implicit solve
 * @param opts Solver options
 * @param implicit_opts Selects which input/output pair defines the rootfinding problem
 * @return metis::Function mapping the remaining inputs to x(...)
 */
inline metis::Function create_implicit_function(const metis::Function &G,
                                                const Eigen::VectorXd &x_guess,
                                                const RootFinderOptions &opts = {},
                                                const ImplicitFunctionOptions &implicit_opts = {}) {
    detail::validate_root_options(opts, "create_implicit_function");

    casadi::Function g_casadi = G.casadi_function();
    detail::validate_implicit_problem(g_casadi, x_guess, implicit_opts);

    casadi::Dict solver_opts = detail::opts_to_dict(opts);
    solver_opts["implicit_input"] = implicit_opts.implicit_input_index;
    solver_opts["implicit_output"] = implicit_opts.implicit_output_index;

    casadi::Function solver = casadi::rootfinder(detail::implicit_function_name(g_casadi), "newton",
                                                 g_casadi, solver_opts);

    std::vector<SymbolicArg> wrapper_inputs;
    wrapper_inputs.reserve(static_cast<std::size_t>(g_casadi.n_in() - 1));

    std::vector<casadi::MX> solver_args;
    solver_args.reserve(static_cast<std::size_t>(g_casadi.n_in()));

    const casadi::DM x0 = detail::vector_to_dm(x_guess);
    for (int i = 0; i < g_casadi.n_in(); ++i) {
        if (i == implicit_opts.implicit_input_index) {
            solver_args.push_back(casadi::MX(x0));
            continue;
        }

        casadi::MX arg =
            metis::sym(g_casadi.name_in(i), g_casadi.size1_in(i), g_casadi.size2_in(i));
        wrapper_inputs.push_back(SymbolicArg(arg));
        solver_args.push_back(arg);
    }

    const auto solver_outputs = solver(solver_args);
    const casadi::MX x_sol = solver_outputs.at(implicit_opts.implicit_output_index);

    return metis::Function(detail::implicit_function_name(g_casadi), wrapper_inputs,
                           std::vector<SymbolicArg>{SymbolicArg(x_sol)});
}

} // namespace metis
