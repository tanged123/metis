#pragma once
/**
 * @file AutoDiff.hpp
 * @brief Symbolic automatic differentiation (Jacobian, gradient, Hessian, sensitivity analysis)
 * @see Calculus.hpp, FiniteDifference.hpp
 */

#include "metis/core/Function.hpp"
#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/core/Sparsity.hpp"
#include <string>
#include <vector>

namespace metis {

/**
 * @brief Sensitivity regime selected for a Jacobian-like derivative workload.
 */
enum class SensitivityRegime {
    Forward,            ///< Many outputs, relatively few parameters
    Adjoint,            ///< Few outputs, many parameters
    CheckpointedAdjoint ///< Adjoint with checkpoint recommendations for long horizons
};

/**
 * @brief Checkpoint interpolation recommendation for long-horizon adjoints.
 */
enum class CheckpointInterpolation {
    None,
    Hermite,
    Polynomial,
};

/**
 * @brief Heuristics controlling automatic forward-vs-adjoint selection.
 */
struct SensitivitySwitchOptions {
    int forward_parameter_threshold = 20;
    int adjoint_parameter_threshold = 100;
    int checkpoint_horizon_threshold = 200;
    int very_long_horizon_threshold = 1000;
    int checkpoint_steps = 20;
    int checkpoint_steps_very_long = 50;
};

/**
 * @brief Result of Metis sensitivity regime selection.
 */
struct SensitivityRecommendation {
    SensitivityRegime regime = SensitivityRegime::Forward;
    CheckpointInterpolation checkpoint_interpolation = CheckpointInterpolation::None;
    int parameter_count = 0;
    int output_count = 0;
    int horizon_length = 1;
    int steps_per_checkpoint = 0;
    bool stiff = false;

    bool uses_forward_mode() const { return regime == SensitivityRegime::Forward; }

    bool uses_reverse_mode() const { return regime != SensitivityRegime::Forward; }

    bool uses_checkpointing() const { return regime == SensitivityRegime::CheckpointedAdjoint; }

    int casadi_direction_count() const {
        return uses_forward_mode() ? parameter_count : output_count;
    }

    /**
     * @brief Convert the recommendation into SUNDIALS/CasADi integrator options.
     *
     * This is intended for downstream integrator construction. It exposes the
     * same regime choice through `nfwd`/`nadj` and, for long-horizon adjoints,
     * checkpoint interpolation and spacing.
     */
    casadi::Dict integrator_options() const {
        casadi::Dict opts;
        if (uses_forward_mode()) {
            opts["nfwd"] = parameter_count;
            opts["fsens_err_con"] = true;
        } else {
            opts["nadj"] = output_count;
            if (uses_checkpointing()) {
                opts["steps_per_checkpoint"] = steps_per_checkpoint;
                if (checkpoint_interpolation == CheckpointInterpolation::Polynomial) {
                    opts["interpolation_type"] = std::string("polynomial");
                } else if (checkpoint_interpolation == CheckpointInterpolation::Hermite) {
                    opts["interpolation_type"] = std::string("hermite");
                }
            }
        }
        return opts;
    }
};

// --- Jacobian (Symbolic Automatic Differentiation) ---
/**
 * @brief Computes Jacobian of an expression with respect to variables.
 *
 * Wraps CasADi's jacobian function. Automatic Jacobian only supported for Symbolic types currently.
 *
 * @tparam Expr Expression type (must be convertible to MX)
 * @tparam Vars Variable types (must be convertible to MX)
 * @param expression Expression to differentiate
 * @param variables Variables to differentiate with respect to
 * @return Jacobian matrix (symbolic)
 */
template <typename Expr, typename... Vars>
auto jacobian(const Expr &expression, const Vars &...variables) {
    if constexpr (std::is_same_v<Expr, SymbolicScalar>) {
        // Collect pack into a vector of MX
        std::vector<SymbolicScalar> vars;
        (vars.push_back(variables), ...);

        SymbolicScalar v_cat = SymbolicScalar::vertcat(vars);
        return SymbolicScalar::jacobian(expression, v_cat);
    } else {
        // Fallback or static_assert for non-symbolic types
        // Ideally we would support numeric autodiff here later
        static_assert(std::is_same_v<Expr, casadi::MX>,
                      "Automatic Jacobian only supported for Symbolic types currently.");
        return Expr(0);
    }
}

// Reuse shared helpers from Sparsity.hpp detail namespace
namespace detail {

inline int input_numel(const Function &fn, int input_idx) {
    const auto &cas_fn = fn.casadi_function();
    return cas_fn.size1_in(input_idx) * cas_fn.size2_in(input_idx);
}

inline int output_numel(const Function &fn, int output_idx) {
    const auto &cas_fn = fn.casadi_function();
    return cas_fn.size1_out(output_idx) * cas_fn.size2_out(output_idx);
}

inline void validate_scalar_output(const Function &fn, int output_idx, const std::string &context) {
    if (output_numel(fn, output_idx) != 1) {
        throw InvalidArgument(context + ": selected function output must be scalar");
    }
}

inline void validate_same_shape(const casadi::MX &lhs, const casadi::MX &rhs,
                                const std::string &context, const std::string &rhs_name) {
    if (lhs.size1() != rhs.size1() || lhs.size2() != rhs.size2()) {
        throw InvalidArgument(context + ": " + rhs_name +
                              " must have the same shape as the differentiation variables");
    }
}

inline std::vector<casadi::MX> symbolic_inputs_like(const Function &fn) {
    return fn.casadi_function().mx_in();
}

inline casadi::MX make_zero_seed(int rows, int cols, int directions) {
    return casadi::MX::zeros(rows, cols * directions);
}

inline casadi::MX make_basis_seed(int rows, int cols) {
    const int directions = rows * cols;
    std::vector<casadi::MX> blocks;
    blocks.reserve(static_cast<std::size_t>(directions));

    for (int direction = 0; direction < directions; ++direction) {
        casadi::DM e = casadi::DM::zeros(directions, 1);
        e(direction) = 1.0;
        blocks.push_back(casadi::MX::reshape(casadi::MX(e), rows, cols));
    }

    return casadi::MX::horzcat(blocks);
}

inline casadi::MX stacked_blocks_to_columns(const casadi::MX &stacked, int block_rows,
                                            int block_cols, int n_blocks) {
    std::vector<casadi::MX> cols;
    cols.reserve(static_cast<std::size_t>(n_blocks));

    for (int block = 0; block < n_blocks; ++block) {
        casadi::MX block_mx =
            stacked(casadi::Slice(), casadi::Slice(block * block_cols, (block + 1) * block_cols));
        cols.push_back(casadi::MX::reshape(block_mx, block_rows * block_cols, 1));
    }

    return casadi::MX::horzcat(cols);
}

inline std::string sensitivity_function_name(const Function &fn, const std::string &suffix,
                                             int output_idx, int input_idx) {
    return fn.casadi_function().name() + "_" + suffix + "_o" + std::to_string(output_idx) + "_i" +
           std::to_string(input_idx);
}

inline std::string lagrangian_function_name(const Function &fn, const std::string &suffix,
                                            int objective_output_idx, int constraint_output_idx,
                                            int input_idx) {
    return fn.casadi_function().name() + "_" + suffix + "_obj" +
           std::to_string(objective_output_idx) + "_con" + std::to_string(constraint_output_idx) +
           "_i" + std::to_string(input_idx);
}

inline void validate_scalar_expression(const casadi::MX &expr, const std::string &context,
                                       const std::string &name) {
    if (expr.numel() != 1) {
        throw InvalidArgument(context + ": " + name + " must be scalar");
    }
}

inline casadi::MX lagrangian_scalar(const SymbolicArg &objective, const SymbolicArg &constraints,
                                    const SymbolicArg &multipliers, const std::string &context) {
    const casadi::MX objective_mx = objective.get();
    const casadi::MX constraints_mx = constraints.get();
    const casadi::MX multipliers_mx = multipliers.get();

    validate_scalar_expression(objective_mx, context, "objective");
    if (constraints_mx.numel() != multipliers_mx.numel()) {
        throw InvalidArgument(context + ": multipliers must have the same number of entries as "
                                        "the constraint output");
    }

    const casadi::MX constraints_vec =
        casadi::MX::reshape(constraints_mx, constraints_mx.numel(), 1);
    const casadi::MX multipliers_vec =
        casadi::MX::reshape(multipliers_mx, multipliers_mx.numel(), 1);
    return objective_mx + casadi::MX::dot(multipliers_vec, constraints_vec);
}

inline casadi::MX forward_block_jacobian(const Function &fn, int output_idx, int input_idx) {
    const auto &cas_fn = fn.casadi_function();
    const int nfwd = input_numel(fn, input_idx);
    const int out_rows = cas_fn.size1_out(output_idx);
    const int out_cols = cas_fn.size2_out(output_idx);

    std::vector<casadi::MX> inputs = symbolic_inputs_like(fn);
    std::vector<casadi::MX> outputs = cas_fn(inputs);

    casadi::Function fwd = cas_fn.forward(nfwd);

    std::vector<casadi::MX> args;
    args.reserve(static_cast<std::size_t>(cas_fn.n_in() * 2 + cas_fn.n_out()));
    args.insert(args.end(), inputs.begin(), inputs.end());
    args.insert(args.end(), outputs.begin(), outputs.end());

    for (int i = 0; i < cas_fn.n_in(); ++i) {
        if (i == input_idx) {
            args.push_back(make_basis_seed(cas_fn.size1_in(i), cas_fn.size2_in(i)));
        } else {
            args.push_back(make_zero_seed(cas_fn.size1_in(i), cas_fn.size2_in(i), nfwd));
        }
    }

    std::vector<casadi::MX> sensitivities = fwd(args);
    return stacked_blocks_to_columns(sensitivities.at(output_idx), out_rows, out_cols, nfwd);
}

inline casadi::MX reverse_block_jacobian(const Function &fn, int output_idx, int input_idx) {
    const auto &cas_fn = fn.casadi_function();
    const int nadj = output_numel(fn, output_idx);
    const int in_rows = cas_fn.size1_in(input_idx);
    const int in_cols = cas_fn.size2_in(input_idx);

    std::vector<casadi::MX> inputs = symbolic_inputs_like(fn);
    std::vector<casadi::MX> outputs = cas_fn(inputs);

    casadi::Function adj = cas_fn.reverse(nadj);

    std::vector<casadi::MX> args;
    args.reserve(static_cast<std::size_t>(cas_fn.n_in() + cas_fn.n_out() * 2));
    args.insert(args.end(), inputs.begin(), inputs.end());
    args.insert(args.end(), outputs.begin(), outputs.end());

    for (int i = 0; i < cas_fn.n_out(); ++i) {
        if (i == output_idx) {
            args.push_back(make_basis_seed(cas_fn.size1_out(i), cas_fn.size2_out(i)));
        } else {
            args.push_back(make_zero_seed(cas_fn.size1_out(i), cas_fn.size2_out(i), nadj));
        }
    }

    std::vector<casadi::MX> sensitivities = adj(args);
    casadi::MX jac_t =
        stacked_blocks_to_columns(sensitivities.at(input_idx), in_rows, in_cols, nadj);
    return jac_t.T();
}

} // namespace detail

/**
 * @brief Recommend a sensitivity regime from parameter/output counts.
 */
inline SensitivityRecommendation
select_sensitivity_regime(int parameter_count, int output_count, int horizon_length = 1,
                          bool stiff = false,
                          const SensitivitySwitchOptions &opts = SensitivitySwitchOptions()) {
    if (parameter_count <= 0) {
        throw InvalidArgument("select_sensitivity_regime: parameter_count must be positive");
    }
    if (output_count <= 0) {
        throw InvalidArgument("select_sensitivity_regime: output_count must be positive");
    }
    if (horizon_length <= 0) {
        throw InvalidArgument("select_sensitivity_regime: horizon_length must be positive");
    }
    if (opts.forward_parameter_threshold <= 0 || opts.adjoint_parameter_threshold <= 0 ||
        opts.checkpoint_horizon_threshold <= 0 || opts.very_long_horizon_threshold <= 0 ||
        opts.checkpoint_steps <= 0 || opts.checkpoint_steps_very_long <= 0) {
        throw InvalidArgument("select_sensitivity_regime: thresholds must be positive");
    }
    if (opts.forward_parameter_threshold > opts.adjoint_parameter_threshold) {
        throw InvalidArgument(
            "select_sensitivity_regime: forward_parameter_threshold must not exceed "
            "adjoint_parameter_threshold");
    }
    if (opts.checkpoint_horizon_threshold > opts.very_long_horizon_threshold) {
        throw InvalidArgument(
            "select_sensitivity_regime: checkpoint_horizon_threshold must not exceed "
            "very_long_horizon_threshold");
    }

    SensitivityRecommendation recommendation;
    recommendation.parameter_count = parameter_count;
    recommendation.output_count = output_count;
    recommendation.horizon_length = horizon_length;
    recommendation.stiff = stiff;

    const bool forward_friendly =
        parameter_count <= opts.forward_parameter_threshold && output_count >= parameter_count;
    const bool long_horizon = horizon_length >= opts.checkpoint_horizon_threshold;
    const bool output_limited = parameter_count > output_count;
    const bool parameter_heavy = parameter_count >= opts.adjoint_parameter_threshold;

    if (forward_friendly) {
        recommendation.regime = SensitivityRegime::Forward;
        return recommendation;
    }

    if (long_horizon && output_limited) {
        recommendation.regime = SensitivityRegime::CheckpointedAdjoint;
        recommendation.checkpoint_interpolation =
            stiff ? CheckpointInterpolation::Polynomial : CheckpointInterpolation::Hermite;
        recommendation.steps_per_checkpoint = horizon_length >= opts.very_long_horizon_threshold
                                                  ? opts.checkpoint_steps_very_long
                                                  : opts.checkpoint_steps;
        return recommendation;
    }

    recommendation.regime = (parameter_heavy || output_limited) ? SensitivityRegime::Adjoint
                                                                : SensitivityRegime::Forward;
    return recommendation;
}

/**
 * @brief Recommend a sensitivity regime for a selected `metis::Function` block.
 */
inline SensitivityRecommendation
select_sensitivity_regime(const Function &fn, int output_idx = 0, int input_idx = 0,
                          int horizon_length = 1, bool stiff = false,
                          const SensitivitySwitchOptions &opts = SensitivitySwitchOptions()) {
    detail::validate_function_indices(fn, output_idx, input_idx, "select_sensitivity_regime");
    return select_sensitivity_regime(detail::input_numel(fn, input_idx),
                                     detail::output_numel(fn, output_idx), horizon_length, stiff,
                                     opts);
}

/**
 * @brief Build a Jacobian function for one output/input block using the recommended regime.
 *
 * The returned Jacobian is always dense and uses column-major vectorization of the selected
 * output and input blocks, matching CasADi/Eigen storage order.
 */
inline Function
sensitivity_jacobian(const Function &fn, int output_idx = 0, int input_idx = 0,
                     int horizon_length = 1, bool stiff = false,
                     const SensitivitySwitchOptions &opts = SensitivitySwitchOptions()) {
    detail::validate_function_indices(fn, output_idx, input_idx, "sensitivity_jacobian");

    const SensitivityRecommendation recommendation =
        select_sensitivity_regime(fn, output_idx, input_idx, horizon_length, stiff, opts);

    casadi::MX jacobian_expr = recommendation.uses_forward_mode()
                                   ? detail::forward_block_jacobian(fn, output_idx, input_idx)
                                   : detail::reverse_block_jacobian(fn, output_idx, input_idx);

    return Function(
        detail::sensitivity_function_name(
            fn, recommendation.uses_forward_mode() ? "fwd_jac" : "adj_jac", output_idx, input_idx),
        detail::to_symbolic_args(detail::symbolic_inputs_like(fn)),
        std::vector<SymbolicArg>{SymbolicArg(jacobian_expr)});
}

/**
 * @brief Computes Jacobian with vector arguments (expressions and variables)
 *
 * @param expressions Vector of symbolic expressions
 * @param variables Vector of symbolic variables
 * @return Jacobian matrix
 */
inline auto jacobian(const std::vector<SymbolicArg> &expressions,
                     const std::vector<SymbolicArg> &variables) {
    SymbolicScalar expr_cat = SymbolicScalar::vertcat(detail::to_mx_vector(expressions));
    SymbolicScalar var_cat = SymbolicScalar::vertcat(detail::to_mx_vector(variables));
    return SymbolicScalar::jacobian(expr_cat, var_cat);
}

/**
 * @brief Symbolic gradient (for scalar-output functions)
 *
 * Computes the symbolic gradient of a scalar expression with respect to variables.
 * Note: Named `sym_gradient` to distinguish from numerical `gradient()` in Calculus.hpp.
 *
 * @param expr Scalar expression to differentiate
 * @param vars Variables to differentiate with respect to
 * @return Column vector of partial derivatives
 */
inline SymbolicVector sym_gradient(const SymbolicArg &expr, const SymbolicArg &vars) {
    // CasADi gradient returns a dense vector (column or row depending on version, usually 1xN or
    // Nx1) We force it to be a column vector
    SymbolicScalar g = SymbolicScalar::gradient(expr.get(), vars.get());

    // Convert to SymbolicVector (Metis type)
    if (g.size2() > 1 && g.size1() == 1) {
        return to_eigen(g.T());
    }
    return to_eigen(g);
}

/**
 * @brief Hessian matrix (second-order derivatives)
 *
 * Computes the matrix of second derivatives: H_ij = d²f / (dx_i dx_j)
 *
 * @param expr Scalar expression
 * @param vars Variables
 * @return Symmetric Hessian matrix (SymbolicMatrix)
 */
inline SymbolicMatrix hessian(const SymbolicArg &expr, const SymbolicArg &vars) {
    // casadi::MX::hessian returns {H, g}
    // We only want H
    // Note: older CasADi versions return H directly, but modern one returns vector
    // Let's use the property that hessian(f, x) = jacobian(gradient(f, x), x) for safety/clarity
    // or use the optimized CasADi call:
    SymbolicScalar H = SymbolicScalar::hessian(expr.get(), vars.get());
    return to_eigen(H);
}

/**
 * @brief Hessian of Lagrangian for constrained optimization
 *
 * Computes ∇²L where L = f(x) + λᵀg(x)
 * This is the exact matrix needed for IPOPT's hessian_lagrangian callback.
 *
 * @param objective Objective function f(x) (scalar)
 * @param constraints Constraint functions g(x) (vector)
 * @param vars Decision variables x (vector)
 * @param multipliers Lagrange multipliers λ (vector, same size as constraints)
 * @return Hessian of Lagrangian (symmetric matrix)
 */
inline SymbolicMatrix hessian_lagrangian(const SymbolicArg &objective,
                                         const SymbolicArg &constraints, const SymbolicArg &vars,
                                         const SymbolicArg &multipliers) {
    return metis::hessian(
        detail::lagrangian_scalar(objective, constraints, multipliers, "hessian_lagrangian"), vars);
}

/**
 * @brief Hessian-vector product for a scalar expression without forming the dense Hessian.
 *
 * This uses CasADi's directional derivative of the reverse-mode gradient, i.e.
 * forward-over-reverse AD.
 *
 * @param expr Scalar expression f(x)
 * @param vars Differentiation variables x
 * @param direction Direction vector/matrix v with the same shape as x
 * @return Matrix/vector with the same shape as x containing ∇²f(x) v
 */
inline SymbolicMatrix hessian_vector_product(const SymbolicArg &expr, const SymbolicArg &vars,
                                             const SymbolicArg &direction) {
    const casadi::MX expr_mx = expr.get();
    const casadi::MX vars_mx = vars.get();
    const casadi::MX direction_mx = direction.get();

    detail::validate_scalar_expression(expr_mx, "hessian_vector_product", "expression");
    detail::validate_same_shape(vars_mx, direction_mx, "hessian_vector_product", "direction");

    const casadi::MX gradient_mx = casadi::MX::gradient(expr_mx, vars_mx);
    const casadi::MX hvp_mx = casadi::MX::jtimes(gradient_mx, vars_mx, direction_mx, false);
    return to_eigen(casadi::MX::reshape(hvp_mx, vars_mx.size1(), vars_mx.size2()));
}

/**
 * @brief Hessian-vector product of a Lagrangian, i.e. a second-order adjoint action.
 *
 * Computes ∇²L(x, λ) v for L(x, λ) = f(x) + λᵀ g(x) without forming the dense Hessian.
 *
 * @param objective Objective function f(x) (scalar)
 * @param constraints Constraint function g(x)
 * @param vars Decision variables x
 * @param multipliers Lagrange multipliers λ (same number of entries as g)
 * @param direction Direction vector/matrix v with the same shape as x
 * @return Matrix/vector with the same shape as x containing ∇²L(x, λ) v
 */
inline SymbolicMatrix lagrangian_hessian_vector_product(const SymbolicArg &objective,
                                                        const SymbolicArg &constraints,
                                                        const SymbolicArg &vars,
                                                        const SymbolicArg &multipliers,
                                                        const SymbolicArg &direction) {
    return hessian_vector_product(detail::lagrangian_scalar(objective, constraints, multipliers,
                                                            "lagrangian_hessian_vector_product"),
                                  vars, direction);
}

/**
 * @brief Build a matrix-free Hessian-vector product function for one scalar output/input block.
 *
 * The returned function takes the original function inputs followed by one extra direction input
 * matching the selected input block shape. Its single output is ∇²y_i / ∇x_j² times that
 * direction, reshaped like the selected input block.
 */
inline Function hessian_vector_product(const Function &fn, int output_idx = 0, int input_idx = 0) {
    detail::validate_function_indices(fn, output_idx, input_idx, "hessian_vector_product");
    detail::validate_scalar_output(fn, output_idx, "hessian_vector_product");

    const auto &cas_fn = fn.casadi_function();
    std::vector<casadi::MX> inputs = detail::symbolic_inputs_like(fn);
    std::vector<casadi::MX> outputs = cas_fn(inputs);
    casadi::MX direction =
        casadi::MX::sym("hvp_direction", cas_fn.size1_in(input_idx), cas_fn.size2_in(input_idx));

    casadi::MX hvp_expr = as_mx(
        metis::hessian_vector_product(outputs.at(output_idx), inputs.at(input_idx), direction));

    std::vector<SymbolicArg> args = detail::to_symbolic_args(inputs);
    args.emplace_back(direction);

    return Function(detail::sensitivity_function_name(fn, "hvp", output_idx, input_idx), args,
                    std::vector<SymbolicArg>{SymbolicArg(hvp_expr)});
}

/**
 * @brief Build a Lagrangian Hessian-vector product function for optimization workflows.
 *
 * The returned function takes the original function inputs, then the constraint multipliers λ for
 * the selected constraint output block, and finally the direction v for the selected input block.
 * The output is ∇²L(x, λ) v, reshaped like the selected input block.
 */
inline Function lagrangian_hessian_vector_product(const Function &fn, int objective_output_idx,
                                                  int constraint_output_idx, int input_idx = 0) {
    detail::validate_function_indices(fn, objective_output_idx, input_idx,
                                      "lagrangian_hessian_vector_product");
    detail::validate_function_indices(fn, constraint_output_idx, input_idx,
                                      "lagrangian_hessian_vector_product");
    detail::validate_scalar_output(fn, objective_output_idx, "lagrangian_hessian_vector_product");

    const auto &cas_fn = fn.casadi_function();
    std::vector<casadi::MX> inputs = detail::symbolic_inputs_like(fn);
    std::vector<casadi::MX> outputs = cas_fn(inputs);
    casadi::MX multipliers =
        casadi::MX::sym("lagrange_multipliers", cas_fn.size1_out(constraint_output_idx),
                        cas_fn.size2_out(constraint_output_idx));
    casadi::MX direction = casadi::MX::sym("lagrangian_hvp_direction", cas_fn.size1_in(input_idx),
                                           cas_fn.size2_in(input_idx));

    casadi::MX hvp_expr = as_mx(metis::lagrangian_hessian_vector_product(
        outputs.at(objective_output_idx), outputs.at(constraint_output_idx), inputs.at(input_idx),
        multipliers, direction));

    std::vector<SymbolicArg> args = detail::to_symbolic_args(inputs);
    args.emplace_back(multipliers);
    args.emplace_back(direction);

    return Function(detail::lagrangian_function_name(fn, "lag_hvp", objective_output_idx,
                                                     constraint_output_idx, input_idx),
                    args, std::vector<SymbolicArg>{SymbolicArg(hvp_expr)});
}

// --- Overloads for vector inputs (convenience) ---

/**
 * @brief Symbolic gradient with vector of variables
 */
inline SymbolicVector sym_gradient(const SymbolicArg &expr, const std::vector<SymbolicArg> &vars) {
    SymbolicScalar v_cat = SymbolicScalar::vertcat(detail::to_mx_vector(vars));
    return sym_gradient(expr, v_cat);
}

/**
 * @brief Hessian with vector of variables
 */
inline SymbolicMatrix hessian(const SymbolicArg &expr, const std::vector<SymbolicArg> &vars) {
    SymbolicScalar v_cat = SymbolicScalar::vertcat(detail::to_mx_vector(vars));
    return metis::hessian(expr, v_cat);
}

/**
 * @brief Hessian of Lagrangian with vector inputs
 */
inline SymbolicMatrix hessian_lagrangian(const SymbolicArg &objective,
                                         const SymbolicArg &constraints,
                                         const std::vector<SymbolicArg> &vars,
                                         const SymbolicArg &multipliers) {
    SymbolicScalar v_cat = SymbolicScalar::vertcat(detail::to_mx_vector(vars));
    return hessian_lagrangian(objective, constraints, v_cat, multipliers);
}

/**
 * @brief Hessian-vector product with vector of variables
 */
inline SymbolicMatrix hessian_vector_product(const SymbolicArg &expr,
                                             const std::vector<SymbolicArg> &vars,
                                             const SymbolicArg &direction) {
    SymbolicScalar v_cat = SymbolicScalar::vertcat(detail::to_mx_vector(vars));
    return hessian_vector_product(expr, v_cat, direction);
}

/**
 * @brief Lagrangian Hessian-vector product with vector of variables
 */
inline SymbolicMatrix lagrangian_hessian_vector_product(const SymbolicArg &objective,
                                                        const SymbolicArg &constraints,
                                                        const std::vector<SymbolicArg> &vars,
                                                        const SymbolicArg &multipliers,
                                                        const SymbolicArg &direction) {
    SymbolicScalar v_cat = SymbolicScalar::vertcat(detail::to_mx_vector(vars));
    return lagrangian_hessian_vector_product(objective, constraints, v_cat, multipliers, direction);
}

} // namespace metis
