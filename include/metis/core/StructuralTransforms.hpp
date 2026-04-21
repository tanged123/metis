/// @file StructuralTransforms.hpp
/// @brief Alias elimination, BLT decomposition, and structural analysis passes
#pragma once

#include "Function.hpp"
#include "MetisError.hpp"
#include "MetisTypes.hpp"
#include "Sparsity.hpp"

#include <algorithm>
#include <casadi/casadi.hpp>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace metis {

/**
 * @brief Options for structural simplification and analysis passes.
 */
struct StructuralTransformOptions {
    int input_idx = 0;         ///< Index of the input block to analyze
    int output_idx = 0;        ///< Index of the output block to analyze
    int max_alias_row_nnz = 2; ///< Max non-zeros in an alias candidate row
    bool require_constant_alias_coefficients =
        true; ///< Require constant (symbol-free) coefficients
};

/**
 * @brief A single eliminated alias or trivial affine variable relation.
 */
struct AliasSubstitution {
    int residual_index = -1;            ///< Row that was eliminated
    int eliminated_variable_index = -1; ///< Variable solved for
    SymbolicScalar replacement;         ///< Expression replacing the eliminated variable
};

/**
 * @brief Result of alias elimination on a selected residual block.
 *
 * `reduced_function` keeps the original input ordering, but the selected input
 * block is reduced to the kept variables and the output is only the kept
 * residual equations. `reconstruct_full_input` maps the reduced input block and
 * any untouched original inputs back to the full original selected input block.
 */
struct AliasEliminationResult {
    Function reduced_function;                    ///< Function with aliases removed
    Function reconstruct_full_input;              ///< Maps reduced inputs back to original ordering
    std::vector<int> kept_variable_indices;       ///< Original indices of kept variables
    std::vector<int> eliminated_variable_indices; ///< Original indices of eliminated variables
    std::vector<int> kept_residual_indices;       ///< Original indices of kept residuals
    std::vector<int> eliminated_residual_indices; ///< Original indices of eliminated residuals
    std::vector<AliasSubstitution> substitutions; ///< Applied substitution records
};

/**
 * @brief One diagonal block in a block-triangular decomposition.
 */
struct StructuralBlock {
    std::vector<int> residual_indices;      ///< Residual rows in this block
    std::vector<int> variable_indices;      ///< Variable columns in this block
    std::vector<int> tear_variable_indices; ///< Recommended tearing variables

    /// @brief Check if the block involves multiple coupled equations
    /// @return true if more than one residual or variable
    bool is_coupled() const { return residual_indices.size() > 1 || variable_indices.size() > 1; }
};

/**
 * @brief Block-triangular decomposition and tearing metadata for a selected block.
 */
struct BLTDecomposition {
    SparsityPattern incidence;                    ///< Incidence (Jacobian sparsity) of the block
    std::vector<int> row_permutation;             ///< BTF row permutation
    std::vector<int> column_permutation;          ///< BTF column permutation
    std::vector<int> row_block_offsets;           ///< Fine row block boundaries
    std::vector<int> column_block_offsets;        ///< Fine column block boundaries
    std::vector<int> coarse_row_block_offsets;    ///< Coarse row block boundaries
    std::vector<int> coarse_column_block_offsets; ///< Coarse column block boundaries
    std::vector<StructuralBlock> blocks;          ///< Diagonal blocks with tearing info
};

/**
 * @brief Combined alias-elimination and BLT analysis pass.
 *
 * The BLT decomposition is performed on `alias_elimination.reduced_function`.
 * The block indices therefore refer to the reduced variable/residual ordering.
 */
struct StructuralAnalysis {
    AliasEliminationResult alias_elimination; ///< Alias elimination pass result
    BLTDecomposition blt;                     ///< BLT decomposition of the reduced system
};

namespace detail {

inline void validate_options(const StructuralTransformOptions &opts, const std::string &context) {
    if (opts.input_idx < 0) {
        throw InvalidArgument(context + ": input_idx cannot be negative");
    }
    if (opts.output_idx < 0) {
        throw InvalidArgument(context + ": output_idx cannot be negative");
    }
    if (opts.max_alias_row_nnz <= 0) {
        throw InvalidArgument(context + ": max_alias_row_nnz must be positive");
    }
}

inline void validate_selected_block(const Function &fn, const StructuralTransformOptions &opts,
                                    const std::string &context) {
    const auto &cfn = fn.casadi_function();
    if (opts.input_idx >= cfn.n_in()) {
        throw InvalidArgument(context + ": input_idx out of range");
    }
    if (opts.output_idx >= cfn.n_out()) {
        throw InvalidArgument(context + ": output_idx out of range");
    }

    const auto in_sp = cfn.sparsity_in(opts.input_idx);
    const auto out_sp = cfn.sparsity_out(opts.output_idx);
    if (!in_sp.is_dense() || !in_sp.is_column()) {
        throw InvalidArgument(context + ": selected input must be a dense column vector");
    }
    if (!out_sp.is_dense() || !out_sp.is_column()) {
        throw InvalidArgument(context + ": selected output must be a dense column vector");
    }
    if (cfn.nnz_in(opts.input_idx) != cfn.nnz_out(opts.output_idx)) {
        throw InvalidArgument(context + ": selected input and output dimensions must match");
    }
}

inline std::vector<int> make_index_vector(int n) {
    std::vector<int> indices(static_cast<std::size_t>(n));
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

inline std::vector<casadi::MX> mx_elements(const casadi::MX &v) {
    std::vector<casadi::MX> elems;
    elems.reserve(static_cast<std::size_t>(v.size1() * v.size2()));
    for (int i = 0; i < v.size1(); ++i) {
        for (int j = 0; j < v.size2(); ++j) {
            elems.push_back(v(i, j));
        }
    }
    return elems;
}

inline casadi::MX vertcat_subset(const std::vector<casadi::MX> &elems,
                                 const std::vector<int> &indices) {
    std::vector<casadi::MX> subset;
    subset.reserve(indices.size());
    for (int index : indices) {
        subset.push_back(elems.at(static_cast<std::size_t>(index)));
    }
    return subset.empty() ? casadi::MX(0, 1) : casadi::MX::vertcat(subset);
}

inline std::vector<int> find_row_nonzeros(const casadi::MX &A, int row) {
    std::vector<int> cols;
    cols.reserve(static_cast<std::size_t>(A.size2()));
    for (int col = 0; col < A.size2(); ++col) {
        if (!A(row, col).is_zero()) {
            cols.push_back(col);
        }
    }
    return cols;
}

inline bool has_no_free_symbols(const casadi::MX &expr) {
    try {
        casadi::Function("structural_constant_probe", std::vector<casadi::MX>{},
                         std::vector<casadi::MX>{expr});
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

inline bool coefficients_are_constant(const casadi::MX &A, int row,
                                      const std::vector<int> &nz_cols) {
    for (int col : nz_cols) {
        if (!has_no_free_symbols(A(row, col))) {
            return false;
        }
    }
    return true;
}

struct AliasCandidate {
    int pivot_index = -1;
    casadi::MX replacement;
};

inline bool try_make_alias_candidate(const casadi::MX &expr, const casadi::MX &state_symbol,
                                     const std::vector<int> &active_var_indices,
                                     const StructuralTransformOptions &opts,
                                     AliasCandidate &candidate) {
    casadi::MX A_row;
    casadi::MX b_row;
    try {
        casadi::MX::linear_coeff(casadi::MX::vertcat(std::vector<casadi::MX>{expr}), state_symbol,
                                 A_row, b_row, true);
    } catch (const std::exception &) {
        return false;
    }

    const std::vector<int> nz_cols_full = find_row_nonzeros(A_row, 0);
    std::vector<int> nz_cols;
    nz_cols.reserve(nz_cols_full.size());
    for (int col : nz_cols_full) {
        if (std::find(active_var_indices.begin(), active_var_indices.end(), col) !=
            active_var_indices.end()) {
            nz_cols.push_back(col);
        }
    }

    if (nz_cols.empty() || static_cast<int>(nz_cols.size()) > opts.max_alias_row_nnz) {
        return false;
    }
    if (opts.require_constant_alias_coefficients && !coefficients_are_constant(A_row, 0, nz_cols)) {
        return false;
    }

    int pivot_index = nz_cols.front();
    for (int col : nz_cols) {
        if (col > pivot_index) {
            pivot_index = col;
        }
    }

    casadi::MX rhs = b_row(0);
    for (int col : nz_cols) {
        if (col == pivot_index) {
            continue;
        }
        rhs += A_row(0, col) * state_symbol(col);
    }

    candidate.pivot_index = pivot_index;
    candidate.replacement = -rhs / A_row(0, pivot_index);
    return true;
}

inline std::vector<int> erase_value(const std::vector<int> &values, int erased_index) {
    std::vector<int> result;
    result.reserve(values.size() > 0 ? values.size() - 1 : 0);
    for (int value : values) {
        if (value != erased_index) {
            result.push_back(value);
        }
    }
    return result;
}

inline std::vector<int> casadi_to_int(const std::vector<casadi_int> &values) {
    return std::vector<int>(values.begin(), values.end());
}

struct TarjanState {
    const std::vector<std::vector<int>> &adjacency;
    const std::vector<bool> &removed;
    std::vector<int> index;
    std::vector<int> lowlink;
    std::vector<int> stack;
    std::vector<bool> on_stack;
    int next_index = 0;
    std::vector<std::vector<int>> components;
};

inline void tarjan_visit(int node, TarjanState &state) {
    state.index[node] = state.next_index;
    state.lowlink[node] = state.next_index;
    state.next_index += 1;
    state.stack.push_back(node);
    state.on_stack[node] = true;

    for (int neighbor : state.adjacency[node]) {
        if (state.removed[neighbor]) {
            continue;
        }
        if (state.index[neighbor] < 0) {
            tarjan_visit(neighbor, state);
            state.lowlink[node] = std::min(state.lowlink[node], state.lowlink[neighbor]);
        } else if (state.on_stack[neighbor]) {
            state.lowlink[node] = std::min(state.lowlink[node], state.index[neighbor]);
        }
    }

    if (state.lowlink[node] == state.index[node]) {
        std::vector<int> component;
        while (!state.stack.empty()) {
            const int top = state.stack.back();
            state.stack.pop_back();
            state.on_stack[top] = false;
            component.push_back(top);
            if (top == node) {
                break;
            }
        }
        state.components.push_back(component);
    }
}

inline std::vector<std::vector<int>>
strongly_connected_components(const std::vector<std::vector<int>> &adjacency,
                              const std::vector<bool> &removed) {
    TarjanState state{adjacency,
                      removed,
                      std::vector<int>(adjacency.size(), -1),
                      std::vector<int>(adjacency.size(), -1),
                      {},
                      std::vector<bool>(adjacency.size(), false),
                      0,
                      {}};

    for (std::size_t node = 0; node < adjacency.size(); ++node) {
        if (removed[node] || state.index[node] >= 0) {
            continue;
        }
        tarjan_visit(static_cast<int>(node), state);
    }
    return state.components;
}

inline std::vector<int> tearing_recommendation(const casadi::Sparsity &incidence,
                                               const std::vector<int> &row_indices,
                                               const std::vector<int> &col_indices) {
    const int block_size = static_cast<int>(std::min(row_indices.size(), col_indices.size()));
    if (block_size <= 1) {
        return {};
    }

    std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(block_size));
    for (int local = 0; local < block_size; ++local) {
        const int row = row_indices.at(static_cast<std::size_t>(local));
        const int matched_col = col_indices.at(static_cast<std::size_t>(local));
        for (int dep_local = 0; dep_local < block_size; ++dep_local) {
            const int dep_col = col_indices.at(static_cast<std::size_t>(dep_local));
            if (dep_col == matched_col) {
                continue;
            }
            if (incidence.has_nz(row, dep_col)) {
                adjacency[static_cast<std::size_t>(local)].push_back(dep_local);
            }
        }
        auto &neighbors = adjacency[static_cast<std::size_t>(local)];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    std::vector<bool> removed(static_cast<std::size_t>(block_size), false);
    std::vector<int> tears;

    while (true) {
        const auto components = strongly_connected_components(adjacency, removed);
        std::vector<int> cyclic_component;
        for (const auto &component : components) {
            if (component.size() > 1) {
                cyclic_component = component;
                break;
            }
        }
        if (cyclic_component.empty()) {
            break;
        }

        int best_node = cyclic_component.front();
        int best_score = -1;
        for (int node : cyclic_component) {
            int score = static_cast<int>(adjacency[static_cast<std::size_t>(node)].size());
            for (int other : cyclic_component) {
                if (std::find(adjacency[static_cast<std::size_t>(other)].begin(),
                              adjacency[static_cast<std::size_t>(other)].end(),
                              node) != adjacency[static_cast<std::size_t>(other)].end()) {
                    score += 1;
                }
            }
            if (score > best_score) {
                best_score = score;
                best_node = node;
            }
        }

        removed[static_cast<std::size_t>(best_node)] = true;
        tears.push_back(col_indices.at(static_cast<std::size_t>(best_node)));
    }

    return tears;
}

} // namespace detail

/**
 * @brief Eliminate trivial affine alias rows from a selected residual block
 * @param fn Function containing the residual system
 * @param opts Transform options (input/output indices, alias thresholds)
 * @return Reduced function, reconstruction map, and substitution records
 * @see block_triangularize, structural_analyze
 */
inline AliasEliminationResult alias_eliminate(const Function &fn,
                                              const StructuralTransformOptions &opts = {}) {
    detail::validate_options(opts, "alias_eliminate");
    detail::validate_selected_block(fn, opts, "alias_eliminate");

    const auto &cfn = fn.casadi_function();
    const auto inputs = cfn.mx_in();
    const auto outputs = cfn(inputs);

    const casadi::MX selected_input = inputs.at(static_cast<std::size_t>(opts.input_idx));
    const casadi::MX selected_output = outputs.at(static_cast<std::size_t>(opts.output_idx));

    const casadi::MX state_symbol =
        metis::sym(cfn.name_in(opts.input_idx) + "_struct", selected_input.nnz(), 1);
    std::vector<casadi::MX> state_exprs = detail::mx_elements(state_symbol);
    std::vector<casadi::MX> residuals =
        detail::mx_elements(casadi::MX::substitute(std::vector<casadi::MX>{selected_output},
                                                   std::vector<casadi::MX>{selected_input},
                                                   std::vector<casadi::MX>{state_symbol})
                                .front());

    std::vector<int> active_var_indices = detail::make_index_vector(selected_input.nnz());
    std::vector<int> active_residual_indices = detail::make_index_vector(selected_output.nnz());

    struct RawAlias {
        int residual_index;
        int variable_index;
        casadi::MX replacement;
    };
    std::vector<RawAlias> raw_aliases;

    bool progress = true;
    while (progress) {
        progress = false;
        for (int residual_index : active_residual_indices) {
            detail::AliasCandidate candidate;
            if (!detail::try_make_alias_candidate(
                    residuals.at(static_cast<std::size_t>(residual_index)), state_symbol,
                    active_var_indices, opts, candidate)) {
                continue;
            }

            const int pivot_original = candidate.pivot_index;
            std::vector<casadi::MX> substitution_exprs = detail::mx_elements(state_symbol);
            substitution_exprs.at(static_cast<std::size_t>(pivot_original)) = candidate.replacement;
            const casadi::MX substitution_vector = casadi::MX::vertcat(substitution_exprs);

            residuals = casadi::MX::substitute(residuals, std::vector<casadi::MX>{state_symbol},
                                               std::vector<casadi::MX>{substitution_vector});
            state_exprs = casadi::MX::substitute(state_exprs, std::vector<casadi::MX>{state_symbol},
                                                 std::vector<casadi::MX>{substitution_vector});

            raw_aliases.push_back(RawAlias{residual_index, pivot_original, candidate.replacement});
            active_var_indices = detail::erase_value(active_var_indices, pivot_original);
            active_residual_indices = detail::erase_value(active_residual_indices, residual_index);
            progress = true;
            break;
        }
    }

    std::vector<int> eliminated_variable_indices;
    std::vector<int> eliminated_residual_indices;
    eliminated_variable_indices.reserve(raw_aliases.size());
    eliminated_residual_indices.reserve(raw_aliases.size());
    for (const auto &alias : raw_aliases) {
        eliminated_variable_indices.push_back(alias.variable_index);
        eliminated_residual_indices.push_back(alias.residual_index);
    }

    std::vector<casadi::MX> wrapper_inputs_mx;
    std::vector<SymbolicArg> wrapper_inputs;
    wrapper_inputs_mx.reserve(static_cast<std::size_t>(cfn.n_in()));
    wrapper_inputs.reserve(static_cast<std::size_t>(cfn.n_in()));

    casadi::MX reduced_input;
    for (int i = 0; i < cfn.n_in(); ++i) {
        if (i == opts.input_idx) {
            reduced_input =
                metis::sym(cfn.name_in(i), static_cast<int>(active_var_indices.size()), 1);
            wrapper_inputs_mx.push_back(reduced_input);
            wrapper_inputs.emplace_back(reduced_input);
        } else {
            casadi::MX arg = metis::sym(cfn.name_in(i), cfn.size1_in(i), cfn.size2_in(i));
            wrapper_inputs_mx.push_back(arg);
            wrapper_inputs.emplace_back(arg);
        }
    }

    std::vector<casadi::MX> subs_from;
    std::vector<casadi::MX> subs_to;
    subs_from.reserve(static_cast<std::size_t>(cfn.n_in()));
    subs_to.reserve(static_cast<std::size_t>(cfn.n_in()));

    std::vector<casadi::MX> reduced_state_substitution(selected_input.nnz(), 0.0);
    for (std::size_t k = 0; k < active_var_indices.size(); ++k) {
        reduced_state_substitution.at(static_cast<std::size_t>(active_var_indices[k])) =
            reduced_input(static_cast<int>(k));
    }
    subs_from.push_back(state_symbol);
    subs_to.push_back(casadi::MX::vertcat(reduced_state_substitution));
    for (int i = 0; i < cfn.n_in(); ++i) {
        if (i == opts.input_idx) {
            continue;
        }
        subs_from.push_back(inputs.at(static_cast<std::size_t>(i)));
        subs_to.push_back(wrapper_inputs_mx.at(static_cast<std::size_t>(i)));
    }

    const std::vector<casadi::MX> reduced_residual_exprs =
        casadi::MX::substitute(residuals, subs_from, subs_to);
    const std::vector<casadi::MX> full_state_exprs =
        casadi::MX::substitute(state_exprs, subs_from, subs_to);

    const casadi::MX reduced_residual =
        detail::vertcat_subset(reduced_residual_exprs, active_residual_indices);
    const casadi::MX reconstructed_state =
        full_state_exprs.empty() ? casadi::MX(0, 1) : casadi::MX::vertcat(full_state_exprs);

    std::vector<AliasSubstitution> substitutions;
    substitutions.reserve(raw_aliases.size());
    for (const auto &alias : raw_aliases) {
        const casadi::MX replacement =
            casadi::MX::substitute(std::vector<casadi::MX>{alias.replacement}, subs_from, subs_to)
                .front();
        substitutions.push_back(
            AliasSubstitution{alias.residual_index, alias.variable_index, replacement});
    }

    return AliasEliminationResult{
        Function(cfn.name() + "_alias_reduced", wrapper_inputs,
                 std::vector<SymbolicArg>{SymbolicArg(reduced_residual)}),
        Function(cfn.name() + "_alias_reconstruct", wrapper_inputs,
                 std::vector<SymbolicArg>{SymbolicArg(reconstructed_state)}),
        active_var_indices,
        eliminated_variable_indices,
        active_residual_indices,
        eliminated_residual_indices,
        substitutions,
    };
}

/**
 * @brief Compute a block-triangular decomposition of a selected residual block
 * @param fn Function with a square residual system
 * @param opts Transform options (input/output indices)
 * @return BLT decomposition with permutations and block structure
 * @see alias_eliminate, structural_analyze
 */
inline BLTDecomposition block_triangularize(const Function &fn,
                                            const StructuralTransformOptions &opts = {}) {
    detail::validate_options(opts, "block_triangularize");
    detail::validate_selected_block(fn, opts, "block_triangularize");

    const casadi::Sparsity incidence_sp =
        get_jacobian_sparsity(fn, opts.output_idx, opts.input_idx).casadi_sparsity();

    std::vector<casadi_int> rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock;
    incidence_sp.btf(rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock);

    std::vector<StructuralBlock> blocks;
    const std::size_t n_blocks = rowblock.size() > 0 ? rowblock.size() - 1 : 0;
    blocks.reserve(n_blocks);
    for (std::size_t block = 0; block < n_blocks; ++block) {
        std::vector<int> block_rows;
        std::vector<int> block_cols;
        for (casadi_int k = rowblock[block]; k < rowblock[block + 1]; ++k) {
            block_rows.push_back(static_cast<int>(rowperm[k]));
        }
        for (casadi_int k = colblock[block]; k < colblock[block + 1]; ++k) {
            block_cols.push_back(static_cast<int>(colperm[k]));
        }

        blocks.push_back(StructuralBlock{
            block_rows,
            block_cols,
            detail::tearing_recommendation(incidence_sp, block_rows, block_cols),
        });
    }

    return BLTDecomposition{
        SparsityPattern(incidence_sp),          detail::casadi_to_int(rowperm),
        detail::casadi_to_int(colperm),         detail::casadi_to_int(rowblock),
        detail::casadi_to_int(colblock),        detail::casadi_to_int(coarse_rowblock),
        detail::casadi_to_int(coarse_colblock), blocks,
    };
}

/**
 * @brief Run alias elimination followed by BLT decomposition
 * @param fn Function containing the residual system
 * @param opts Transform options
 * @return Combined alias-elimination and BLT analysis
 * @see alias_eliminate, block_triangularize
 */
inline StructuralAnalysis structural_analyze(const Function &fn,
                                             const StructuralTransformOptions &opts = {}) {
    AliasEliminationResult alias = alias_eliminate(fn, opts);

    StructuralTransformOptions reduced_opts = opts;
    reduced_opts.output_idx = 0;
    BLTDecomposition blt = block_triangularize(alias.reduced_function, reduced_opts);

    return StructuralAnalysis{std::move(alias), std::move(blt)};
}

} // namespace metis
