/// @file Diagnostics.hpp
/// @brief Structural observability and identifiability analysis
#pragma once

#include "Function.hpp"
#include "MetisError.hpp"
#include "Sparsity.hpp"

#include <algorithm>
#include <casadi/casadi.hpp>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace metis {

/**
 * @brief Structural property being analyzed from a symbolic sensitivity pattern.
 */
enum class StructuralProperty {
    Observability,   ///< State observability from outputs
    Identifiability, ///< Parameter identifiability from outputs
};

/**
 * @brief Output-selection options shared by the structural diagnostics helpers.
 *
 * If `output_indices` is empty, all function outputs are flattened and analyzed.
 */
struct StructuralSensitivityOptions {
    std::vector<int> output_indices; ///< Indices of outputs to analyze (empty = all)
};

/**
 * @brief One scalarized element of the selected input block.
 */
struct DiagnosticInputRef {
    int local_index = -1; ///< Scalar index within the input block
    std::string label;    ///< Human-readable label (e.g. "x[0]")
};

/**
 * @brief One scalarized row in the selected output stack.
 */
struct DiagnosticOutputRef {
    int flat_row = -1;   ///< Row in the flattened output stack
    int output_idx = -1; ///< Index of the output block
    int local_row = -1;  ///< Row within the output block
    int local_col = -1;  ///< Column within the output block
    std::string label;   ///< Human-readable label
};

/**
 * @brief One structurally deficient connected component in the sensitivity graph.
 */
struct StructuralDeficiencyGroup {
    std::vector<int> input_local_indices; ///< Inputs in this connected component
    std::vector<int> output_rows;         ///< Outputs in this connected component
    int structural_rank = 0;              ///< Rank of this sub-block
    int rank_deficiency = 0;              ///< Number of structurally deficient inputs
};

/**
 * @brief One user-facing structural diagnostic with an attached remediation hint.
 */
struct StructuralDiagnosticIssue {
    std::string message;                  ///< Human-readable diagnostic message
    std::vector<int> input_local_indices; ///< Affected input indices
    std::vector<int> output_rows;         ///< Affected output rows
};

/**
 * @brief Structural rank analysis of selected outputs with respect to one input block.
 */
struct StructuralSensitivityReport {
    StructuralProperty property = StructuralProperty::Observability; ///< Analysis type
    int input_idx = -1;                                              ///< Analyzed input block index
    std::string input_name;                                          ///< Name of the input block
    std::vector<int> output_indices;                                 ///< Selected output indices
    std::vector<std::string> output_names;                           ///< Names of selected outputs
    int variable_dimension = 0;                                      ///< Number of scalar variables
    int output_dimension = 0;                                        ///< Number of scalar outputs
    int structural_rank = 0;                         ///< Structural rank of Jacobian
    int rank_deficiency = 0;                         ///< variable_dimension - structural_rank
    SparsityPattern sensitivity_sparsity;            ///< Jacobian sparsity pattern
    std::vector<DiagnosticInputRef> inputs;          ///< Per-element input metadata
    std::vector<DiagnosticOutputRef> outputs;        ///< Per-element output metadata
    std::vector<int> deficient_local_indices;        ///< All structurally deficient inputs
    std::vector<int> zero_sensitivity_local_indices; ///< Inputs with zero Jacobian columns
    std::vector<StructuralDeficiencyGroup> deficiency_groups; ///< Connected deficient components
    std::vector<StructuralDiagnosticIssue> issues;            ///< User-facing diagnostic messages

    /// @brief Check if the Jacobian has full structural rank
    /// @return true if rank_deficiency == 0
    bool full_rank() const { return rank_deficiency == 0; }
};

/**
 * @brief Combined observability and identifiability analysis options.
 *
 * Leave either input index negative to skip that analysis. At least one of
 * `state_input_idx` or `parameter_input_idx` must be non-negative.
 */
struct StructuralDiagnosticsOptions {
    std::vector<int> output_indices; ///< Output indices to analyze (empty = all)
    int state_input_idx = -1;        ///< State input block index (-1 to skip observability)
    int parameter_input_idx = -1;    ///< Parameter input block index (-1 to skip identifiability)
};

/**
 * @brief Combined structural diagnostics report.
 */
struct StructuralDiagnosticsReport {
    std::optional<StructuralSensitivityReport>
        observability; ///< Observability result (if requested)
    std::optional<StructuralSensitivityReport>
        identifiability; ///< Identifiability result (if requested)

    /// @brief Check if any analysis found a rank deficiency
    /// @return true if observability or identifiability is deficient
    bool has_deficiency() const {
        return (observability.has_value() && !observability->full_rank()) ||
               (identifiability.has_value() && !identifiability->full_rank());
    }
};

namespace detail {

inline std::vector<int> sort_unique(std::vector<int> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

inline std::vector<casadi_int> to_casadi_int(const std::vector<int> &values) {
    return std::vector<casadi_int>(values.begin(), values.end());
}

inline std::string join_labels(const std::vector<std::string> &labels) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << labels[i];
    }
    return oss.str();
}

inline std::string property_name(StructuralProperty property) {
    switch (property) {
    case StructuralProperty::Observability:
        return "observability";
    case StructuralProperty::Identifiability:
        return "identifiability";
    }
    throw InvalidArgument("Unsupported structural property");
}

inline std::string property_subject_plural(StructuralProperty property) {
    switch (property) {
    case StructuralProperty::Observability:
        return "states";
    case StructuralProperty::Identifiability:
        return "parameters";
    }
    throw InvalidArgument("Unsupported structural property");
}

inline std::string property_subject_singular(StructuralProperty property) {
    switch (property) {
    case StructuralProperty::Observability:
        return "state";
    case StructuralProperty::Identifiability:
        return "parameter";
    }
    throw InvalidArgument("Unsupported structural property");
}

inline std::string property_fix_hint(StructuralProperty property) {
    switch (property) {
    case StructuralProperty::Observability:
        return "add sensors that depend on them or constrain/fix them";
    case StructuralProperty::Identifiability:
        return "add measurements that separate them or constrain/fix them";
    }
    throw InvalidArgument("Unsupported structural property");
}

inline std::string input_label(const std::string &input_name, int local_index) {
    return input_name + "[" + std::to_string(local_index) + "]";
}

inline std::string output_label(const std::string &output_name, int rows, int cols,
                                int flat_index) {
    if (rows == 1 && cols == 1) {
        return output_name;
    }
    const int row = flat_index % rows;
    const int col = flat_index / rows;
    if (cols == 1) {
        return output_name + "[" + std::to_string(row) + "]";
    }
    if (rows == 1) {
        return output_name + "[" + std::to_string(col) + "]";
    }
    return output_name + "(" + std::to_string(row) + "," + std::to_string(col) + ")";
}

inline std::vector<int> canonical_output_indices(const casadi::Function &cfn,
                                                 const std::vector<int> &requested,
                                                 const std::string &context) {
    if (requested.empty()) {
        std::vector<int> all(static_cast<std::size_t>(cfn.n_out()));
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    std::vector<int> indices;
    indices.reserve(requested.size());
    for (int output_idx : requested) {
        if (output_idx < 0 || output_idx >= cfn.n_out()) {
            throw InvalidArgument(context + ": output index out of range");
        }
        indices.push_back(output_idx);
    }
    indices = sort_unique(std::move(indices));
    if (indices.size() != requested.size()) {
        throw InvalidArgument(context + ": output_indices must not contain duplicates");
    }
    return indices;
}

inline void validate_input_block(const casadi::Function &cfn, int input_idx,
                                 const std::string &context) {
    if (input_idx < 0 || input_idx >= cfn.n_in()) {
        throw InvalidArgument(context + ": input_idx out of range");
    }

    const casadi::Sparsity input_sp = cfn.sparsity_in(input_idx);
    if (!input_sp.is_dense() || !input_sp.is_column()) {
        throw InvalidArgument(context + ": selected input must be a dense column vector");
    }
}

inline void validate_output_blocks(const casadi::Function &cfn,
                                   const std::vector<int> &output_indices,
                                   const std::string &context) {
    for (int output_idx : output_indices) {
        const casadi::Sparsity output_sp = cfn.sparsity_out(output_idx);
        if (!output_sp.is_dense()) {
            throw InvalidArgument(context + ": selected outputs must be dense");
        }
    }
}

inline casadi::MX flatten_output(const casadi::MX &output) {
    return casadi::MX::reshape(output, output.numel(), 1);
}

inline std::vector<DiagnosticInputRef> make_input_refs(const casadi::Function &cfn, int input_idx) {
    std::vector<DiagnosticInputRef> refs;
    refs.reserve(static_cast<std::size_t>(cfn.nnz_in(input_idx)));
    for (int local_index = 0; local_index < cfn.nnz_in(input_idx); ++local_index) {
        refs.push_back(DiagnosticInputRef{
            local_index,
            input_label(cfn.name_in(input_idx), local_index),
        });
    }
    return refs;
}

inline std::vector<DiagnosticOutputRef> make_output_refs(const casadi::Function &cfn,
                                                         const std::vector<int> &output_indices) {
    std::vector<DiagnosticOutputRef> refs;
    int flat_row = 0;
    for (int output_idx : output_indices) {
        const int rows = cfn.size1_out(output_idx);
        const int cols = cfn.size2_out(output_idx);
        const std::string &name = cfn.name_out(output_idx);
        for (int linear = 0; linear < rows * cols; ++linear) {
            refs.push_back(DiagnosticOutputRef{
                flat_row,
                output_idx,
                linear % rows,
                linear / rows,
                output_label(name, rows, cols, linear),
            });
            flat_row += 1;
        }
    }
    return refs;
}

inline casadi::MX collect_selected_outputs(const std::vector<casadi::MX> &outputs,
                                           const std::vector<int> &output_indices) {
    std::vector<casadi::MX> flattened;
    flattened.reserve(output_indices.size());
    for (int output_idx : output_indices) {
        flattened.push_back(flatten_output(outputs.at(static_cast<std::size_t>(output_idx))));
    }
    if (flattened.empty()) {
        return casadi::MX(0, 1);
    }
    return casadi::MX::vertcat(flattened);
}

struct BipartiteComponent {
    std::vector<int> rows;
    std::vector<int> cols;
};

inline std::vector<BipartiteComponent> connected_components(const casadi::Sparsity &sp) {
    const int n_rows = static_cast<int>(sp.size1());
    const int n_cols = static_cast<int>(sp.size2());

    std::vector<casadi_int> row_indices_ci;
    std::vector<casadi_int> col_indices_ci;
    sp.get_triplet(row_indices_ci, col_indices_ci);

    std::vector<std::vector<int>> row_to_cols(static_cast<std::size_t>(n_rows));
    std::vector<std::vector<int>> col_to_rows(static_cast<std::size_t>(n_cols));
    for (std::size_t k = 0; k < row_indices_ci.size(); ++k) {
        const int row = static_cast<int>(row_indices_ci[k]);
        const int col = static_cast<int>(col_indices_ci[k]);
        row_to_cols.at(static_cast<std::size_t>(row)).push_back(col);
        col_to_rows.at(static_cast<std::size_t>(col)).push_back(row);
    }

    std::vector<bool> row_visited(static_cast<std::size_t>(n_rows), false);
    std::vector<bool> col_visited(static_cast<std::size_t>(n_cols), false);
    std::vector<BipartiteComponent> components;

    for (int start_col = 0; start_col < n_cols; ++start_col) {
        if (col_visited.at(static_cast<std::size_t>(start_col))) {
            continue;
        }

        BipartiteComponent component;
        std::queue<std::pair<bool, int>> frontier;
        frontier.push({false, start_col});
        col_visited.at(static_cast<std::size_t>(start_col)) = true;

        while (!frontier.empty()) {
            const auto [is_row, index] = frontier.front();
            frontier.pop();

            if (is_row) {
                component.rows.push_back(index);
                for (int col : row_to_cols.at(static_cast<std::size_t>(index))) {
                    if (!col_visited.at(static_cast<std::size_t>(col))) {
                        col_visited.at(static_cast<std::size_t>(col)) = true;
                        frontier.push({false, col});
                    }
                }
            } else {
                component.cols.push_back(index);
                for (int row : col_to_rows.at(static_cast<std::size_t>(index))) {
                    if (!row_visited.at(static_cast<std::size_t>(row))) {
                        row_visited.at(static_cast<std::size_t>(row)) = true;
                        frontier.push({true, row});
                    }
                }
            }
        }

        component.rows = sort_unique(std::move(component.rows));
        component.cols = sort_unique(std::move(component.cols));
        components.push_back(std::move(component));
    }

    return components;
}

inline int structural_rank(const casadi::Sparsity &sp) {
    return static_cast<int>(casadi::Sparsity::sprank(sp));
}

inline int structural_rank_of_component(const casadi::Sparsity &sp,
                                        const BipartiteComponent &component) {
    const std::vector<casadi_int> rows = to_casadi_int(component.rows);
    const std::vector<casadi_int> cols = to_casadi_int(component.cols);
    std::vector<casadi_int> mapping;
    const casadi::Sparsity sub = sp.sub(rows, cols, mapping);
    return structural_rank(sub);
}

inline std::vector<int> zero_sensitivity_columns(const casadi::Sparsity &sp) {
    std::vector<int> zeros;
    zeros.reserve(static_cast<std::size_t>(sp.size2()));
    for (int col = 0; col < sp.size2(); ++col) {
        bool has_nonzero = false;
        for (int nz = sp.colind(col); nz < sp.colind(col + 1); ++nz) {
            if (sp.row(nz) >= 0) {
                has_nonzero = true;
                break;
            }
        }
        if (!has_nonzero) {
            zeros.push_back(col);
        }
    }
    return zeros;
}

inline std::vector<std::string> labels_for_inputs(const std::vector<DiagnosticInputRef> &inputs,
                                                  const std::vector<int> &indices) {
    std::vector<std::string> labels;
    labels.reserve(indices.size());
    for (int index : indices) {
        labels.push_back(inputs.at(static_cast<std::size_t>(index)).label);
    }
    return labels;
}

inline std::vector<std::string> labels_for_outputs(const std::vector<DiagnosticOutputRef> &outputs,
                                                   const std::vector<int> &rows) {
    std::vector<std::string> labels;
    labels.reserve(rows.size());
    for (int row : rows) {
        labels.push_back(outputs.at(static_cast<std::size_t>(row)).label);
    }
    return labels;
}

inline std::vector<StructuralDiagnosticIssue>
build_issues(StructuralProperty property, const std::vector<DiagnosticInputRef> &inputs,
             const std::vector<DiagnosticOutputRef> &outputs,
             const std::vector<int> &zero_sensitivity_local_indices,
             const std::vector<StructuralDeficiencyGroup> &deficiency_groups) {
    std::vector<StructuralDiagnosticIssue> issues;

    if (!zero_sensitivity_local_indices.empty()) {
        issues.push_back(StructuralDiagnosticIssue{
            "Selected outputs have no structural dependence on " +
                join_labels(labels_for_inputs(inputs, zero_sensitivity_local_indices)) + "; " +
                property_fix_hint(property) + ".",
            zero_sensitivity_local_indices,
            {},
        });
    }

    for (const auto &group : deficiency_groups) {
        if (group.output_rows.empty()) {
            continue;
        }
        issues.push_back(StructuralDiagnosticIssue{
            "Selected outputs only provide structural rank " +
                std::to_string(group.structural_rank) + " for " +
                std::to_string(group.input_local_indices.size()) + " " +
                property_subject_plural(property) + " in {" +
                join_labels(labels_for_inputs(inputs, group.input_local_indices)) +
                "}. Add "
                "independent measurements involving {" +
                join_labels(labels_for_outputs(outputs, group.output_rows)) + "} or " +
                property_fix_hint(property) + ".",
            group.input_local_indices,
            group.output_rows,
        });
    }

    return issues;
}

inline StructuralSensitivityReport analyze_property(const Function &fn, int input_idx,
                                                    StructuralProperty property,
                                                    const StructuralSensitivityOptions &opts) {
    const std::string context = "analyze_structural_" + property_name(property);
    const casadi::Function &cfn = fn.casadi_function();

    validate_input_block(cfn, input_idx, context);
    const std::vector<int> output_indices =
        canonical_output_indices(cfn, opts.output_indices, context);
    validate_output_blocks(cfn, output_indices, context);

    const std::vector<casadi::MX> inputs = cfn.mx_in();
    const std::vector<casadi::MX> outputs = cfn(inputs);

    const casadi::MX selected_input = inputs.at(static_cast<std::size_t>(input_idx));
    const casadi::MX selected_outputs = collect_selected_outputs(outputs, output_indices);
    const casadi::Sparsity jac_sp =
        casadi::MX::jacobian(selected_outputs, selected_input).sparsity();

    StructuralSensitivityReport report;
    report.property = property;
    report.input_idx = input_idx;
    report.input_name = cfn.name_in(input_idx);
    report.output_indices = output_indices;
    report.variable_dimension = static_cast<int>(selected_input.numel());
    report.output_dimension = static_cast<int>(selected_outputs.numel());
    report.structural_rank = structural_rank(jac_sp);
    report.rank_deficiency = report.variable_dimension - report.structural_rank;
    report.sensitivity_sparsity = SparsityPattern(jac_sp);
    report.inputs = make_input_refs(cfn, input_idx);
    report.outputs = make_output_refs(cfn, output_indices);
    report.output_names.reserve(output_indices.size());
    for (int output_idx : output_indices) {
        report.output_names.push_back(cfn.name_out(output_idx));
    }

    report.zero_sensitivity_local_indices = zero_sensitivity_columns(jac_sp);

    const std::vector<BipartiteComponent> components = connected_components(jac_sp);
    for (const auto &component : components) {
        if (component.cols.empty()) {
            continue;
        }

        const int component_rank = structural_rank_of_component(jac_sp, component);
        if (component_rank < static_cast<int>(component.cols.size())) {
            report.deficiency_groups.push_back(StructuralDeficiencyGroup{
                component.cols,
                component.rows,
                component_rank,
                static_cast<int>(component.cols.size()) - component_rank,
            });
        }
    }

    std::vector<int> deficient = report.zero_sensitivity_local_indices;
    for (const auto &group : report.deficiency_groups) {
        deficient.insert(deficient.end(), group.input_local_indices.begin(),
                         group.input_local_indices.end());
    }
    report.deficient_local_indices = sort_unique(std::move(deficient));

    report.issues = build_issues(property, report.inputs, report.outputs,
                                 report.zero_sensitivity_local_indices, report.deficiency_groups);
    return report;
}

} // namespace detail

/**
 * @brief Analyze which states are structurally observable from selected outputs
 * @param fn Function whose Jacobian sparsity is analyzed
 * @param state_input_idx Index of the state input block
 * @param opts Output selection options
 * @return Structural sensitivity report
 * @see analyze_structural_identifiability, analyze_structural_diagnostics
 */
inline StructuralSensitivityReport
analyze_structural_observability(const Function &fn, int state_input_idx = 0,
                                 const StructuralSensitivityOptions &opts = {}) {
    return detail::analyze_property(fn, state_input_idx, StructuralProperty::Observability, opts);
}

/**
 * @brief Analyze which parameters are structurally identifiable from selected outputs
 * @param fn Function whose Jacobian sparsity is analyzed
 * @param parameter_input_idx Index of the parameter input block
 * @param opts Output selection options
 * @return Structural sensitivity report
 * @see analyze_structural_observability
 */
inline StructuralSensitivityReport
analyze_structural_identifiability(const Function &fn, int parameter_input_idx,
                                   const StructuralSensitivityOptions &opts = {}) {
    return detail::analyze_property(fn, parameter_input_idx, StructuralProperty::Identifiability,
                                    opts);
}

/**
 * @brief Run structural observability and identifiability checks together
 * @param fn Function to analyze
 * @param opts Combined analysis options (at least one input index must be non-negative)
 * @return Combined diagnostics report
 * @see analyze_structural_observability, analyze_structural_identifiability
 */
inline StructuralDiagnosticsReport
analyze_structural_diagnostics(const Function &fn, const StructuralDiagnosticsOptions &opts) {
    if (opts.state_input_idx < 0 && opts.parameter_input_idx < 0) {
        throw InvalidArgument("analyze_structural_diagnostics: at least one of state_input_idx or "
                              "parameter_input_idx must be non-negative");
    }

    StructuralDiagnosticsReport report;
    StructuralSensitivityOptions sensitivity_opts;
    sensitivity_opts.output_indices = opts.output_indices;

    if (opts.state_input_idx >= 0) {
        report.observability =
            analyze_structural_observability(fn, opts.state_input_idx, sensitivity_opts);
    }
    if (opts.parameter_input_idx >= 0) {
        report.identifiability =
            analyze_structural_identifiability(fn, opts.parameter_input_idx, sensitivity_opts);
    }
    return report;
}

} // namespace metis
