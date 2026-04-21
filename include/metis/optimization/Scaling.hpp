/**
 * @file Scaling.hpp
 * @brief Scaling diagnostics and metadata for optimization problems
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

namespace metis {

/**
 * @brief Severity used by Opti scaling diagnostics
 *
 * @see ScalingIssue for individual diagnostic items
 * @see Opti::analyze_scaling for generating diagnostics
 */
enum class ScalingIssueLevel {
    Warning, ///< Potential scaling concern
    Critical ///< Severe scaling issue likely to cause solver failure
};

/**
 * @brief Diagnostic category for a scaling issue
 */
enum class ScalingIssueKind {
    Variable,   ///< Issue with variable scaling
    Constraint, ///< Issue with constraint scaling
    Objective,  ///< Issue with objective scaling
    Summary     ///< Aggregate issue across the problem
};

/**
 * @brief Thresholds controlling Opti scaling diagnostics
 */
struct ScalingAnalysisOptions {
    double normalized_low_warn = 1e-3;      ///< Warn when |value| / scale falls below this
    double normalized_high_warn = 1e3;      ///< Warn when |value| / scale exceeds this
    double normalized_high_critical = 1e6;  ///< Escalate to critical beyond this
    double variable_scale_ratio_warn = 1e6; ///< Warn when max(scale) / min(scale) exceeds this
};

/**
 * @brief One diagnostic item produced by `Opti::analyze_scaling()`
 *
 * @see ScalingReport for the aggregate container
 */
struct ScalingIssue {
    ScalingIssueLevel level = ScalingIssueLevel::Warning;
    ScalingIssueKind kind = ScalingIssueKind::Variable;
    int index = -1;                    ///< Variable block index or constraint row
    std::string label;                 ///< Human-readable identifier
    std::string message;               ///< Explanation and suggested action
    double raw_magnitude = 0.0;        ///< Magnitude in physical units
    double applied_scale = 1.0;        ///< Current scale
    double normalized_magnitude = 0.0; ///< |raw| / scale
    double suggested_scale = 1.0;      ///< Heuristic recommendation
};

/**
 * @brief Scaling metadata for one declared variable block
 */
struct VariableScalingInfo {
    int block_index = -1;
    int size = 0;
    std::string category = "Uncategorized";
    bool frozen = false;
    bool user_supplied_scale = false;
    double scale = 1.0;
    double init_abs_mean = 0.0;
    double init_abs_max = 0.0;
    double normalized_init_abs_mean = 0.0;
    double normalized_init_abs_max = 0.0;
    std::optional<double> lower_bound;
    std::optional<double> upper_bound;
    double suggested_scale = 1.0;
};

/**
 * @brief Scaling metadata for one scalarized constraint row
 */
struct ConstraintScalingInfo {
    int row = -1;
    bool has_lower_bound = false;
    bool has_upper_bound = false;
    bool equality = false;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
    double value_at_initial = 0.0;
    double scale = 1.0;
    double normalized_magnitude = 0.0;
    double normalized_violation = 0.0;
    double suggested_scale = 1.0;
};

/**
 * @brief Scaling metadata for the current objective
 */
struct ObjectiveScalingInfo {
    bool configured = false;
    bool maximize = false;
    bool user_supplied_scale = false;
    double value_at_initial = 0.0;
    double scale = 1.0;
    double normalized_value = 0.0;
    double suggested_scale = 1.0;
};

/**
 * @brief Top-level scalar summary for an Opti scaling report
 */
struct ScalingSummary {
    int variable_blocks = 0;
    int scalar_constraints = 0;
    int jacobian_nnz = 0;
    double jacobian_density = 0.0;
    double min_variable_scale = 1.0;
    double max_variable_scale = 1.0;
    double variable_scale_ratio = 1.0;
};

/**
 * @brief Aggregate result returned by `Opti::analyze_scaling()`
 *
 * @see Opti::analyze_scaling for generating this report
 */
struct ScalingReport {
    ScalingSummary summary;                         ///< Top-level numeric summary
    ObjectiveScalingInfo objective;                 ///< Objective scaling metadata
    std::vector<VariableScalingInfo> variables;     ///< Per-variable-block metadata
    std::vector<ConstraintScalingInfo> constraints; ///< Per-constraint-row metadata
    std::vector<ScalingIssue> issues;               ///< Detected scaling issues

    /// @brief Check if any scaling issues were detected
    /// @return true if the issues list is non-empty
    bool has_issues() const { return !issues.empty(); }
};

} // namespace metis
