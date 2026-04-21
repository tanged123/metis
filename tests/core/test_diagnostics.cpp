#include <gtest/gtest.h>
#include <metis/core/Diagnostics.hpp>
#include <metis/core/Function.hpp>
#include <metis/core/MetisError.hpp>
#include <metis/core/MetisTypes.hpp>

#include <algorithm>
#include <string>
#include <vector>

using namespace metis;

namespace {

bool contains_message(const std::vector<StructuralDiagnosticIssue> &issues,
                      const std::string &needle) {
    return std::any_of(issues.begin(), issues.end(), [&](const StructuralDiagnosticIssue &issue) {
        return issue.message.find(needle) != std::string::npos;
    });
}

} // namespace

TEST(StructuralDiagnosticsTests, ObservabilityDetectsZeroSensitivityState) {
    auto x = sym("x", 3, 1);
    auto y = SymbolicScalar::vertcat({
        x(0) + x(1),
        x(1),
    });

    Function fn("observe_subset", {x}, {y});
    auto report = analyze_structural_observability(fn);

    EXPECT_EQ(report.variable_dimension, 3);
    EXPECT_EQ(report.output_dimension, 2);
    EXPECT_EQ(report.structural_rank, 2);
    EXPECT_EQ(report.rank_deficiency, 1);
    EXPECT_EQ(report.zero_sensitivity_local_indices, std::vector<int>({2}));
    EXPECT_EQ(report.deficient_local_indices, std::vector<int>({2}));
    EXPECT_FALSE(report.full_rank());
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues.front().input_local_indices, std::vector<int>({2}));
    EXPECT_TRUE(contains_message(report.issues, "no structural dependence"));
    EXPECT_TRUE(contains_message(report.issues, "add sensors"));
}

TEST(StructuralDiagnosticsTests, IdentifiabilityFlagsCoupledParameterGroup) {
    auto x = sym("x");
    auto p = sym("p", 3, 1);
    auto y = SymbolicScalar::vertcat({
        p(0) + p(1) + x,
        p(1) + p(2),
    });

    Function fn("identify_group", {x, p}, {y});
    auto report = analyze_structural_identifiability(fn, 1);

    EXPECT_EQ(report.structural_rank, 2);
    EXPECT_EQ(report.rank_deficiency, 1);
    EXPECT_TRUE(report.zero_sensitivity_local_indices.empty());
    EXPECT_EQ(report.deficient_local_indices, std::vector<int>({0, 1, 2}));
    ASSERT_EQ(report.deficiency_groups.size(), 1u);
    EXPECT_EQ(report.deficiency_groups.front().input_local_indices, std::vector<int>({0, 1, 2}));
    EXPECT_EQ(report.deficiency_groups.front().output_rows, std::vector<int>({0, 1}));
    EXPECT_EQ(report.deficiency_groups.front().structural_rank, 2);
    EXPECT_EQ(report.deficiency_groups.front().rank_deficiency, 1);
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues.front().input_local_indices, std::vector<int>({0, 1, 2}));
    EXPECT_EQ(report.issues.front().output_rows, std::vector<int>({0, 1}));
    EXPECT_TRUE(contains_message(report.issues, "structural rank 2"));
    EXPECT_TRUE(contains_message(report.issues, "constrain/fix"));
}

TEST(StructuralDiagnosticsTests, ObservabilityRespectsSelectedOutputSubset) {
    auto x = sym("x", 2, 1);
    auto y0 = x(0);
    auto y1 = x(1);

    Function fn("output_subset", {x}, {y0, y1});

    StructuralSensitivityOptions opts;
    opts.output_indices = {0};
    auto report = analyze_structural_observability(fn, 0, opts);

    EXPECT_EQ(report.output_indices, std::vector<int>({0}));
    EXPECT_EQ(report.output_names, std::vector<std::string>({"o0"}));
    EXPECT_EQ(report.output_dimension, 1);
    EXPECT_EQ(report.structural_rank, 1);
    EXPECT_EQ(report.zero_sensitivity_local_indices, std::vector<int>({1}));
    EXPECT_EQ(report.deficient_local_indices, std::vector<int>({1}));
}

TEST(StructuralDiagnosticsTests, CombinedDiagnosticsReportsObservabilityAndIdentifiability) {
    auto x = sym("x", 2, 1);
    auto p = sym("p", 2, 1);
    auto y = SymbolicScalar::vertcat({
        x(0),
        x(1) + p(0),
    });

    Function fn("combined_diagnostics", {x, p}, {y});

    StructuralDiagnosticsOptions opts;
    opts.state_input_idx = 0;
    opts.parameter_input_idx = 1;
    auto report = analyze_structural_diagnostics(fn, opts);

    ASSERT_TRUE(report.observability.has_value());
    ASSERT_TRUE(report.identifiability.has_value());
    EXPECT_TRUE(report.observability->full_rank());
    EXPECT_EQ(report.observability->structural_rank, 2);
    EXPECT_FALSE(report.identifiability->full_rank());
    EXPECT_EQ(report.identifiability->zero_sensitivity_local_indices, std::vector<int>({1}));
    EXPECT_TRUE(report.has_deficiency());
}

TEST(StructuralDiagnosticsTests, DiagnosticsRequireAtLeastOneRequestedAnalysis) {
    auto x = sym("x");
    Function fn("invalid_diag", {x}, {x});

    StructuralDiagnosticsOptions opts;
    EXPECT_THROW(analyze_structural_diagnostics(fn, opts), InvalidArgument);
}
