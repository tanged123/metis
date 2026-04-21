#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/core/StructuralTransforms.hpp>
#include <metis/math/Trig.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace metis;

namespace {

std::vector<int> sorted_copy(std::vector<int> values) {
    std::sort(values.begin(), values.end());
    return values;
}

bool block_matches(const StructuralBlock &block, const std::vector<int> &residuals,
                   const std::vector<int> &variables) {
    return sorted_copy(block.residual_indices) == sorted_copy(residuals) &&
           sorted_copy(block.variable_indices) == sorted_copy(variables);
}

} // namespace

TEST(StructuralTransformsTests, AliasEliminationBuildsReducedResidualAndReconstructionMap) {
    auto x = sym("x", 3, 1);
    auto p = sym("p");
    SymbolicScalar x0 = x(0);
    SymbolicScalar x1 = x(1);
    SymbolicScalar x2 = x(2);
    auto residual = SymbolicScalar::vertcat({
        x0 - x1,
        x2 - p,
        metis::sin(x0) + x2 - 2.0,
    });

    Function fn("alias_system", {x, p}, {residual});
    auto alias = alias_eliminate(fn);

    EXPECT_EQ(alias.kept_variable_indices, std::vector<int>({0}));
    EXPECT_EQ(alias.eliminated_variable_indices, std::vector<int>({1, 2}));
    EXPECT_EQ(alias.kept_residual_indices, std::vector<int>({2}));
    EXPECT_EQ(alias.eliminated_residual_indices, std::vector<int>({0, 1}));
    ASSERT_EQ(alias.substitutions.size(), 2u);
    EXPECT_EQ(alias.substitutions[0].eliminated_variable_index, 1);
    EXPECT_EQ(alias.substitutions[1].eliminated_variable_index, 2);

    NumericMatrix x_reduced(1, 1);
    x_reduced(0, 0) = 1.25;

    const NumericMatrix reconstructed = alias.reconstruct_full_input.eval(x_reduced, 0.5);
    ASSERT_EQ(reconstructed.rows(), 3);
    ASSERT_EQ(reconstructed.cols(), 1);
    EXPECT_NEAR(reconstructed(0, 0), 1.25, 1e-12);
    EXPECT_NEAR(reconstructed(1, 0), 1.25, 1e-12);
    EXPECT_NEAR(reconstructed(2, 0), 0.5, 1e-12);

    const NumericMatrix reduced_residual = alias.reduced_function.eval(x_reduced, 0.5);
    ASSERT_EQ(reduced_residual.rows(), 1);
    ASSERT_EQ(reduced_residual.cols(), 1);
    EXPECT_NEAR(reduced_residual(0, 0), std::sin(1.25) - 1.5, 1e-12);
}

TEST(StructuralTransformsTests, BlockTriangularizationFindsIndependentAndCoupledBlocks) {
    auto x = sym("x", 4, 1);
    auto residual = SymbolicScalar::vertcat({
        x(0) - 1.0,
        x(1) + x(2),
        x(1) - x(2),
        x(3) - 3.0,
    });

    Function fn("blt_blocks", {x}, {residual});
    auto blt = block_triangularize(fn);

    ASSERT_EQ(blt.blocks.size(), 3u);

    bool saw_x0 = false;
    bool saw_coupled = false;
    bool saw_x3 = false;
    for (const auto &block : blt.blocks) {
        if (block_matches(block, {0}, {0})) {
            saw_x0 = true;
            EXPECT_FALSE(block.is_coupled());
        } else if (block_matches(block, {1, 2}, {1, 2})) {
            saw_coupled = true;
            EXPECT_TRUE(block.is_coupled());
            EXPECT_EQ(block.tear_variable_indices.size(), 1u);
        } else if (block_matches(block, {3}, {3})) {
            saw_x3 = true;
            EXPECT_FALSE(block.is_coupled());
        }
    }

    EXPECT_TRUE(saw_x0);
    EXPECT_TRUE(saw_coupled);
    EXPECT_TRUE(saw_x3);
}

TEST(StructuralTransformsTests, TearingRecommendsSingleIterationVariableForSimpleCycle) {
    auto x = sym("x", 3, 1);
    auto residual = SymbolicScalar::vertcat({
        x(0) + x(1),
        x(1) + x(2),
        x(2) + x(0),
    });

    Function fn("tearing_cycle", {x}, {residual});
    auto blt = block_triangularize(fn);

    ASSERT_EQ(blt.blocks.size(), 1u);
    const auto &block = blt.blocks.front();
    EXPECT_TRUE(block.is_coupled());
    ASSERT_EQ(block.tear_variable_indices.size(), 1u);
    EXPECT_GE(block.tear_variable_indices.front(), 0);
    EXPECT_LT(block.tear_variable_indices.front(), 3);
}

TEST(StructuralTransformsTests, StructuralAnalyzeRunsAliasEliminationBeforeBLT) {
    auto x = sym("x", 3, 1);
    auto p = sym("p");
    SymbolicScalar x0 = x(0);
    SymbolicScalar x1 = x(1);
    SymbolicScalar x2 = x(2);
    auto residual = SymbolicScalar::vertcat({
        x0 - x1,
        x2 - p,
        metis::sin(x0) + x2 - 2.0,
    });

    Function fn("structural_pipeline", {x, p}, {residual});
    auto analysis = structural_analyze(fn);

    EXPECT_EQ(analysis.alias_elimination.kept_variable_indices, std::vector<int>({0}));
    EXPECT_EQ(analysis.alias_elimination.kept_residual_indices, std::vector<int>({2}));
    ASSERT_EQ(analysis.blt.blocks.size(), 1u);
    EXPECT_TRUE(block_matches(analysis.blt.blocks.front(), {0}, {0}));

    NumericMatrix x_reduced(1, 1);
    x_reduced(0, 0) = 1.4;
    const NumericMatrix reduced_residual =
        analysis.alias_elimination.reduced_function.eval(x_reduced, 0.45);
    EXPECT_NEAR(reduced_residual(0, 0), std::sin(1.4) - 1.55, 1e-12);
}
