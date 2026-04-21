#include <gtest/gtest.h>
#include <metis/math/Arithmetic.hpp>
#include <metis/math/Logic.hpp>
#include <metis/utils/GTestDiffTest.hpp>

// ============================================================================
// Smooth Logic Functions (differentiable)
// ============================================================================

TEST(LogicDiffTests, SigmoidBlend) {
    // sigmoid_blend(x, val_low, val_high, sharpness)
    metis::diff_test::expect_differentiable(
        [](auto x) { return metis::sigmoid_blend(x, 0.0, 1.0, 10.0); },
        {{-1.0}, {0.0}, {0.3}, {0.5}, {0.7}, {1.0}});
}

// ============================================================================
// Dual-Mode Only (non-smooth switching)
// ============================================================================

TEST(LogicDiffTests, WhereDualMode) {
    // where is discontinuous at the switching point
    // Cast values to same type as x so where() routes correctly for MX
    metis::diff_test::expect_dual_mode(
        [](auto x) {
            using S = std::decay_t<decltype(x)>;
            return metis::where(x > S(0.0), S(1.0), S(-1.0));
        },
        {{-2.0}, {-0.5}, {0.5}, {2.0}});
}

TEST(LogicDiffTests, MinDualMode) {
    metis::diff_test::expect_dual_mode([](auto x, auto y) { return metis::min(x, y); },
                                       {{1.0, 3.0}, {3.0, 1.0}, {-1.0, 2.0}});
}

TEST(LogicDiffTests, MaxDualMode) {
    metis::diff_test::expect_dual_mode([](auto x, auto y) { return metis::max(x, y); },
                                       {{1.0, 3.0}, {3.0, 1.0}, {-1.0, 2.0}});
}

TEST(LogicDiffTests, ClampDualMode) {
    metis::diff_test::expect_dual_mode([](auto x) { return metis::clamp(x, -1.0, 1.0); },
                                       {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

// ============================================================================
// Logic combinators and select (dual-mode only — non-smooth)
// ============================================================================

TEST(LogicDiffTests, SelectDualMode) {
    // metis::select — multi-way branching
    // Use auto for condition type since double comparisons return bool, MX returns MX
    metis::diff_test::expect_dual_mode(
        [](auto x) {
            using S = std::decay_t<decltype(x)>;
            auto c1 = x < S(-1.0);
            auto c2 = x < S(1.0);
            using CondType = decltype(c1);
            std::vector<CondType> conditions = {c1, c2};
            std::vector<S> values = {S(-1.0), S(0.0)};
            return metis::select(conditions, values, S(1.0));
        },
        {{-2.0}, {0.0}, {2.0}});
}

TEST(LogicDiffTests, LogicalAndScalarDualMode) {
    // logical_and takes MetisScalar (double/MX), not bool
    // Test with scalar truth values: nonzero = true, zero = false
    metis::diff_test::expect_dual_mode(
        [](auto x, auto y) {
            using S = std::decay_t<decltype(x)>;
            auto cond = metis::logical_and(x, y);
            return metis::where(cond, S(1.0), S(0.0));
        },
        {{1.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}});
}

TEST(LogicDiffTests, LogicalOrScalarDualMode) {
    metis::diff_test::expect_dual_mode(
        [](auto x, auto y) {
            using S = std::decay_t<decltype(x)>;
            auto cond = metis::logical_or(x, y);
            return metis::where(cond, S(1.0), S(0.0));
        },
        {{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}});
}

TEST(LogicDiffTests, LogicalNotScalarDualMode) {
    metis::diff_test::expect_dual_mode(
        [](auto x) {
            using S = std::decay_t<decltype(x)>;
            auto cond = metis::logical_not(x);
            return metis::where(cond, S(1.0), S(0.0));
        },
        {{1.0}, {0.0}, {-1.0}});
}

// ============================================================================
// Smooth compositions using sigmoid_blend (differentiable)
// ============================================================================

TEST(LogicDiffTests, SmoothAbsViaSigmoid) {
    // Smooth approximation of abs using sigmoid_blend
    metis::diff_test::expect_differentiable(
        [](auto x) {
            // smooth |x| ≈ sigmoid_blend between -x and x, centered at 0
            return metis::sigmoid_blend(x, -x, x, 50.0);
        },
        {{-2.0}, {-0.5}, {0.5}, {2.0}});
}
