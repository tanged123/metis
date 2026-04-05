#include <gtest/gtest.h>
#include <janus/math/Arithmetic.hpp>
#include <janus/math/Logic.hpp>
#include <janus/utils/GTestDiffTest.hpp>

// ============================================================================
// Smooth Logic Functions (differentiable)
// ============================================================================

TEST(LogicDiffTests, SigmoidBlend) {
    // sigmoid_blend(x, val_low, val_high, sharpness)
    janus::diff_test::expect_differentiable(
        [](auto x) { return janus::sigmoid_blend(x, 0.0, 1.0, 10.0); },
        {{-1.0}, {0.0}, {0.3}, {0.5}, {0.7}, {1.0}});
}

// ============================================================================
// Dual-Mode Only (non-smooth switching)
// ============================================================================

TEST(LogicDiffTests, WhereDualMode) {
    // where is discontinuous at the switching point
    // Cast values to same type as x so where() routes correctly for MX
    janus::diff_test::expect_dual_mode(
        [](auto x) {
            using S = std::decay_t<decltype(x)>;
            return janus::where(x > S(0.0), S(1.0), S(-1.0));
        },
        {{-2.0}, {-0.5}, {0.5}, {2.0}});
}

TEST(LogicDiffTests, MinDualMode) {
    janus::diff_test::expect_dual_mode([](auto x, auto y) { return janus::min(x, y); },
                                       {{1.0, 3.0}, {3.0, 1.0}, {-1.0, 2.0}});
}

TEST(LogicDiffTests, MaxDualMode) {
    janus::diff_test::expect_dual_mode([](auto x, auto y) { return janus::max(x, y); },
                                       {{1.0, 3.0}, {3.0, 1.0}, {-1.0, 2.0}});
}

TEST(LogicDiffTests, ClampDualMode) {
    janus::diff_test::expect_dual_mode([](auto x) { return janus::clamp(x, -1.0, 1.0); },
                                       {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

// ============================================================================
// Logic combinators and select (dual-mode only — non-smooth)
// ============================================================================

TEST(LogicDiffTests, SelectDualMode) {
    // janus::select — multi-way branching
    // Use auto for condition type since double comparisons return bool, MX returns MX
    janus::diff_test::expect_dual_mode(
        [](auto x) {
            using S = std::decay_t<decltype(x)>;
            auto c1 = x < S(-1.0);
            auto c2 = x < S(1.0);
            using CondType = decltype(c1);
            std::vector<CondType> conditions = {c1, c2};
            std::vector<S> values = {S(-1.0), S(0.0)};
            return janus::select(conditions, values, S(1.0));
        },
        {{-2.0}, {0.0}, {2.0}});
}

TEST(LogicDiffTests, LogicalAndScalarDualMode) {
    // logical_and takes JanusScalar (double/MX), not bool
    // Test with scalar truth values: nonzero = true, zero = false
    janus::diff_test::expect_dual_mode(
        [](auto x, auto y) {
            using S = std::decay_t<decltype(x)>;
            auto cond = janus::logical_and(x, y);
            return janus::where(cond, S(1.0), S(0.0));
        },
        {{1.0, 1.0}, {1.0, 0.0}, {0.0, 0.0}});
}

TEST(LogicDiffTests, LogicalOrScalarDualMode) {
    janus::diff_test::expect_dual_mode(
        [](auto x, auto y) {
            using S = std::decay_t<decltype(x)>;
            auto cond = janus::logical_or(x, y);
            return janus::where(cond, S(1.0), S(0.0));
        },
        {{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}});
}

TEST(LogicDiffTests, LogicalNotScalarDualMode) {
    janus::diff_test::expect_dual_mode(
        [](auto x) {
            using S = std::decay_t<decltype(x)>;
            auto cond = janus::logical_not(x);
            return janus::where(cond, S(1.0), S(0.0));
        },
        {{1.0}, {0.0}, {-1.0}});
}

// ============================================================================
// Smooth compositions using sigmoid_blend (differentiable)
// ============================================================================

TEST(LogicDiffTests, SmoothAbsViaSigmoid) {
    // Smooth approximation of abs using sigmoid_blend
    janus::diff_test::expect_differentiable(
        [](auto x) {
            // smooth |x| ≈ sigmoid_blend between -x and x, centered at 0
            return janus::sigmoid_blend(x, -x, x, 50.0);
        },
        {{-2.0}, {-0.5}, {0.5}, {2.0}});
}
