#pragma once
/**
 * @file GTestDiffTest.hpp
 * @brief GoogleTest wrappers for the differentiability test harness
 *
 * Thin layer on top of DiffTestHarness.hpp that translates results into
 * GTest assertions with informative failure messages.
 *
 * @see DiffTestHarness.hpp
 */

#include <gtest/gtest.h>
#include <janus/utils/DiffTestHarness.hpp>

namespace janus::diff_test {

/**
 * @brief Assert that a dual-mode function compiles symbolically and that
 *        symbolic output matches numeric output at all test points.
 *
 * @code
 * TEST(MyTests, FloorDualMode) {
 *     janus::diff_test::expect_dual_mode(
 *         [](auto x) { return janus::floor(x); },
 *         {{0.5}, {1.7}, {-2.3}}
 *     );
 * }
 * @endcode
 */
template <typename Func>
void expect_dual_mode(Func &&f, const std::vector<std::vector<double>> &test_points,
                      const DiffTestOptions &opts = {}) {
    auto result = verify_dual_mode(std::forward<Func>(f), test_points, opts);
    ASSERT_TRUE(result.symbolic_compiles) << result.failure_detail;
    EXPECT_TRUE(result.values_match) << result.failure_detail;
}

/**
 * @brief Assert that a dual-mode function is differentiable:
 *        symbolic compilation + value match + AD Jacobian matches FD Jacobian.
 *
 * @code
 * TEST(MyTests, SinDifferentiable) {
 *     janus::diff_test::expect_differentiable(
 *         [](auto x) { return janus::sin(x); },
 *         {{0.5}, {1.0}, {2.0}}
 *     );
 * }
 * @endcode
 */
template <typename Func>
void expect_differentiable(Func &&f, const std::vector<std::vector<double>> &test_points,
                           const DiffTestOptions &opts = {}) {
    auto result = verify_differentiable(std::forward<Func>(f), test_points, opts);
    ASSERT_TRUE(result.symbolic_compiles) << result.failure_detail;
    ASSERT_TRUE(result.values_match) << result.failure_detail;
    EXPECT_TRUE(result.jacobian_matches) << result.failure_detail;
}

} // namespace janus::diff_test
