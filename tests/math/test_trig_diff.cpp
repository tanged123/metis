#include <gtest/gtest.h>
#include <janus/math/Arithmetic.hpp>
#include <janus/math/Trig.hpp>
#include <janus/utils/GTestDiffTest.hpp>

// ============================================================================
// Basic Trigonometric Functions
// ============================================================================

TEST(TrigDiffTests, Sin) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::sin(x); },
                                            {{0.0}, {0.5}, {1.0}, {2.0}, {-1.0}});
}

TEST(TrigDiffTests, Cos) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::cos(x); },
                                            {{0.0}, {0.5}, {1.0}, {2.0}, {-1.0}});
}

TEST(TrigDiffTests, Tan) {
    // Avoid pi/2 where tan is undefined
    janus::diff_test::expect_differentiable([](auto x) { return janus::tan(x); },
                                            {{0.0}, {0.3}, {0.7}, {-0.5}});
}

// ============================================================================
// Inverse Trigonometric Functions
// ============================================================================

TEST(TrigDiffTests, Asin) {
    // Domain: (-1, 1), avoid endpoints where derivative is infinite
    janus::diff_test::expect_differentiable([](auto x) { return janus::asin(x); },
                                            {{-0.5}, {0.0}, {0.3}, {0.7}});
}

TEST(TrigDiffTests, Acos) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::acos(x); },
                                            {{-0.5}, {0.0}, {0.3}, {0.7}});
}

TEST(TrigDiffTests, Atan) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::atan(x); },
                                            {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

TEST(TrigDiffTests, Atan2) {
    // Avoid origin and negative x-axis where atan2 is non-smooth
    janus::diff_test::expect_differentiable([](auto y, auto x) { return janus::atan2(y, x); },
                                            {{1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}, {-1.0, 2.0}});
}

// ============================================================================
// Inverse Hyperbolic Functions
// ============================================================================

TEST(TrigDiffTests, Asinh) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::asinh(x); },
                                            {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

TEST(TrigDiffTests, Acosh) {
    // Domain: [1, inf), avoid x=1 where derivative is infinite
    janus::diff_test::expect_differentiable([](auto x) { return janus::acosh(x); },
                                            {{1.1}, {1.5}, {2.0}, {3.0}});
}

TEST(TrigDiffTests, Atanh) {
    // Domain: (-1, 1), avoid endpoints
    janus::diff_test::expect_differentiable([](auto x) { return janus::atanh(x); },
                                            {{-0.5}, {0.0}, {0.3}, {0.7}});
}

// ============================================================================
// Compositions of Trig Functions
// ============================================================================

TEST(TrigDiffTests, SinCosComposition) {
    // f(x) = sin(x) * cos(x) = 0.5 * sin(2x)
    janus::diff_test::expect_differentiable([](auto x) { return janus::sin(x) * janus::cos(x); },
                                            {{0.0}, {0.5}, {1.0}, {2.0}});
}

TEST(TrigDiffTests, TanIdentity) {
    // f(x) = sin(x) / cos(x) should equal tan(x)
    janus::diff_test::expect_differentiable([](auto x) { return janus::sin(x) / janus::cos(x); },
                                            {{0.3}, {0.7}, {-0.5}});
}

TEST(TrigDiffTests, AtanSqrt) {
    // f(x) = atan(sqrt(x))
    janus::diff_test::expect_differentiable([](auto x) { return janus::atan(janus::sqrt(x)); },
                                            {{0.25}, {1.0}, {4.0}});
}
