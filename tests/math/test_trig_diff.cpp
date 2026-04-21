#include <gtest/gtest.h>
#include <metis/math/Arithmetic.hpp>
#include <metis/math/Trig.hpp>
#include <metis/utils/GTestDiffTest.hpp>

// ============================================================================
// Basic Trigonometric Functions
// ============================================================================

TEST(TrigDiffTests, Sin) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::sin(x); },
                                            {{0.0}, {0.5}, {1.0}, {2.0}, {-1.0}});
}

TEST(TrigDiffTests, Cos) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::cos(x); },
                                            {{0.0}, {0.5}, {1.0}, {2.0}, {-1.0}});
}

TEST(TrigDiffTests, Tan) {
    // Avoid pi/2 where tan is undefined
    metis::diff_test::expect_differentiable([](auto x) { return metis::tan(x); },
                                            {{0.0}, {0.3}, {0.7}, {-0.5}});
}

// ============================================================================
// Inverse Trigonometric Functions
// ============================================================================

TEST(TrigDiffTests, Asin) {
    // Domain: (-1, 1), avoid endpoints where derivative is infinite
    metis::diff_test::expect_differentiable([](auto x) { return metis::asin(x); },
                                            {{-0.5}, {0.0}, {0.3}, {0.7}});
}

TEST(TrigDiffTests, Acos) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::acos(x); },
                                            {{-0.5}, {0.0}, {0.3}, {0.7}});
}

TEST(TrigDiffTests, Atan) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::atan(x); },
                                            {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

TEST(TrigDiffTests, Atan2) {
    // Avoid origin and negative x-axis where atan2 is non-smooth
    metis::diff_test::expect_differentiable([](auto y, auto x) { return metis::atan2(y, x); },
                                            {{1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}, {-1.0, 2.0}});
}

// ============================================================================
// Inverse Hyperbolic Functions
// ============================================================================

TEST(TrigDiffTests, Asinh) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::asinh(x); },
                                            {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

TEST(TrigDiffTests, Acosh) {
    // Domain: [1, inf), avoid x=1 where derivative is infinite
    metis::diff_test::expect_differentiable([](auto x) { return metis::acosh(x); },
                                            {{1.1}, {1.5}, {2.0}, {3.0}});
}

TEST(TrigDiffTests, Atanh) {
    // Domain: (-1, 1), avoid endpoints
    metis::diff_test::expect_differentiable([](auto x) { return metis::atanh(x); },
                                            {{-0.5}, {0.0}, {0.3}, {0.7}});
}

// ============================================================================
// Compositions of Trig Functions
// ============================================================================

TEST(TrigDiffTests, SinCosComposition) {
    // f(x) = sin(x) * cos(x) = 0.5 * sin(2x)
    metis::diff_test::expect_differentiable([](auto x) { return metis::sin(x) * metis::cos(x); },
                                            {{0.0}, {0.5}, {1.0}, {2.0}});
}

TEST(TrigDiffTests, TanIdentity) {
    // f(x) = sin(x) / cos(x) should equal tan(x)
    metis::diff_test::expect_differentiable([](auto x) { return metis::sin(x) / metis::cos(x); },
                                            {{0.3}, {0.7}, {-0.5}});
}

TEST(TrigDiffTests, AtanSqrt) {
    // f(x) = atan(sqrt(x))
    metis::diff_test::expect_differentiable([](auto x) { return metis::atan(metis::sqrt(x)); },
                                            {{0.25}, {1.0}, {4.0}});
}
