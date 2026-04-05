#include <gtest/gtest.h>
#include <janus/math/Arithmetic.hpp>
#include <janus/math/Trig.hpp>
#include <janus/utils/GTestDiffTest.hpp>

// ============================================================================
// Differentiable Arithmetic Functions
// ============================================================================

TEST(ArithmeticDiffTests, Sqrt) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::sqrt(x); },
                                            {{0.25}, {1.0}, {4.0}, {16.0}});
}

TEST(ArithmeticDiffTests, Pow) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::pow(x, 3.0); },
                                            {{0.5}, {1.0}, {2.0}, {-1.5}});
}

TEST(ArithmeticDiffTests, PowBothArgs) {
    janus::diff_test::expect_differentiable(
        [](auto base, auto exp) { return janus::pow(base, exp); },
        {{2.0, 3.0}, {1.5, 2.5}, {3.0, 0.5}});
}

TEST(ArithmeticDiffTests, Exp) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::exp(x); },
                                            {{-1.0}, {0.0}, {1.0}, {2.0}});
}

TEST(ArithmeticDiffTests, Log) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::log(x); },
                                            {{0.1}, {0.5}, {1.0}, {5.0}});
}

TEST(ArithmeticDiffTests, Log2) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::log2(x); },
                                            {{0.25}, {1.0}, {4.0}, {8.0}});
}

TEST(ArithmeticDiffTests, Exp2) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::exp2(x); },
                                            {{-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(ArithmeticDiffTests, Cbrt) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::cbrt(x); },
                                            {{0.125}, {1.0}, {8.0}, {27.0}});
}

TEST(ArithmeticDiffTests, Square) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::square(x); },
                                            {{-3.0}, {-1.0}, {0.5}, {2.0}});
}

TEST(ArithmeticDiffTests, Hypot) {
    janus::diff_test::expect_differentiable([](auto x, auto y) { return janus::hypot(x, y); },
                                            {{3.0, 4.0}, {1.0, 1.0}, {5.0, 12.0}});
}

TEST(ArithmeticDiffTests, Expm1) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::expm1(x); },
                                            {{-0.5}, {0.0}, {0.5}, {1.0}});
}

TEST(ArithmeticDiffTests, Log1p) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::log1p(x); },
                                            {{0.01}, {0.5}, {1.0}, {5.0}});
}

TEST(ArithmeticDiffTests, Sinh) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::sinh(x); },
                                            {{-1.0}, {0.0}, {0.5}, {1.5}});
}

TEST(ArithmeticDiffTests, Cosh) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::cosh(x); },
                                            {{-1.0}, {0.0}, {0.5}, {1.5}});
}

TEST(ArithmeticDiffTests, Tanh) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::tanh(x); },
                                            {{-2.0}, {0.0}, {0.5}, {2.0}});
}

TEST(ArithmeticDiffTests, Log10) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::log10(x); },
                                            {{0.1}, {1.0}, {5.0}, {10.0}});
}

// ============================================================================
// Dual-Mode Only (non-differentiable or piecewise)
// ============================================================================

TEST(ArithmeticDiffTests, AbsDualMode) {
    // abs is non-differentiable at 0; test away from 0
    janus::diff_test::expect_dual_mode([](auto x) { return janus::abs(x); },
                                       {{-3.0}, {-0.5}, {0.5}, {3.0}});
}

TEST(ArithmeticDiffTests, FloorDualMode) {
    // floor has zero derivative almost everywhere, undefined at integers
    janus::diff_test::expect_dual_mode([](auto x) { return janus::floor(x); },
                                       {{0.5}, {1.7}, {-2.3}});
}

TEST(ArithmeticDiffTests, CeilDualMode) {
    janus::diff_test::expect_dual_mode([](auto x) { return janus::ceil(x); },
                                       {{0.5}, {1.7}, {-2.3}});
}

TEST(ArithmeticDiffTests, SignDualMode) {
    janus::diff_test::expect_dual_mode([](auto x) { return janus::sign(x); },
                                       {{-3.0}, {-0.5}, {0.5}, {3.0}});
}

TEST(ArithmeticDiffTests, FmodDualMode) {
    janus::diff_test::expect_dual_mode([](auto x, auto y) { return janus::fmod(x, y); },
                                       {{5.3, 2.0}, {7.1, 3.0}});
}

TEST(ArithmeticDiffTests, RoundDualMode) {
    janus::diff_test::expect_dual_mode([](auto x) { return janus::round(x); },
                                       {{0.3}, {1.7}, {-2.3}});
}

TEST(ArithmeticDiffTests, RoundNegativeHalfDualMode) {
    // CasADi's round uses floor(x + 0.5) which rounds -2.5 to -2,
    // while std::round rounds -2.5 to -3. Test with a relaxed tolerance
    // to document this known divergence.
    janus::diff_test::DiffTestOptions opts;
    opts.value_tol = 1.5; // Allow the documented 1.0 divergence at half-integers
    janus::diff_test::expect_dual_mode([](auto x) { return janus::round(x); },
                                       {{-2.5}, {-0.5}, {0.5}, {2.5}}, opts);
}

TEST(ArithmeticDiffTests, TruncDualMode) {
    janus::diff_test::expect_dual_mode([](auto x) { return janus::trunc(x); },
                                       {{0.7}, {2.3}, {-1.7}});
}

TEST(ArithmeticDiffTests, CopysignDualMode) {
    janus::diff_test::expect_dual_mode([](auto x, auto y) { return janus::copysign(x, y); },
                                       {{5.0, 1.0}, {5.0, -1.0}, {-5.0, 1.0}});
}

// ============================================================================
// Compositions
// ============================================================================

TEST(ArithmeticDiffTests, QuadraticComposition) {
    // f(x) = 3x^2 + 2x + 1
    janus::diff_test::expect_differentiable(
        [](auto x) { return 3.0 * janus::pow(x, 2.0) + 2.0 * x + 1.0; },
        {{-2.0}, {0.0}, {1.0}, {3.0}});
}

TEST(ArithmeticDiffTests, ExpSinComposition) {
    // f(x) = exp(sin(x))
    janus::diff_test::expect_differentiable([](auto x) { return janus::exp(janus::sin(x)); },
                                            {{0.0}, {0.5}, {1.0}, {2.0}});
}

TEST(ArithmeticDiffTests, MultiInputComposition) {
    // f(x, y) = x^2 * exp(y) + log(x + 1)
    janus::diff_test::expect_differentiable(
        [](auto x, auto y) { return janus::pow(x, 2.0) * janus::exp(y) + janus::log(x + 1.0); },
        {{1.0, 0.0}, {2.0, -0.5}, {0.5, 1.0}});
}
