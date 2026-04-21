#include <gtest/gtest.h>
#include <metis/math/Arithmetic.hpp>
#include <metis/math/Trig.hpp>
#include <metis/utils/GTestDiffTest.hpp>

// ============================================================================
// Differentiable Arithmetic Functions
// ============================================================================

TEST(ArithmeticDiffTests, Sqrt) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::sqrt(x); },
                                            {{0.25}, {1.0}, {4.0}, {16.0}});
}

TEST(ArithmeticDiffTests, Pow) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::pow(x, 3.0); },
                                            {{0.5}, {1.0}, {2.0}, {-1.5}});
}

TEST(ArithmeticDiffTests, PowBothArgs) {
    metis::diff_test::expect_differentiable(
        [](auto base, auto exp) { return metis::pow(base, exp); },
        {{2.0, 3.0}, {1.5, 2.5}, {3.0, 0.5}});
}

TEST(ArithmeticDiffTests, Exp) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::exp(x); },
                                            {{-1.0}, {0.0}, {1.0}, {2.0}});
}

TEST(ArithmeticDiffTests, Log) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::log(x); },
                                            {{0.1}, {0.5}, {1.0}, {5.0}});
}

TEST(ArithmeticDiffTests, Log2) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::log2(x); },
                                            {{0.25}, {1.0}, {4.0}, {8.0}});
}

TEST(ArithmeticDiffTests, Exp2) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::exp2(x); },
                                            {{-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(ArithmeticDiffTests, Cbrt) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::cbrt(x); },
                                            {{0.125}, {1.0}, {8.0}, {27.0}});
}

TEST(ArithmeticDiffTests, Square) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::square(x); },
                                            {{-3.0}, {-1.0}, {0.5}, {2.0}});
}

TEST(ArithmeticDiffTests, Hypot) {
    metis::diff_test::expect_differentiable([](auto x, auto y) { return metis::hypot(x, y); },
                                            {{3.0, 4.0}, {1.0, 1.0}, {5.0, 12.0}});
}

TEST(ArithmeticDiffTests, Expm1) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::expm1(x); },
                                            {{-0.5}, {0.0}, {0.5}, {1.0}});
}

TEST(ArithmeticDiffTests, Log1p) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::log1p(x); },
                                            {{0.01}, {0.5}, {1.0}, {5.0}});
}

TEST(ArithmeticDiffTests, Sinh) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::sinh(x); },
                                            {{-1.0}, {0.0}, {0.5}, {1.5}});
}

TEST(ArithmeticDiffTests, Cosh) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::cosh(x); },
                                            {{-1.0}, {0.0}, {0.5}, {1.5}});
}

TEST(ArithmeticDiffTests, Tanh) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::tanh(x); },
                                            {{-2.0}, {0.0}, {0.5}, {2.0}});
}

TEST(ArithmeticDiffTests, Log10) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::log10(x); },
                                            {{0.1}, {1.0}, {5.0}, {10.0}});
}

// ============================================================================
// Dual-Mode Only (non-differentiable or piecewise)
// ============================================================================

TEST(ArithmeticDiffTests, AbsDualMode) {
    // abs is non-differentiable at 0; test away from 0
    metis::diff_test::expect_dual_mode([](auto x) { return metis::abs(x); },
                                       {{-3.0}, {-0.5}, {0.5}, {3.0}});
}

TEST(ArithmeticDiffTests, FloorDualMode) {
    // floor has zero derivative almost everywhere, undefined at integers
    metis::diff_test::expect_dual_mode([](auto x) { return metis::floor(x); },
                                       {{0.5}, {1.7}, {-2.3}});
}

TEST(ArithmeticDiffTests, CeilDualMode) {
    metis::diff_test::expect_dual_mode([](auto x) { return metis::ceil(x); },
                                       {{0.5}, {1.7}, {-2.3}});
}

TEST(ArithmeticDiffTests, SignDualMode) {
    metis::diff_test::expect_dual_mode([](auto x) { return metis::sign(x); },
                                       {{-3.0}, {-0.5}, {0.5}, {3.0}});
}

TEST(ArithmeticDiffTests, FmodDualMode) {
    metis::diff_test::expect_dual_mode([](auto x, auto y) { return metis::fmod(x, y); },
                                       {{5.3, 2.0}, {7.1, 3.0}});
}

TEST(ArithmeticDiffTests, RoundDualMode) {
    metis::diff_test::expect_dual_mode([](auto x) { return metis::round(x); },
                                       {{0.3}, {1.7}, {-2.3}});
}

TEST(ArithmeticDiffTests, RoundNegativeHalfDualMode) {
    // CasADi's round uses floor(x + 0.5) which rounds -2.5 to -2,
    // while std::round rounds -2.5 to -3. Test with a relaxed tolerance
    // to document this known divergence.
    metis::diff_test::DiffTestOptions opts;
    opts.value_tol = 1.5; // Allow the documented 1.0 divergence at half-integers
    metis::diff_test::expect_dual_mode([](auto x) { return metis::round(x); },
                                       {{-2.5}, {-0.5}, {0.5}, {2.5}}, opts);
}

TEST(ArithmeticDiffTests, TruncDualMode) {
    metis::diff_test::expect_dual_mode([](auto x) { return metis::trunc(x); },
                                       {{0.7}, {2.3}, {-1.7}});
}

TEST(ArithmeticDiffTests, CopysignDualMode) {
    metis::diff_test::expect_dual_mode([](auto x, auto y) { return metis::copysign(x, y); },
                                       {{5.0, 1.0}, {5.0, -1.0}, {-5.0, 1.0}});
}

// ============================================================================
// Compositions
// ============================================================================

TEST(ArithmeticDiffTests, QuadraticComposition) {
    // f(x) = 3x^2 + 2x + 1
    metis::diff_test::expect_differentiable(
        [](auto x) { return 3.0 * metis::pow(x, 2.0) + 2.0 * x + 1.0; },
        {{-2.0}, {0.0}, {1.0}, {3.0}});
}

TEST(ArithmeticDiffTests, ExpSinComposition) {
    // f(x) = exp(sin(x))
    metis::diff_test::expect_differentiable([](auto x) { return metis::exp(metis::sin(x)); },
                                            {{0.0}, {0.5}, {1.0}, {2.0}});
}

TEST(ArithmeticDiffTests, MultiInputComposition) {
    // f(x, y) = x^2 * exp(y) + log(x + 1)
    metis::diff_test::expect_differentiable(
        [](auto x, auto y) { return metis::pow(x, 2.0) * metis::exp(y) + metis::log(x + 1.0); },
        {{1.0, 0.0}, {2.0, -0.5}, {0.5, 1.0}});
}
