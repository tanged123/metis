#include <gtest/gtest.h>
#include <janus/math/SurrogateModel.hpp>
#include <janus/utils/GTestDiffTest.hpp>

// ============================================================================
// SurrogateModel.hpp smooth dual-mode functions
// ============================================================================

TEST(SurrogateDiffTests, Softplus) {
    // softplus(x, beta) = log(1 + exp(beta * x)) / beta
    janus::diff_test::expect_differentiable([](auto x) { return janus::softplus(x); },
                                            {{-2.0}, {0.0}, {1.0}, {5.0}});
}

TEST(SurrogateDiffTests, SoftplusBeta) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::softplus(x, 5.0); },
                                            {{-1.0}, {0.0}, {0.5}, {2.0}});
}

TEST(SurrogateDiffTests, SmoothAbs) {
    // smooth_abs(x, hardness) — smooth approximation of |x|
    janus::diff_test::expect_differentiable([](auto x) { return janus::smooth_abs(x, 10.0); },
                                            {{-2.0}, {-0.5}, {0.5}, {2.0}});
}

TEST(SurrogateDiffTests, SmoothClamp) {
    // smooth_clamp(x, low, high, hardness) — smooth clamp
    janus::diff_test::expect_differentiable(
        [](auto x) { return janus::smooth_clamp(x, -1.0, 1.0, 10.0); },
        {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

TEST(SurrogateDiffTests, SigmoidTanh) {
    // sigmoid(x) — default Tanh mode
    janus::diff_test::expect_differentiable([](auto x) { return janus::sigmoid(x); },
                                            {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, SigmoidLogistic) {
    janus::diff_test::expect_differentiable(
        [](auto x) { return janus::sigmoid(x, janus::SigmoidType::Logistic); },
        {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, SigmoidArctan) {
    janus::diff_test::expect_differentiable(
        [](auto x) { return janus::sigmoid(x, janus::SigmoidType::Arctan); },
        {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, SigmoidPolynomial) {
    janus::diff_test::expect_differentiable(
        [](auto x) { return janus::sigmoid(x, janus::SigmoidType::Polynomial); },
        {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, Swish) {
    // swish(x, beta) = x * sigmoid(beta * x)
    janus::diff_test::expect_differentiable([](auto x) { return janus::swish(x); },
                                            {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

TEST(SurrogateDiffTests, SwishBeta) {
    janus::diff_test::expect_differentiable([](auto x) { return janus::swish(x, 2.0); },
                                            {{-1.0}, {0.0}, {0.5}, {1.5}});
}

TEST(SurrogateDiffTests, Blend) {
    // blend(switch_val, val_high, val_low) — smooth step
    janus::diff_test::expect_differentiable([](auto x) { return janus::blend(x, 1.0, -1.0); },
                                            {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, KsMax) {
    // Kreisselmeier-Steinhauser smooth max
    janus::diff_test::expect_differentiable(
        [](auto a, auto b) {
            using S = std::decay_t<decltype(a)>;
            std::vector<S> vals = {a, b};
            return janus::ks_max(vals, 10.0);
        },
        {{1.0, 3.0}, {3.0, 1.0}, {-1.0, 2.0}});
}

TEST(SurrogateDiffTests, Softmax) {
    // softmax is a smooth-max approximation, returns a single scalar
    janus::diff_test::expect_differentiable(
        [](auto a, auto b) {
            using S = std::decay_t<decltype(a)>;
            std::vector<S> vals = {a, b};
            return janus::softmax(vals, 1.0);
        },
        {{1.0, 2.0}, {0.0, 0.0}, {-1.0, 1.0}});
}

TEST(SurrogateDiffTests, Softmax2Arg) {
    // Convenience 2-arg overload
    janus::diff_test::expect_differentiable(
        [](auto a, auto b) { return janus::softmax(a, b, 5.0); },
        {{1.0, 3.0}, {3.0, 1.0}, {2.0, 2.0}});
}
