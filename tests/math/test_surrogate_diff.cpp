#include <gtest/gtest.h>
#include <metis/math/SurrogateModel.hpp>
#include <metis/utils/GTestDiffTest.hpp>

// ============================================================================
// SurrogateModel.hpp smooth dual-mode functions
// ============================================================================

TEST(SurrogateDiffTests, Softplus) {
    // softplus(x, beta) = log(1 + exp(beta * x)) / beta
    metis::diff_test::expect_differentiable([](auto x) { return metis::softplus(x); },
                                            {{-2.0}, {0.0}, {1.0}, {5.0}});
}

TEST(SurrogateDiffTests, SoftplusBeta) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::softplus(x, 5.0); },
                                            {{-1.0}, {0.0}, {0.5}, {2.0}});
}

TEST(SurrogateDiffTests, SmoothAbs) {
    // smooth_abs(x, hardness) — smooth approximation of |x|
    metis::diff_test::expect_differentiable([](auto x) { return metis::smooth_abs(x, 10.0); },
                                            {{-2.0}, {-0.5}, {0.5}, {2.0}});
}

TEST(SurrogateDiffTests, SmoothClamp) {
    // smooth_clamp(x, low, high, hardness) — smooth clamp
    metis::diff_test::expect_differentiable(
        [](auto x) { return metis::smooth_clamp(x, -1.0, 1.0, 10.0); },
        {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

TEST(SurrogateDiffTests, SigmoidTanh) {
    // sigmoid(x) — default Tanh mode
    metis::diff_test::expect_differentiable([](auto x) { return metis::sigmoid(x); },
                                            {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, SigmoidLogistic) {
    metis::diff_test::expect_differentiable(
        [](auto x) { return metis::sigmoid(x, metis::SigmoidType::Logistic); },
        {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, SigmoidArctan) {
    metis::diff_test::expect_differentiable(
        [](auto x) { return metis::sigmoid(x, metis::SigmoidType::Arctan); },
        {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, SigmoidPolynomial) {
    metis::diff_test::expect_differentiable(
        [](auto x) { return metis::sigmoid(x, metis::SigmoidType::Polynomial); },
        {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, Swish) {
    // swish(x, beta) = x * sigmoid(beta * x)
    metis::diff_test::expect_differentiable([](auto x) { return metis::swish(x); },
                                            {{-2.0}, {-0.5}, {0.0}, {0.5}, {2.0}});
}

TEST(SurrogateDiffTests, SwishBeta) {
    metis::diff_test::expect_differentiable([](auto x) { return metis::swish(x, 2.0); },
                                            {{-1.0}, {0.0}, {0.5}, {1.5}});
}

TEST(SurrogateDiffTests, Blend) {
    // blend(switch_val, val_high, val_low) — smooth step
    metis::diff_test::expect_differentiable([](auto x) { return metis::blend(x, 1.0, -1.0); },
                                            {{-3.0}, {-1.0}, {0.0}, {1.0}, {3.0}});
}

TEST(SurrogateDiffTests, KsMax) {
    // Kreisselmeier-Steinhauser smooth max
    metis::diff_test::expect_differentiable(
        [](auto a, auto b) {
            using S = std::decay_t<decltype(a)>;
            std::vector<S> vals = {a, b};
            return metis::ks_max(vals, 10.0);
        },
        {{1.0, 3.0}, {3.0, 1.0}, {-1.0, 2.0}});
}

TEST(SurrogateDiffTests, Softmax) {
    // softmax is a smooth-max approximation, returns a single scalar
    metis::diff_test::expect_differentiable(
        [](auto a, auto b) {
            using S = std::decay_t<decltype(a)>;
            std::vector<S> vals = {a, b};
            return metis::softmax(vals, 1.0);
        },
        {{1.0, 2.0}, {0.0, 0.0}, {-1.0, 1.0}});
}

TEST(SurrogateDiffTests, Softmax2Arg) {
    // Convenience 2-arg overload
    metis::diff_test::expect_differentiable(
        [](auto a, auto b) { return metis::softmax(a, b, 5.0); },
        {{1.0, 3.0}, {3.0, 1.0}, {2.0, 2.0}});
}
