#include "../utils/TestUtils.hpp" // specific path to TestUtils
#include "metis/core/Function.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/math/AutoDiff.hpp"
#include "metis/math/SurrogateModel.hpp"
#include <gtest/gtest.h>

namespace metis {
namespace test {
namespace {

template <typename Fn> double forward_difference(Fn &&fn, double x, double h) {
    return (fn(x + h) - fn(x)) / h;
}

template <typename Fn> double backward_difference(Fn &&fn, double x, double h) {
    return (fn(x) - fn(x - h)) / h;
}

template <typename Fn> double forward_second_difference(Fn &&fn, double x, double h) {
    return (fn(x + 2.0 * h) - 2.0 * fn(x + h) + fn(x)) / (h * h);
}

template <typename Fn> double backward_second_difference(Fn &&fn, double x, double h) {
    return (fn(x) - 2.0 * fn(x - h) + fn(x - 2.0 * h)) / (h * h);
}

double logistic(double x) { return 1.0 / (1.0 + std::exp(-x)); }

} // namespace

// ======================================================================
// Softmax Tests
// ======================================================================

TEST(SurrogateTests, SoftmaxNumeric) {
    std::vector<double> vals = {1.0, 2.0, 3.0};

    // With softness -> 0 (hard max), should approach max(vals) = 3.0
    // But implementation divides by softness, so we can't use 0.
    // Use small softness.
    double hardish = softmax(vals, 0.01);
    EXPECT_NEAR(hardish, 3.0, 1e-2);

    // With large softness, should be smoother/larger
    double soft = softmax(vals, 1.0);
    // Formula: 3 + 1 * log(exp(-2) + exp(-1) + exp(0))
    // exp(0)=1, exp(-1)~0.368, exp(-2)~0.135 -> sum ~ 1.503
    // log(1.503) ~ 0.407
    // result ~ 3.407
    EXPECT_GT(soft, 3.0);
    EXPECT_NEAR(soft, 3.0 + std::log(std::exp(-2) + std::exp(-1) + std::exp(0)), 1e-5);
}

TEST(SurrogateTests, SoftmaxSymbolic) {
    // Symbolic check
    casadi::MX x = casadi::MX::sym("x");
    casadi::MX y = casadi::MX::sym("y");
    std::vector<casadi::MX> args = {x, y};

    // Just structural check that it compiles and returns an expression
    auto res = softmax(args);

    // Evaluate
    metis::Function f({x, y}, {res});
    // f(1, 3) with default softness 1.0
    // max=3. sum= exp(-2)+exp(0) = 0.135+1 = 1.135. log(1.135)~0.126. res 3.126
    auto val = f.eval(1.0, 3.0);
    EXPECT_NEAR(static_cast<double>(val(0, 0)), 3.0 + std::log(std::exp(-2) + 1.0), 1e-5);
}

TEST(SurrogateTests, Softmax2Arg) {
    // Convenience overload
    double val = softmax(1.0, 5.0, 0.1);
    EXPECT_NEAR(val, 5.0, 0.1); // Close to max
}

// ======================================================================
// Softmin Tests
// ======================================================================

TEST(SurrogateTests, SoftminNumeric) {
    std::vector<double> vals = {1.0, 2.0, 3.0};

    // Should be close to min = 1.0
    double val = softmin(vals, 0.01);
    EXPECT_NEAR(val, 1.0, 1e-2);
}

TEST(SurrogateTests, SoftminSymbolic) {
    auto x = metis::sym("x");
    std::vector<casadi::MX> args = {x, casadi::MX(2.0)};
    auto expr = softmin(args, 0.1);

    // softmin(1.0, 2.0) ~ min(1,2) = 1
    double val = metis::Function({x}, {expr}).eval(1.0)(0, 0);
    EXPECT_NEAR(val, 1.0, 0.1);
}

// ======================================================================
// Softplus Tests
// ======================================================================

TEST(SurrogateTests, SoftplusNumeric) {
    // softplus(0) = log(1+1) = log(2) ~ 0.693 (beta=1)
    EXPECT_NEAR(softplus(0.0), std::log(2.0), 1e-5);

    // Large negative -> 0
    EXPECT_NEAR(softplus(-100.0), 0.0, 1e-5);

    // Large positive -> linear
    EXPECT_NEAR(softplus(1000.0), 1000.0, 1e-5);
}

TEST(SurrogateTests, SoftplusSymbolic) {
    auto x = metis::sym("x");
    auto expr = softplus(x);
    metis::Function f({x}, {expr});

    EXPECT_NEAR(f.eval(0.0)(0, 0), std::log(2.0), 1e-5);
    EXPECT_NEAR(f.eval(1000.0)(0, 0), 1000.0, 1e-5);
}

TEST(SurrogateTests, SoftplusNumericDerivativesRemainSmoothAcrossLegacyThreshold) {
    constexpr double beta = 3.0;
    constexpr double threshold = 2.0;
    constexpr double transition = threshold / beta;
    constexpr double h = 1e-4;

    auto fn = [&](double x) { return softplus(x, beta, threshold); };

    const double d1_left = backward_difference(fn, transition, h);
    const double d1_right = forward_difference(fn, transition, h);
    const double d2_left = backward_second_difference(fn, transition, h);
    const double d2_right = forward_second_difference(fn, transition, h);

    const double sigma = logistic(beta * transition);
    const double expected_second = beta * sigma * (1.0 - sigma);

    EXPECT_NEAR(d1_left, sigma, 5e-4);
    EXPECT_NEAR(d1_right, sigma, 5e-4);
    EXPECT_NEAR(d1_left, d1_right, 5e-4);

    EXPECT_NEAR(d2_left, expected_second, 5e-3);
    EXPECT_NEAR(d2_right, expected_second, 5e-3);
    EXPECT_NEAR(d2_left, d2_right, 5e-3);
}

TEST(SurrogateTests, SoftplusSymbolicDerivativesRemainSmoothAcrossLegacyThreshold) {
    constexpr double beta = 3.0;
    constexpr double threshold = 2.0;
    constexpr double transition = threshold / beta;
    constexpr double h = 1e-4;

    auto x = metis::sym("x");
    auto expr = softplus(x, beta, threshold);
    auto first = metis::jacobian(expr, x);
    auto second = metis::hessian(expr, x);

    metis::Function first_fn({x}, {first});
    metis::Function second_fn({x}, {second});

    const double d1_left = first_fn.eval(transition - h)(0, 0);
    const double d1_right = first_fn.eval(transition + h)(0, 0);
    const double d2_left = second_fn.eval(transition - h)(0, 0);
    const double d2_right = second_fn.eval(transition + h)(0, 0);

    const double sigma_left = logistic(beta * (transition - h));
    const double sigma_right = logistic(beta * (transition + h));
    const double expected_second_left = beta * sigma_left * (1.0 - sigma_left);
    const double expected_second_right = beta * sigma_right * (1.0 - sigma_right);

    EXPECT_NEAR(d1_left, sigma_left, 1e-9);
    EXPECT_NEAR(d1_right, sigma_right, 1e-9);
    EXPECT_NEAR(d1_left, d1_right, 1e-3);

    EXPECT_NEAR(d2_left, expected_second_left, 1e-9);
    EXPECT_NEAR(d2_right, expected_second_right, 1e-9);
    EXPECT_NEAR(d2_left, d2_right, 1e-3);
}

// ======================================================================
// Smooth Approximation Suite Tests
// ======================================================================

TEST(SurrogateTests, SmoothAbsNumericConvergesToAbsoluteValue) {
    constexpr double hardness = 80.0;

    EXPECT_NEAR(smooth_abs(-2.0, hardness), 2.0, 1e-6);
    EXPECT_NEAR(smooth_abs(-0.5, hardness), 0.5, 1e-6);
    EXPECT_NEAR(smooth_abs(1.25, hardness), 1.25, 1e-6);
    EXPECT_NEAR(smooth_abs(0.0, hardness), std::log(2.0) / hardness, 1e-10);
}

TEST(SurrogateTests, SmoothAbsSymbolicDerivativesExistAtOrigin) {
    constexpr double hardness = 20.0;

    auto x = metis::sym("x");
    auto expr = smooth_abs(x, hardness);
    auto first = metis::jacobian(expr, x);
    auto second = metis::hessian(expr, x);

    metis::Function f({x}, {expr});
    metis::Function first_fn({x}, {first});
    metis::Function second_fn({x}, {second});

    EXPECT_NEAR(f.eval(0.0)(0, 0), std::log(2.0) / hardness, 1e-10);
    EXPECT_NEAR(first_fn.eval(0.0)(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(second_fn.eval(0.0)(0, 0), hardness, 1e-10);
}

TEST(SurrogateTests, SmoothMaxMinNumericConvergeToHardOperators) {
    constexpr double hardness = 100.0;

    EXPECT_NEAR(smooth_max(1.0, 3.0, hardness), 3.0, 1e-10);
    EXPECT_NEAR(smooth_min(1.0, 3.0, hardness), 1.0, 1e-10);
    EXPECT_NEAR(smooth_max(-2.0, -5.0, hardness), -2.0, 1e-10);
    EXPECT_NEAR(smooth_min(-2.0, -5.0, hardness), -5.0, 1e-10);
}

TEST(SurrogateTests, SmoothMaxMinSymbolicGradientsExistAtTie) {
    constexpr double hardness = 15.0;

    auto a = metis::sym("a");
    auto b = metis::sym("b");

    auto smax = smooth_max(a, b, hardness);
    auto smin = smooth_min(a, b, hardness);

    auto grad_max = metis::jacobian(smax, a, b);
    auto grad_min = metis::jacobian(smin, a, b);

    metis::Function grad_max_fn({a, b}, {grad_max});
    metis::Function grad_min_fn({a, b}, {grad_min});

    auto max_grad = grad_max_fn.eval(0.0, 0.0);
    auto min_grad = grad_min_fn.eval(0.0, 0.0);

    EXPECT_NEAR(max_grad(0, 0), 0.5, 1e-10);
    EXPECT_NEAR(max_grad(0, 1), 0.5, 1e-10);
    EXPECT_NEAR(min_grad(0, 0), 0.5, 1e-10);
    EXPECT_NEAR(min_grad(0, 1), 0.5, 1e-10);
}

TEST(SurrogateTests, SmoothClampNumericConvergesToClamp) {
    constexpr double hardness = 100.0;

    EXPECT_NEAR(smooth_clamp(-2.0, 0.0, 1.0, hardness), 0.0, 1e-10);
    EXPECT_NEAR(smooth_clamp(0.4, 0.0, 1.0, hardness), 0.4, 1e-10);
    EXPECT_NEAR(smooth_clamp(2.0, 0.0, 1.0, hardness), 1.0, 1e-10);
}

TEST(SurrogateTests, SmoothClampSymbolicGradientExistsAtBounds) {
    constexpr double hardness = 25.0;

    auto x = metis::sym("x");
    auto expr = smooth_clamp(x, 0.0, 1.0, hardness);
    auto grad = metis::jacobian(expr, x);

    metis::Function grad_fn({x}, {grad});

    const double grad_at_low = grad_fn.eval(0.0)(0, 0);
    const double grad_at_high = grad_fn.eval(1.0)(0, 0);

    EXPECT_GT(grad_at_low, 0.0);
    EXPECT_LT(grad_at_low, 1.0);
    EXPECT_GT(grad_at_high, 0.0);
    EXPECT_LT(grad_at_high, 1.0);
}

TEST(SurrogateTests, KsMaxNumericConvergesToMaximum) {
    constexpr double rho = 120.0;
    const std::vector<double> values = {1.0, 2.0, 3.0};

    EXPECT_NEAR(ks_max(values, rho), 3.0, 1e-10);

    metis::NumericVector eigen_values(3);
    eigen_values << 1.0, 2.0, 3.0;
    EXPECT_NEAR(ks_max(eigen_values, rho), 3.0, 1e-10);
}

TEST(SurrogateTests, KsMaxSymbolicGradientExistsAndFormsConvexWeights) {
    constexpr double rho = 9.0;

    auto x = metis::sym("x");
    auto y = metis::sym("y");
    auto z = metis::sym("z");

    auto expr = ks_max(std::vector<metis::SymbolicScalar>{x, y, z}, rho);
    auto grad = metis::jacobian(expr, x, y, z);

    metis::Function grad_fn({x, y, z}, {grad});
    auto grad_val = grad_fn.eval(0.0, 0.0, 0.0);

    EXPECT_NEAR(grad_val(0, 0), 1.0 / 3.0, 1e-10);
    EXPECT_NEAR(grad_val(0, 1), 1.0 / 3.0, 1e-10);
    EXPECT_NEAR(grad_val(0, 2), 1.0 / 3.0, 1e-10);
    EXPECT_NEAR(grad_val.row(0).sum(), 1.0, 1e-10);
}

// ======================================================================
// Sigmoid Tests
// ======================================================================

TEST(SurrogateTests, SigmoidNumeric) {
    // Tanh type (default): tanh(0) = 0. Normalized to [0, 1] -> 0.5
    EXPECT_NEAR(sigmoid(0.0), 0.5, 1e-5);

    // Large pos -> 1.0
    EXPECT_NEAR(sigmoid(100.0), 1.0, 1e-5);

    // Large neg -> 0.0
    EXPECT_NEAR(sigmoid(-100.0), 0.0, 1e-5);

    // Custom range [-1, 1]
    EXPECT_NEAR(sigmoid(0.0, SigmoidType::Tanh, -1.0, 1.0), 0.0, 1e-5);
}

TEST(SurrogateTests, SigmoidSymbolic) {
    auto x = metis::sym("x");
    auto expr = sigmoid(x); // default 0 to 1
    double val = metis::Function({x}, {expr}).eval(0.0)(0, 0);
    EXPECT_NEAR(val, 0.5, 1e-5);
}

// ======================================================================
// Swish Tests
// ======================================================================

TEST(SurrogateTests, SwishNumeric) {
    // swish(0) = 0 / ... = 0
    EXPECT_NEAR(swish(0.0), 0.0, 1e-5);

    // swish(large) -> large
    EXPECT_NEAR(swish(10.0), 10.0 / (1.0 + std::exp(-10.0)), 1e-5);
}

TEST(SurrogateTests, SwishSymbolic) {
    auto x = metis::sym("x");
    auto expr = swish(x);
    double val = metis::Function({x}, {expr}).eval(2.0)(0, 0);
    // 2 / (1 + exp(-2))
    EXPECT_NEAR(val, 2.0 / (1.0 + std::exp(-2.0)), 1e-5);
}

// ======================================================================
// Blend Tests
// ======================================================================

TEST(SurrogateTests, BlendNumeric) {
    double low = 0.0;
    double high = 10.0;

    // switch=0 -> roughly average depending on sigmoid shape.
    // sigmoid(0) is 0.5. blend = 10*0.5 + 0*0.5 = 5.
    EXPECT_NEAR(blend(0.0, high, low), 5.0, 1e-5);

    // switch large positive -> high
    EXPECT_NEAR(blend(10.0, high, low), 10.0, 1e-2);

    // switch large negative -> low
    EXPECT_NEAR(blend(-10.0, high, low), 0.0, 1e-2);
}

TEST(SurrogateTests, BlendSymbolic) {
    auto s = metis::sym("s");
    auto expr = blend(s, 10.0, 0.0);

    // s=0 -> 5.0
    double val = metis::Function({s}, {expr}).eval(0.0)(0, 0);
    EXPECT_NEAR(val, 5.0, 1e-5);
}

TEST(SurrogateTests, CoverageErrors) {
    std::vector<double> empty;
    EXPECT_THROW(metis::softmax(empty), metis::InvalidArgument);

    std::vector<double> valid = {1.0, 2.0};
    EXPECT_THROW(metis::softmax(valid, -1.0), metis::InvalidArgument); // Invalid softness
    EXPECT_THROW(metis::softmax(valid, 0.0), metis::InvalidArgument);  // Invalid softness

    EXPECT_THROW(metis::smooth_abs(0.0, 0.0), metis::InvalidArgument);
    EXPECT_THROW(metis::smooth_max(0.0, 1.0, -1.0), metis::InvalidArgument);
    EXPECT_THROW(metis::smooth_min(0.0, 1.0, -1.0), metis::InvalidArgument);
    EXPECT_THROW(metis::smooth_clamp(0.0, -1.0, 1.0, -1.0), metis::InvalidArgument);
    EXPECT_THROW(metis::ks_max(std::vector<double>{}, 1.0), metis::InvalidArgument);
    EXPECT_THROW(metis::ks_max(std::vector<double>{1.0, 2.0}, 0.0), metis::InvalidArgument);
}

} // namespace test
} // namespace metis
