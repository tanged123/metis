#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Trig.hpp>
#include <numbers>

template <typename Scalar> void test_trig_ops() {
    Scalar zero = 0.0;
    Scalar half_pi = std::numbers::pi_v<double> / 2.0; // 1.5707...
    Scalar pi = std::numbers::pi_v<double>;

    // Sin, Cos, Tan
    auto s = metis::sin(zero);
    auto c = metis::cos(zero);
    auto t = metis::tan(zero);

    // Inverse
    auto as = metis::asin(1.0);        // pi/2
    auto ac = metis::acos(1.0);        // 0
    auto at = metis::atan(zero);       // 0
    auto at2 = metis::atan2(1.0, 1.0); // pi/4

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(s, 0.0, 1e-9);
        EXPECT_NEAR(c, 1.0, 1e-9);
        EXPECT_NEAR(t, 0.0, 1e-9);

        EXPECT_NEAR(as, half_pi, 1e-9);
        EXPECT_NEAR(ac, 0.0, 1e-9);
        EXPECT_NEAR(at, 0.0, 1e-9);
        EXPECT_NEAR(at2, pi / 4.0, 1e-9);
    } else {
        EXPECT_NEAR(metis::eval(s), 0.0, 1e-9);
        EXPECT_NEAR(metis::eval(c), 1.0, 1e-9);
        EXPECT_NEAR(metis::eval(t), 0.0, 1e-9);

        EXPECT_NEAR(metis::eval(as), std::numbers::pi_v<double> / 2.0, 1e-9);
        EXPECT_NEAR(metis::eval(ac), 0.0, 1e-9);
        EXPECT_NEAR(metis::eval(at), 0.0, 1e-9);
        EXPECT_NEAR(metis::eval(at2), std::numbers::pi_v<double> / 4.0, 1e-9);
    }
}

TEST(TrigTests, Numeric) { test_trig_ops<double>(); }

TEST(TrigTests, Symbolic) { test_trig_ops<metis::SymbolicScalar>(); }

// --- Inverse Hyperbolic Functions ---

template <typename Scalar> void test_inverse_hyperbolic_ops() {
    // asinh: inverse of sinh, domain is all reals
    Scalar val_asinh = 1.0;
    auto res_asinh = metis::asinh(val_asinh);

    // acosh: inverse of cosh, domain is [1, inf)
    Scalar val_acosh = 2.0;
    auto res_acosh = metis::acosh(val_acosh);

    // atanh: inverse of tanh, domain is (-1, 1)
    Scalar val_atanh = 0.5;
    auto res_atanh = metis::atanh(val_atanh);

    // Test negative value for asinh (should work)
    Scalar val_asinh_neg = -1.0;
    auto res_asinh_neg = metis::asinh(val_asinh_neg);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(res_asinh, std::asinh(1.0), 1e-9);
        EXPECT_NEAR(res_acosh, std::acosh(2.0), 1e-9);
        EXPECT_NEAR(res_atanh, std::atanh(0.5), 1e-9);
        EXPECT_NEAR(res_asinh_neg, std::asinh(-1.0), 1e-9);

        // Verify inverse properties: sinh(asinh(x)) == x
        EXPECT_NEAR(std::sinh(res_asinh), 1.0, 1e-9);
        // cosh(acosh(x)) == x
        EXPECT_NEAR(std::cosh(res_acosh), 2.0, 1e-9);
        // tanh(atanh(x)) == x
        EXPECT_NEAR(std::tanh(res_atanh), 0.5, 1e-9);
    } else {
        EXPECT_FALSE(res_asinh.is_empty());
        EXPECT_NEAR(metis::eval(res_asinh), std::asinh(1.0), 1e-9);
        EXPECT_NEAR(metis::eval(res_acosh), std::acosh(2.0), 1e-9);
        EXPECT_NEAR(metis::eval(res_atanh), std::atanh(0.5), 1e-9);
        EXPECT_NEAR(metis::eval(res_asinh_neg), std::asinh(-1.0), 1e-9);
    }
}

TEST(TrigTests, InverseHyperbolicNumeric) { test_inverse_hyperbolic_ops<double>(); }

TEST(TrigTests, InverseHyperbolicSymbolic) { test_inverse_hyperbolic_ops<metis::SymbolicScalar>(); }

TEST(TrigTests, InverseHyperbolicMatrix) {
    // Numeric matrix
    metis::NumericMatrix M(2, 2);
    M << 0.5, 1.0, 1.5, 2.0;

    auto asinh_result = metis::asinh(M);
    EXPECT_NEAR(asinh_result(0, 0), std::asinh(0.5), 1e-9);
    EXPECT_NEAR(asinh_result(1, 1), std::asinh(2.0), 1e-9);

    // acosh needs values >= 1
    metis::NumericMatrix M_acosh(2, 2);
    M_acosh << 1.0, 1.5, 2.0, 3.0;
    auto acosh_result = metis::acosh(M_acosh);
    EXPECT_NEAR(acosh_result(0, 0), std::acosh(1.0), 1e-9);
    EXPECT_NEAR(acosh_result(1, 1), std::acosh(3.0), 1e-9);

    // atanh needs values in (-1, 1)
    metis::NumericMatrix M_atanh(2, 2);
    M_atanh << -0.5, 0.0, 0.5, 0.9;
    auto atanh_result = metis::atanh(M_atanh);
    EXPECT_NEAR(atanh_result(0, 0), std::atanh(-0.5), 1e-9);
    EXPECT_NEAR(atanh_result(1, 0), std::atanh(0.5), 1e-9);

    // Symbolic matrix
    metis::SymbolicMatrix Ms = metis::to_eigen(metis::to_mx(M));
    auto asinh_s = metis::asinh(Ms);
    auto asinh_s_eval = metis::eval(asinh_s);
    EXPECT_NEAR(asinh_s_eval(0, 0), std::asinh(0.5), 1e-9);
}
