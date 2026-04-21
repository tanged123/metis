#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Linalg.hpp> // to_mx
#include <metis/math/Rotations.hpp>
#include <numbers>

template <typename Scalar> void test_rotations() {
    Scalar theta = std::numbers::pi_v<double> / 2.0;
    auto R2 = metis::rotation_matrix_2d(theta);
    auto R3 = metis::rotation_matrix_3d(theta, 2); // Z-axis

    if constexpr (std::is_same_v<Scalar, double>) {
        // R2: pi/2 -> [0 -1; 1 0]
        EXPECT_NEAR(R2(0, 0), 0.0, 1e-9);
        EXPECT_NEAR(R2(0, 1), -1.0, 1e-9);

        // R3: Z axis similar to 2D in top-left
        EXPECT_NEAR(R3(0, 0), 0.0, 1e-9);
        EXPECT_NEAR(R3(0, 1), -1.0, 1e-9);
        EXPECT_NEAR(R3(2, 2), 1.0, 1e-9);
    } else {
        auto R2_eval = metis::eval(R2);
        EXPECT_NEAR(R2_eval(0, 0), 0.0, 1e-9);
        EXPECT_NEAR(R2_eval(0, 1), -1.0, 1e-9);

        auto R3_eval = metis::eval(R3);
        EXPECT_NEAR(R3_eval(0, 0), 0.0, 1e-9);
        EXPECT_NEAR(R3_eval(0, 1), -1.0, 1e-9);
        EXPECT_NEAR(R3_eval(2, 2), 1.0, 1e-9);
    }

    // --- Euler Angles Test ---
    // Roll=pi, Pitch=0, Yaw=0 -> 180 deg around X. Result: Diag(1, -1, -1)
    Scalar pi = std::numbers::pi_v<double>;
    auto R_euler = metis::rotation_matrix_from_euler_angles(pi, Scalar(0.0), Scalar(0.0));

    // --- Validation Test ---
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_TRUE(metis::is_valid_rotation_matrix(R2, 1e-9)); // 2D matrix?
        // Note: is_valid expects 3x3 for the cross product logic?
        // Wait, my implementation uses transpose*self=I and det=1.
        // It works for any square matrix.
        EXPECT_TRUE(metis::is_valid_rotation_matrix(R3));
        EXPECT_TRUE(metis::is_valid_rotation_matrix(R_euler));

        EXPECT_NEAR(R_euler(0, 0), 1.0, 1e-9);
        EXPECT_NEAR(R_euler(1, 1), -1.0, 1e-9);
        EXPECT_NEAR(R_euler(2, 2), -1.0, 1e-9);

        // Invalid matrix test
        metis::Mat3<Scalar> bad = metis::Mat3<Scalar>::Identity() * 2.0;
        EXPECT_FALSE(metis::is_valid_rotation_matrix(bad));
    } else {
        auto valid_R3 = metis::is_valid_rotation_matrix(R3);
        EXPECT_NEAR(metis::eval(valid_R3), 1.0, 1e-9);

        auto valid_euler = metis::is_valid_rotation_matrix(R_euler);
        EXPECT_NEAR(metis::eval(valid_euler), 1.0, 1e-9);

        auto R_euler_eval = metis::eval(R_euler);
        EXPECT_NEAR(R_euler_eval(0, 0), 1.0, 1e-9);
        EXPECT_NEAR(R_euler_eval(1, 1), -1.0, 1e-9);
        EXPECT_NEAR(R_euler_eval(2, 2), -1.0, 1e-9);

        // Invalid matrix test
        // Construct invalid symbolic matrix
        metis::Mat3<metis::SymbolicScalar> bad;
        bad.setIdentity();
        bad = bad * 2.0;
        auto valid_bad = metis::is_valid_rotation_matrix(bad);
        EXPECT_NEAR(metis::eval(valid_bad), 0.0, 1e-9);
    }
}

TEST(RotationsTests, RotationsNumeric) { test_rotations<double>(); }

TEST(RotationsTests, RotationsSymbolic) { test_rotations<metis::SymbolicScalar>(); }

TEST(RotationsTests, CoverageAxes) {
    // Explicitly test all axes for rotation_matrix_3d coverage
    double theta = 0.1;
    auto R0 = metis::rotation_matrix_3d(theta, 0);
    auto R1 = metis::rotation_matrix_3d(theta, 1);
    auto R2 = metis::rotation_matrix_3d(theta, 2);

    EXPECT_NEAR(R0(1, 1), std::cos(theta), 1e-9);
    EXPECT_NEAR(R1(0, 0), std::cos(theta), 1e-9);
    EXPECT_NEAR(R2(0, 0), std::cos(theta), 1e-9);

    // Invalid axis should throw
    EXPECT_THROW(metis::rotation_matrix_3d(theta, 99), metis::InvalidArgument);
}
