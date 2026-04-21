#include "../utils/TestUtils.hpp"
#include "metis/math/MetisMath.hpp"
#include "metis/math/Quaternion.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <numbers>

// Helper to check approximate equality of quaternions (account for q == -q duality?)
// For now, strict element-wise check is easier if we fix inputs.

template <typename Scalar> void test_quaternion_construction() {
    metis::Quaternion<Scalar> q_id;

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(q_id.w, 1.0, 1e-9);
        EXPECT_NEAR(q_id.x, 0.0, 1e-9);
    } else {
        EXPECT_NEAR(metis::eval(q_id.w), 1.0, 1e-9);
        EXPECT_NEAR(metis::eval(q_id.x), 0.0, 1e-9);
    }

    metis::Quaternion<Scalar> q(1.0, 2.0, 3.0, 4.0);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(q.w, 1.0, 1e-9);
        EXPECT_NEAR(q.z, 4.0, 1e-9);
    } else {
        EXPECT_NEAR(metis::eval(q.w), 1.0, 1e-9);
        EXPECT_NEAR(metis::eval(q.z), 4.0, 1e-9);
    }

    // From Eigen Vector
    Eigen::Matrix<Scalar, 4, 1> v;
    v << static_cast<Scalar>(1.0), static_cast<Scalar>(2.0), static_cast<Scalar>(3.0),
        static_cast<Scalar>(4.0);
    metis::Quaternion<Scalar> qv(v);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(qv.y, 3.0, 1e-9);
    } else {
        EXPECT_NEAR(metis::eval(qv.y), 3.0, 1e-9);
    }
}

TEST(QuaternionTest, ConstructionNumeric) { test_quaternion_construction<double>(); }

TEST(QuaternionTest, ConstructionSymbolic) {
    test_quaternion_construction<metis::SymbolicScalar>();
}

template <typename Scalar> void test_quaternion_algebra() {
    // i * j = k
    metis::Quaternion<Scalar> qi(0.0, 1.0, 0.0, 0.0);
    metis::Quaternion<Scalar> qj(0.0, 0.0, 1.0, 0.0);
    metis::Quaternion<Scalar> qk = qi * qj;

    if constexpr (std::is_same_v<Scalar, double>) {
        // k = (0, 0, 0, 1)
        EXPECT_NEAR(qk.w, 0.0, 1e-9);
        EXPECT_NEAR(qk.x, 0.0, 1e-9);
        EXPECT_NEAR(qk.y, 0.0, 1e-9);
        EXPECT_NEAR(qk.z, 1.0, 1e-9);

        // j * i = -k
        metis::Quaternion<Scalar> qnk = qj * qi;
        EXPECT_NEAR(qnk.z, -1.0, 1e-9);

        // Conjugate
        auto qj_conj = qj.conjugate();
        EXPECT_NEAR(qj_conj.y, -1.0, 1e-9);

        // Inverse
        auto qj_inv = qj.inverse();
        EXPECT_NEAR(qj_inv.y, -1.0, 1e-9);

        // General Inverse
        metis::Quaternion<Scalar> q(1.0, 1.0, 1.0, 1.0);
        auto q_imp = q.inverse();
        EXPECT_NEAR(q_imp.w, 0.25, 1e-9);
        EXPECT_NEAR(q_imp.x, -0.25, 1e-9);

        // Reversibility: q * q_inv = Identity
        auto q_unity = q * q_imp;
        EXPECT_NEAR(q_unity.w, 1.0, 1e-9);
        EXPECT_NEAR(q_unity.x, 0.0, 1e-9);
    } else {
        EXPECT_NEAR(metis::eval(qk.w), 0.0, 1e-9);
        EXPECT_NEAR(metis::eval(qk.z), 1.0, 1e-9);

        metis::Quaternion<Scalar> qnk = qj * qi;
        EXPECT_NEAR(metis::eval(qnk.z), -1.0, 1e-9);

        auto qj_conj = qj.conjugate();
        EXPECT_NEAR(metis::eval(qj_conj.y), -1.0, 1e-9);

        metis::Quaternion<Scalar> q(1.0, 1.0, 1.0, 1.0);
        auto q_imp = q.inverse();
        EXPECT_NEAR(metis::eval(q_imp.w), 0.25, 1e-9);

        auto q_unity = q * q_imp;
        EXPECT_NEAR(metis::eval(q_unity.w), 1.0, 1e-9);
    }
}

TEST(QuaternionTest, AlgebraNumeric) { test_quaternion_algebra<double>(); }

TEST(QuaternionTest, AlgebraSymbolic) { test_quaternion_algebra<metis::SymbolicScalar>(); }

template <typename Scalar> void test_rotation() {
    // Rotate vector [1, 0, 0] by 90 degrees around Z axis.
    // Result should be [0, 1, 0].

    Scalar yaw = std::numbers::pi_v<double> / 2.0;
    metis::Quaternion<Scalar> q = metis::Quaternion<Scalar>::from_euler(
        static_cast<Scalar>(0.0), static_cast<Scalar>(0.0), yaw);

    Eigen::Matrix<Scalar, 3, 1> v;
    v << static_cast<Scalar>(1.0), static_cast<Scalar>(0.0), static_cast<Scalar>(0.0);

    auto v_rot = q.rotate(v);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(v_rot(0), 0.0, 1e-9);
        EXPECT_NEAR(v_rot(1), 1.0, 1e-9);
        EXPECT_NEAR(v_rot(2), 0.0, 1e-9);

        // Rotation vector (Axis-Angle)
        Eigen::Matrix<Scalar, 3, 1> axis;
        axis << 0.0, 0.0, 1.0;
        auto q_aa = metis::Quaternion<Scalar>::from_axis_angle(axis, yaw);

        EXPECT_NEAR(q.w, q_aa.w, 1e-9);
        EXPECT_NEAR(q.z, q_aa.z, 1e-9);

        // From rotation vector
        Eigen::Matrix<Scalar, 3, 1> rot_vec = axis * yaw;
        auto q_rv = metis::Quaternion<Scalar>::from_rotation_vector(rot_vec);
        EXPECT_NEAR(q.w, q_rv.w, 1e-9);
        EXPECT_NEAR(q.z, q_rv.z, 1e-9);
    } else {
        auto v_rot_eval = metis::eval(v_rot);
        EXPECT_NEAR(v_rot_eval(0), 0.0, 1e-9);
        EXPECT_NEAR(v_rot_eval(1), 1.0, 1e-9);
        EXPECT_NEAR(v_rot_eval(2), 0.0, 1e-9);
    }
}

TEST(QuaternionTest, RotationNumeric) { test_rotation<double>(); }

TEST(QuaternionTest, RotationSymbolic) { test_rotation<metis::SymbolicScalar>(); }

template <typename Scalar> void test_conversions() {
    Scalar roll = 0.5;
    Scalar pitch = -0.3;
    Scalar yaw = 1.2;

    auto q = metis::Quaternion<Scalar>::from_euler(roll, pitch, yaw);
    auto R = q.to_rotation_matrix();

    if constexpr (std::is_same_v<Scalar, double>) {
        // Matrix -> Quat
        auto q_recon = metis::Quaternion<Scalar>::from_rotation_matrix(R);

        // Check round trip (q_recon should match q or -q)
        Scalar dot = q.w * q_recon.w + q.x * q_recon.x + q.y * q_recon.y + q.z * q_recon.z;
        EXPECT_NEAR(std::abs(dot), 1.0, 1e-9);

        // Quat -> Euler
        auto euler = q.to_euler();

        EXPECT_NEAR(euler(0), roll, 1e-9);
        EXPECT_NEAR(euler(1), pitch, 1e-9);
        EXPECT_NEAR(euler(2), yaw, 1e-9);
    } else {
        // For symbolic, just verify to_rotation_matrix produces valid output
        auto R_eval = metis::eval(R);
        // Should be orthonormal (R^T * R ≈ I)
        auto should_be_I = R_eval.transpose() * R_eval;
        EXPECT_NEAR(should_be_I(0, 0), 1.0, 1e-9);
        EXPECT_NEAR(should_be_I(1, 1), 1.0, 1e-9);
        EXPECT_NEAR(should_be_I(2, 2), 1.0, 1e-9);
    }
}

TEST(QuaternionTest, ConversionNumeric) { test_conversions<double>(); }

TEST(QuaternionTest, ConversionSymbolic) { test_conversions<metis::SymbolicScalar>(); }

template <typename Scalar> void test_slerp() {
    // 0 deg and 90 deg Z-rotation
    auto q0 = metis::Quaternion<Scalar>::from_euler(
        static_cast<Scalar>(0.0), static_cast<Scalar>(0.0), static_cast<Scalar>(0.0));
    auto q1 = metis::Quaternion<Scalar>::from_euler(
        static_cast<Scalar>(0.0), static_cast<Scalar>(0.0),
        static_cast<Scalar>(std::numbers::pi_v<double> / 2.0));

    auto q_half = metis::slerp(q0, q1, static_cast<Scalar>(0.5));
    auto euler = q_half.to_euler();

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(euler(2), std::numbers::pi_v<double> / 4.0, 1e-9);

        // --- Shortest path test ---
        auto q1_neg = metis::Quaternion<Scalar>(-q1.w, -q1.x, -q1.y, -q1.z);
        auto q_half_neg = metis::slerp(q0, q1_neg, 0.5);
        auto euler_neg = q_half_neg.to_euler();

        EXPECT_NEAR(euler_neg(2), std::numbers::pi_v<double> / 4.0, 1e-9);

        // --- Near-identity test (nlerp fallback) ---
        auto q_tiny = metis::Quaternion<Scalar>::from_euler(0.0, 0.0, 1e-8);
        auto q_identity = metis::Quaternion<Scalar>::from_euler(0.0, 0.0, 0.0);
        auto q_interp = metis::slerp(q_identity, q_tiny, 0.5);

        auto euler_tiny = q_interp.to_euler();
        EXPECT_NEAR(euler_tiny(2), 0.5e-8, 1e-12);
    } else {
        auto euler_eval = metis::eval(euler);
        EXPECT_NEAR(euler_eval(2), std::numbers::pi_v<double> / 4.0, 1e-9);
    }
}

TEST(QuaternionTest, SlerpNumeric) { test_slerp<double>(); }

TEST(QuaternionTest, SlerpSymbolic) { test_slerp<metis::SymbolicScalar>(); }

TEST(QuaternionTest, CoverageBranches) {
    // 1. rom_rotation_matrix branches (Trace <= 0)
    // Case 1: max diag is 0,0 (already tested implicitly or rarely hit?)
    // Construct matrices that force specific branches

    // Rotation 180 deg around X: [1 0 0; 0 -1 0; 0 0 -1], trace = -1. Max diag is 0,0.
    Eigen::Matrix3d Rx_180;
    Rx_180 << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    auto qx = metis::Quaternion<double>::from_rotation_matrix(Rx_180);
    EXPECT_NEAR(qx.x, 1.0, 1e-9);

    // Rotation 180 deg around Y: [-1 0 0; 0 1 0; 0 0 -1], trace = -1. Max diag is 1,1.
    Eigen::Matrix3d Ry_180;
    Ry_180 << -1, 0, 0, 0, 1, 0, 0, 0, -1;
    auto qy = metis::Quaternion<double>::from_rotation_matrix(Ry_180);
    EXPECT_NEAR(qy.y, 1.0, 1e-9);

    // Rotation 180 deg around Z: [-1 0 0; 0 -1 0; 0 0 1], trace = -1. Max diag is 2,2.
    Eigen::Matrix3d Rz_180;
    Rz_180 << -1, 0, 0, 0, -1, 0, 0, 0, 1;
    auto qz = metis::Quaternion<double>::from_rotation_matrix(Rz_180);
    EXPECT_NEAR(qz.z, 1.0, 1e-9);
}

TEST(QuaternionTest, CoverageSlerpLimits) {
    auto q0 = metis::Quaternion<double>();           // Identity
    auto q1 = metis::Quaternion<double>(0, 1, 0, 0); // 180 deg rot around X

    // t=0
    auto res0 = metis::slerp(q0, q1, 0.0);
    EXPECT_NEAR(res0.w, 1.0, 1e-9);

    // t=1
    auto res1 = metis::slerp(q0, q1, 1.0);
    EXPECT_NEAR(res1.x, 1.0, 1e-9);

    // Test for Quaternion addition operator line 51 and mult operator line 43
    metis::Quaternion<double> qa(1, 0, 0, 0);
    metis::Quaternion<double> qb(0, 1, 0, 0);
    auto qc = qa + qb;
    EXPECT_NEAR(qc.w, 1.0, 1e-9);
    EXPECT_NEAR(qc.x, 1.0, 1e-9);

    // Symbolic addition/mult check
    metis::Quaternion<metis::SymbolicScalar> sa, sb;
    auto sc = sa + sb; // hits line 50/51 symbolic instantiation
    EXPECT_NEAR(metis::eval(sc.w), 2.0, 1e-9);

    // Line 262/263: Pitch gimbal lock case (sinp >= 1)
    // Create quaternion corresponding to pitch=pi/2
    auto q_lock = metis::Quaternion<double>::from_euler(0.0, std::numbers::pi_v<double> / 2.0, 0.0);
    auto euler = q_lock.to_euler();
    EXPECT_NEAR(euler(1), std::numbers::pi_v<double> / 2.0, 1e-5);

    auto q_lock_neg =
        metis::Quaternion<double>::from_euler(0.0, -std::numbers::pi_v<double> / 2.0, 0.0);
    auto euler_neg = q_lock_neg.to_euler();
    EXPECT_NEAR(euler_neg(1), -std::numbers::pi_v<double> / 2.0, 1e-5);
}
