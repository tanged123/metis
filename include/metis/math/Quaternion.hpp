#pragma once
/**
 * @file Quaternion.hpp
 * @brief Quaternion algebra, rotation, and SLERP interpolation
 * @see Rotations.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Arithmetic.hpp"
#include "metis/math/Linalg.hpp"
#include "metis/math/Logic.hpp"
#include "metis/math/Trig.hpp"

namespace metis {

/**
 * @brief Quaternion class for rotation representation
 *
 * Stores quaternion in (w, x, y, z) convention where w is the scalar part.
 * All operations support both numeric and symbolic types.
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 */
template <typename Scalar> class Quaternion {
  public:
    Scalar w, x, y, z;

    // --- Constructors ---

    /// Default constructor: Identity quaternion (1, 0, 0, 0)
    Quaternion()
        : w(static_cast<Scalar>(1.0)), x(static_cast<Scalar>(0.0)), y(static_cast<Scalar>(0.0)),
          z(static_cast<Scalar>(0.0)) {}

    /// Component constructor
    Quaternion(Scalar w, Scalar x, Scalar y, Scalar z) : w(w), x(x), y(y), z(z) {}

    /// Construct from Vec4 [w, x, y, z]
    explicit Quaternion(const Vec4<Scalar> &v) : w(v(0)), x(v(1)), y(v(2)), z(v(3)) {}

    // --- Algebraic Operations ---

    /// @brief Hamilton product
    /// @param other Right-hand quaternion
    /// @return Product quaternion
    Quaternion operator*(const Quaternion &other) const {
        return Quaternion(w * other.w - x * other.x - y * other.y - z * other.z,
                          w * other.x + x * other.w + y * other.z - z * other.y,
                          w * other.y - x * other.z + y * other.w + z * other.x,
                          w * other.z + x * other.y - y * other.x + z * other.w);
    }

    /// @brief Scalar multiplication
    /// @param s Scalar factor
    /// @return Scaled quaternion
    Quaternion operator*(const Scalar &s) const { return Quaternion(w * s, x * s, y * s, z * s); }

    /// @brief Quaternion addition
    /// @param other Right-hand quaternion
    /// @return Sum quaternion
    Quaternion operator+(const Quaternion &other) const {
        return Quaternion(w + other.w, x + other.x, y + other.y, z + other.z);
    }

    /// @brief Conjugate (w, -x, -y, -z)
    /// @return Conjugate quaternion
    Quaternion conjugate() const { return Quaternion(w, -x, -y, -z); }

    /// @brief Inverse (conjugate / norm_sq)
    /// @return Inverse quaternion
    Quaternion inverse() const { return conjugate() * (static_cast<Scalar>(1.0) / squared_norm()); }

    /// @brief Squared norm (w^2 + x^2 + y^2 + z^2)
    /// @return Squared norm
    Scalar squared_norm() const { return w * w + x * x + y * y + z * z; }

    /// @brief Quaternion norm
    /// @return Euclidean norm
    Scalar norm() const { return metis::sqrt(squared_norm()); }

    /// @brief Return unit quaternion
    /// @return Normalized copy
    Quaternion normalized() const {
        Scalar n = norm();
        // Avoid division by zero check for symbolic if possible,
        // strictly speaking we should probably use a safe variant or assume it's not zero.
        // For now, standard division.
        return Quaternion(w / n, x / n, y / n, z / n);
    }

    /// @brief Rotate a 3D vector: v_rot = q * v * q_conj
    /// @param v Input vector
    /// @return Rotated vector
    Vec3<Scalar> rotate(const Vec3<Scalar> &v) const {
        // Optimization: q * (0, v) * q_conj
        // Or specific formula: v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)

        Scalar q0 = w;
        Scalar q1 = x;
        Scalar q2 = y;
        Scalar q3 = z;

        // Extract vector part of quaternion
        Vec3<Scalar> q_vec;
        q_vec << q1, q2, q3;

        Vec3<Scalar> t = static_cast<Scalar>(2.0) * metis::cross(q_vec, v);
        return v + (q0 * t) + metis::cross(q_vec, t);
    }

    // --- Conversions ---

    /// @brief Convert to 3x3 rotation matrix
    /// @return Rotation matrix
    Mat3<Scalar> to_rotation_matrix() const {
        Mat3<Scalar> R;
        Scalar one = static_cast<Scalar>(1.0);
        Scalar two = static_cast<Scalar>(2.0);

        Scalar xx = x * x;
        Scalar yy = y * y;
        Scalar zz = z * z;
        Scalar xy = x * y;
        Scalar xz = x * z;
        Scalar yz = y * z;
        Scalar wx = w * x;
        Scalar wy = w * y;
        Scalar wz = w * z;

        R(0, 0) = one - two * (yy + zz);
        R(0, 1) = two * (xy - wz);
        R(0, 2) = two * (xz + wy);

        R(1, 0) = two * (xy + wz);
        R(1, 1) = one - two * (xx + zz);
        R(1, 2) = two * (yz - wx);

        R(2, 0) = two * (xz - wy);
        R(2, 1) = two * (yz + wx);
        R(2, 2) = one - two * (xx + yy);

        return R;
    }

    /// @brief Export as vector [w, x, y, z]
    /// @return 4-element coefficient vector
    Vec4<Scalar> coeffs() const {
        Vec4<Scalar> res;
        res << w, x, y, z;
        return res;
    }

    // --- Static Factories ---

    /**
     * @brief Create from Euler Angles (Yaw-Pitch-Roll / Z-Y-X sequence)
     * @param roll Roll angle (radians)
     * @param pitch Pitch angle (radians)
     * @param yaw Yaw angle (radians)
     * @return Quaternion
     * @see rotation_matrix_from_euler_angles
     */
    static Quaternion from_euler(Scalar roll, Scalar pitch, Scalar yaw) {
        Scalar half = static_cast<Scalar>(0.5);
        Scalar cr = metis::cos(roll * half);
        Scalar sr = metis::sin(roll * half);
        Scalar cp = metis::cos(pitch * half);
        Scalar sp = metis::sin(pitch * half);
        Scalar cy = metis::cos(yaw * half);
        Scalar sy = metis::sin(yaw * half);

        return Quaternion(cr * cp * cy + sr * sp * sy, // w
                          sr * cp * cy - cr * sp * sy, // x
                          cr * sp * cy + sr * cp * sy, // y
                          cr * cp * sy - sr * sp * cy  // z
        );
    }

    /**
     * @brief Create from axis-angle representation
     * @param axis Rotation axis (will be normalized)
     * @param angle Rotation angle (radians)
     * @return Quaternion
     */
    static Quaternion from_axis_angle(const Vec3<Scalar> &axis, Scalar angle) {
        Scalar half = static_cast<Scalar>(0.5);
        Scalar s = metis::sin(angle * half);
        Scalar c = metis::cos(angle * half);

        // Assume axis is normalized? Usually safer to normalize.
        // If symbolic, normalization adds complexity, but for correctness it's good.
        // Let's assume user passes normalized axis or we normalize it.
        // Standard library implementations usually assume normalized or normalize.
        // We will normalize to be safe.
        auto n_axis = axis / metis::norm(axis);

        return Quaternion(c, n_axis(0) * s, n_axis(1) * s, n_axis(2) * s);
    }

    /**
     * @brief Create from rotation vector (axis * angle)
     * @param rot_vec Rotation vector
     * @return Quaternion
     */
    static Quaternion from_rotation_vector(const Vec3<Scalar> &rot_vec) {
        Scalar half = static_cast<Scalar>(0.5);
        Scalar eps = static_cast<Scalar>(1e-12);
        Scalar angle = metis::norm(rot_vec);
        Scalar half_angle = angle * half;
        Scalar safe_angle = angle + eps;

        // sin(angle/2)/angle with small-angle fallback (limit = 0.5)
        Scalar scale_raw = metis::sin(half_angle) / safe_angle;
        Scalar scale = metis::where(angle > eps, scale_raw, half);

        return Quaternion(metis::cos(half_angle), rot_vec(0) * scale, rot_vec(1) * scale,
                          rot_vec(2) * scale);
    }

    /**
     * @brief Create from 3x3 rotation matrix
     * @param mat Rotation matrix
     * @return Quaternion
     */
    static Quaternion from_rotation_matrix(const Mat3<Scalar> &mat) {
        // Implementation based on standard robust algorithms (e.g., Eigen's or Shepperd's)
        // Here we use a simplified version for brevity but covering standard cases.
        // For symbolic compatibility, we need to be careful with branching.
        // It is notoriously hard to do robust rotation matrix -> quaternion symbolicly because of
        // the 4-way branching. If we must be symbolic, we might pick one branch (e.g. max trace)
        // and hope, or use 'where'.

        // For now, let's implement a standard numeric-friendly trace check.
        // If Scalar is symbolic (casadi::MX), regular if/else won't work on values.

        Scalar trace = mat.trace();
        Scalar q_w, q_x, q_y, q_z;
        Scalar one = static_cast<Scalar>(1.0);
        Scalar half = static_cast<Scalar>(0.5);
        Scalar two = static_cast<Scalar>(2.0);

        if constexpr (std::is_floating_point_v<Scalar>) {
            // Numeric implementation (efficient branching)
            if (trace > 0) {
                Scalar s = static_cast<Scalar>(0.5) / metis::sqrt(trace + 1.0);
                q_w = 0.25 / s;
                q_x = (mat(2, 1) - mat(1, 2)) * s;
                q_y = (mat(0, 2) - mat(2, 0)) * s;
                q_z = (mat(1, 0) - mat(0, 1)) * s;
            } else {
                if (mat(0, 0) > mat(1, 1) && mat(0, 0) > mat(2, 2)) {
                    Scalar s = 2.0 * metis::sqrt(1.0 + mat(0, 0) - mat(1, 1) - mat(2, 2));
                    q_w = (mat(2, 1) - mat(1, 2)) / s;
                    q_x = 0.25 * s;
                    q_y = (mat(0, 1) + mat(1, 0)) / s;
                    q_z = (mat(0, 2) + mat(2, 0)) / s;
                } else if (mat(1, 1) > mat(2, 2)) {
                    Scalar s = 2.0 * metis::sqrt(1.0 + mat(1, 1) - mat(0, 0) - mat(2, 2));
                    q_w = (mat(0, 2) - mat(2, 0)) / s;
                    q_x = (mat(0, 1) + mat(1, 0)) / s;
                    q_y = 0.25 * s;
                    q_z = (mat(1, 2) + mat(2, 1)) / s;
                } else {
                    Scalar s = 2.0 * metis::sqrt(1.0 + mat(2, 2) - mat(0, 0) - mat(1, 1));
                    q_w = (mat(1, 0) - mat(0, 1)) / s;
                    q_x = (mat(0, 2) + mat(2, 0)) / s;
                    q_y = (mat(1, 2) + mat(2, 1)) / s;
                    q_z = 0.25 * s;
                }
            }
        } else {
            // Symbolic: Full 4-branch using nested metis::where (Shepperd's method)

            // Guard radicands: in symbolic mode all branches are eagerly evaluated,
            // so untaken branches can have negative radicands. Clamp to eps.
            Scalar eps = static_cast<Scalar>(1e-12);

            // Branch 0: trace > 0
            Scalar r0 = metis::where(trace + one > eps, trace + one, eps);
            Scalar s0 = half / metis::sqrt(r0);
            Scalar w0 = static_cast<Scalar>(0.25) / s0;
            Scalar x0 = (mat(2, 1) - mat(1, 2)) * s0;
            Scalar y0 = (mat(0, 2) - mat(2, 0)) * s0;
            Scalar z0 = (mat(1, 0) - mat(0, 1)) * s0;

            // Branch 1: mat(0,0) is largest diagonal
            Scalar r1 = one + mat(0, 0) - mat(1, 1) - mat(2, 2);
            Scalar safe_r1 = metis::where(r1 > eps, r1, eps);
            Scalar s1 = two * metis::sqrt(safe_r1);
            Scalar w1 = (mat(2, 1) - mat(1, 2)) / s1;
            Scalar x1 = static_cast<Scalar>(0.25) * s1;
            Scalar y1 = (mat(0, 1) + mat(1, 0)) / s1;
            Scalar z1 = (mat(0, 2) + mat(2, 0)) / s1;

            // Branch 2: mat(1,1) is largest diagonal
            Scalar r2 = one + mat(1, 1) - mat(0, 0) - mat(2, 2);
            Scalar safe_r2 = metis::where(r2 > eps, r2, eps);
            Scalar s2 = two * metis::sqrt(safe_r2);
            Scalar w2 = (mat(0, 2) - mat(2, 0)) / s2;
            Scalar x2 = (mat(0, 1) + mat(1, 0)) / s2;
            Scalar y2 = static_cast<Scalar>(0.25) * s2;
            Scalar z2 = (mat(1, 2) + mat(2, 1)) / s2;

            // Branch 3: mat(2,2) is largest diagonal
            Scalar r3 = one + mat(2, 2) - mat(0, 0) - mat(1, 1);
            Scalar safe_r3 = metis::where(r3 > eps, r3, eps);
            Scalar s3 = two * metis::sqrt(safe_r3);
            Scalar w3 = (mat(1, 0) - mat(0, 1)) / s3;
            Scalar x3 = (mat(0, 2) + mat(2, 0)) / s3;
            Scalar y3 = (mat(1, 2) + mat(2, 1)) / s3;
            Scalar z3 = static_cast<Scalar>(0.25) * s3;

            // Select via nested where
            auto cond_trace = trace > static_cast<Scalar>(0.0);
            auto cond_r00 = metis::logical_and(mat(0, 0) > mat(1, 1), mat(0, 0) > mat(2, 2));
            auto cond_r11 = mat(1, 1) > mat(2, 2);

            // Inner: branch2 vs branch3
            Scalar wi = metis::where(cond_r11, w2, w3);
            Scalar xi = metis::where(cond_r11, x2, x3);
            Scalar yi = metis::where(cond_r11, y2, y3);
            Scalar zi = metis::where(cond_r11, z2, z3);

            // Middle: branch1 vs inner
            Scalar wm = metis::where(cond_r00, w1, wi);
            Scalar xm = metis::where(cond_r00, x1, xi);
            Scalar ym = metis::where(cond_r00, y1, yi);
            Scalar zm = metis::where(cond_r00, z1, zi);

            // Outer: branch0 vs middle
            q_w = metis::where(cond_trace, w0, wm);
            q_x = metis::where(cond_trace, x0, xm);
            q_y = metis::where(cond_trace, y0, ym);
            q_z = metis::where(cond_trace, z0, zm);
        }
        return Quaternion(q_w, q_x, q_y, q_z);
    }

    /// @brief Extract Euler angles (Roll-Pitch-Yaw / XYZ)
    /// @return Vec3 of (roll, pitch, yaw)
    Vec3<Scalar> to_euler() const {
        // Roll (x-axis rotation)
        Scalar sinr_cosp = static_cast<Scalar>(2.0) * (w * x + y * z);
        Scalar cosr_cosp = static_cast<Scalar>(1.0) - static_cast<Scalar>(2.0) * (x * x + y * y);
        Scalar roll = metis::atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        Scalar sinp = static_cast<Scalar>(2.0) * (w * y - z * x);
        Scalar pitch;
        // Check for gimbal lock
        // Logic::where ideally
        if constexpr (std::is_floating_point_v<Scalar>) {
            if (std::abs(sinp) >= 1)
                pitch = std::copysign(std::numbers::pi_v<double> / 2,
                                      sinp); // use 90 degrees if out of range
            else
                pitch = std::asin(sinp);
        } else {
            // Symbolic: assume no gimbal lock or underlying library handles asin domain
            pitch = metis::asin(sinp);
        }

        // Yaw (z-axis rotation)
        Scalar siny_cosp = static_cast<Scalar>(2.0) * (w * z + x * y);
        Scalar cosy_cosp = static_cast<Scalar>(1.0) - static_cast<Scalar>(2.0) * (y * y + z * z);
        Scalar yaw = metis::atan2(siny_cosp, cosy_cosp);

        return Vec3<Scalar>(roll, pitch, yaw);
    }
};

// --- Free Functions ---

/**
 * @brief Spherical Linear Interpolation (full fidelity)
 *
 * Shortest-path aware with nlerp fallback for near-identity angles.
 *
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 * @param q0 Start quaternion
 * @param q1 End quaternion
 * @param t Interpolation parameter in [0, 1]
 * @return Interpolated unit quaternion
 */
template <typename Scalar>
Quaternion<Scalar> slerp(const Quaternion<Scalar> &q0, const Quaternion<Scalar> &q1, Scalar t) {
    Scalar one = static_cast<Scalar>(1.0);
    Scalar zero = static_cast<Scalar>(0.0);
    Scalar dot_threshold = static_cast<Scalar>(0.9995); // Threshold for linear fallback

    // Compute dot product
    Scalar dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z;

    // --- Shortest path fix ---
    // If dot < 0, negate q1 to take shorter arc.
    // We compute a sign factor: sign = where(dot < 0, -1, 1)
    Scalar sign = metis::where(dot < zero, -one, one);

    // Effective q1 and dot (flipped if needed)
    Quaternion<Scalar> q1_eff(q1.w * sign, q1.x * sign, q1.y * sign, q1.z * sign);
    Scalar dot_eff = dot * sign; // Now dot_eff >= 0

    // --- Numerical stability: handle near-identity case ---
    // If dot_eff is very close to 1, theta ≈ 0 and sin(theta) ≈ 0 (division issues).
    // Fall back to normalized linear interpolation (nlerp).

    Scalar theta = metis::acos(dot_eff);
    Scalar sin_theta = metis::sin(theta);

    // Compute slerp weights
    Scalar wa_slerp = metis::sin((one - t) * theta) / sin_theta;
    Scalar wb_slerp = metis::sin(t * theta) / sin_theta;

    // Compute nlerp weights (simple linear blend, then normalize result)
    Scalar wa_nlerp = one - t;
    Scalar wb_nlerp = t;

    // Use metis::where to select between slerp and nlerp based on dot_eff
    Scalar use_slerp = dot_eff < dot_threshold; // True if slerp is safe

    Scalar wa = metis::where(use_slerp, wa_slerp, wa_nlerp);
    Scalar wb = metis::where(use_slerp, wb_slerp, wb_nlerp);

    // Compute result
    Quaternion<Scalar> result = q0 * wa + q1_eff * wb;

    // Normalize for nlerp case (harmless for slerp case, just ensures unit quaternion)
    return result.normalized();
}

} // namespace metis
