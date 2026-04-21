/**
 * @file aircraft_attitudes.cpp
 * @brief Graduation Example: Aircraft Attitude Dynamics
 *
 * Demonstrates Phase 3 Metis features:
 * - Euler angles and rotation matrices
 * - Discrete integration for state propagation
 * - Finite differences for rate estimation
 * - Gradient computation
 * - Symbolic/Numeric duality
 *
 * Simulates a simple aircraft performing a coordinated turn:
 * - Roll into turn
 * - Hold bank angle
 * - Roll out of turn
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <numbers>

using namespace metis;

int main() {
    std::cout << "=== Aircraft Attitude Dynamics ===" << std::endl;
    std::cout << "Demonstrating Phase 3 Metis features\n" << std::endl;

    // --- 1. Define time history ---
    constexpr int N = 51;                    // Time steps
    constexpr double dt = 0.1;               // 100ms steps
    constexpr double T_total = (N - 1) * dt; // 5 seconds total

    auto t = linspace(0.0, T_total, N);

    std::cout << "1. Simulation Setup" << std::endl;
    std::cout << "   Duration: " << T_total << " s" << std::endl;
    std::cout << "   Time steps: " << N << std::endl;

    // --- 2. Define attitude angles (Euler: roll, pitch, yaw) ---
    std::cout << "\n2. Attitude Profile (Coordinated Turn)" << std::endl;

    MetisVector<double> roll(N), pitch(N), yaw(N);

    // Maneuver phases:
    // Phase 1 (0-1s): Roll into 30° bank
    // Phase 2 (1-4s): Hold bank, yaw rate = g*tan(bank)/V ≈ const
    // Phase 3 (4-5s): Roll out
    constexpr double bank_max = 30.0 * std::numbers::pi / 180.0; // 30 degrees
    constexpr double yaw_rate = 0.15;                            // rad/s (coordinated turn rate)

    for (int i = 0; i < N; ++i) {
        double time = t(i);

        // Roll profile (smooth ramp using blend)
        if (time < 1.0) {
            // Roll in: 0 to bank_max over 1s
            roll(i) = bank_max * sigmoid((time - 0.5) * 5.0); // Smooth S-curve
        } else if (time < 4.0) {
            roll(i) = bank_max; // Hold
        } else {
            // Roll out
            roll(i) = bank_max * (1.0 - sigmoid((time - 4.5) * 5.0));
        }

        // Pitch: slight nose-up in turn to maintain altitude
        pitch(i) = 0.02 * roll(i) / bank_max; // 2° at max bank

        // Yaw: integrate turn rate while banked
        if (i == 0) {
            yaw(i) = 0.0;
        } else {
            // Simple Euler integration of yaw rate proportional to roll
            double bank_factor = std::sin(roll(i - 1)) / std::sin(bank_max);
            yaw(i) = yaw(i - 1) + yaw_rate * bank_factor * dt;
        }
    }

    std::cout << "   Max bank angle: " << bank_max * 180.0 / std::numbers::pi << " deg"
              << std::endl;
    std::cout << "   Final heading change: " << yaw(N - 1) * 180.0 / std::numbers::pi << " deg"
              << std::endl;

    // --- 3. Compute rotation matrices at key points ---
    std::cout << "\n3. Rotation Matrices (Body to Earth)" << std::endl;

    // At max bank (t ≈ 2.5s, i=25)
    int i_mid = N / 2;
    auto R_mid = rotation_matrix_from_euler_angles(roll(i_mid), pitch(i_mid), yaw(i_mid));

    std::cout << "   At t=" << t(i_mid) << "s (max bank):" << std::endl;
    std::cout << "   Roll=" << roll(i_mid) * 180 / std::numbers::pi << "°, ";
    std::cout << "Pitch=" << pitch(i_mid) * 180 / std::numbers::pi << "°, ";
    std::cout << "Yaw=" << yaw(i_mid) * 180 / std::numbers::pi << "°" << std::endl;

    // Validate rotation matrix
    bool is_valid = is_valid_rotation_matrix(R_mid);
    std::cout << "   Rotation matrix valid: " << (is_valid ? "Yes" : "No") << std::endl;

    // Transform a body-axis vector to earth axes
    Eigen::Matrix<double, 3, 1> body_fwd;
    body_fwd << 1.0, 0.0, 0.0; // Forward in body frame
    auto earth_fwd = R_mid * body_fwd;
    std::cout << "   Forward vector (earth): [" << earth_fwd.transpose() << "]" << std::endl;

    // --- 4. Angular rates from finite differences ---
    std::cout << "\n4. Angular Rates (Finite Differences)" << std::endl;

    // Compute roll rate using gradient
    auto roll_rate = gradient(roll, t, 2); // edge_order=2 for accuracy
    auto pitch_rate = gradient(pitch, t, 2);
    auto yaw_rate_computed = gradient(yaw, t, 2);

    std::cout << "   Max roll rate: " << roll_rate.array().abs().maxCoeff() * 180 / std::numbers::pi
              << " deg/s" << std::endl;
    std::cout << "   Max yaw rate: "
              << yaw_rate_computed.array().abs().maxCoeff() * 180 / std::numbers::pi << " deg/s"
              << std::endl;

    // --- 5. Path integration ---
    std::cout << "\n5. Turn Integration (Discrete Integration)" << std::endl;

    // Integrate yaw rate to verify final heading
    auto yaw_integrated = integrate_discrete_intervals(yaw_rate_computed, t, true, "simpson");
    double total_heading_change = yaw_integrated.sum();

    std::cout << "   Integrated heading change: " << total_heading_change * 180 / std::numbers::pi
              << " deg" << std::endl;
    std::cout << "   Direct yaw change: " << yaw(N - 1) * 180 / std::numbers::pi << " deg"
              << std::endl;
    std::cout << "   Error: "
              << std::abs(total_heading_change - yaw(N - 1)) * 180 / std::numbers::pi << " deg"
              << std::endl;

    // --- 6. Smoothness analysis ---
    std::cout << "\n6. Smoothness Analysis (Squared Curvature)" << std::endl;

    auto roll_curvature = integrate_discrete_squared_curvature(roll, t, "hybrid_simpson_cubic");
    auto yaw_curvature = integrate_discrete_squared_curvature(yaw, t, "hybrid_simpson_cubic");

    std::cout << "   Roll smoothness penalty: " << roll_curvature.sum() << std::endl;
    std::cout << "   Yaw smoothness penalty: " << yaw_curvature.sum() << std::endl;
    std::cout << "   (Lower = smoother maneuver)" << std::endl;

    // --- 7. Symbolic gradient computation ---
    std::cout << "\n7. Symbolic Sensitivity Analysis" << std::endl;

    // Demonstrate symbolic mode: sensitivity of final yaw to initial roll rate
    auto roll_sym = sym("roll");
    auto yaw_sym = sym("yaw");

    // Simple model: yaw = f(roll) via turn coordination
    auto yaw_model = yaw_sym + 0.5 * metis::sin(roll_sym);

    // Compute Jacobian
    auto J = jacobian(yaw_model, roll_sym);
    std::cout << "   d(yaw)/d(roll) = " << J << std::endl;

    // Evaluate at specific roll angle
    Function eval_jac("eval_jac", {roll_sym}, {J});
    auto J_val = eval_jac(bank_max);
    std::cout << "   At roll=30°: d(yaw)/d(roll) = " << metis::eval(J_val[0](0, 0)) << std::endl;

    // --- 8. Summary table ---
    std::cout << "\n8. Maneuver Summary Table" << std::endl;
    std::cout << std::setw(8) << "Time" << std::setw(12) << "Roll(deg)" << std::setw(12)
              << "Yaw(deg)" << std::setw(14) << "YawRate(°/s)" << std::endl;
    std::cout << std::string(46, '-') << std::endl;

    for (int i = 0; i < N; i += 10) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(8) << t(i) << std::setw(12) << roll(i) * 180 / std::numbers::pi
                  << std::setw(12) << yaw(i) * 180 / std::numbers::pi << std::setw(14)
                  << yaw_rate_computed(i) * 180 / std::numbers::pi << std::endl;
    }

    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}
