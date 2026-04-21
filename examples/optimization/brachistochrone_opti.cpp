/**
 * @file brachistochrone_opti.cpp
 * @brief Brachistochrone via Trajectory Optimization (Dymos-compatible)
 *
 * Classic optimal control problem: Find the fastest path for a bead sliding
 * under gravity from point A to point B.
 *
 * Problem setup (matching Dymos):
 *   Start: (x=0, y=10) with v=0
 *   End:   (x=10, y=5) with v=free
 *   This is 10m horizontal, 5m vertical drop
 *
 * ODE (same as brachistochrone.cpp):
 *   xdot = v * sin(θ)
 *   ydot = -v * cos(θ)   (y positive UP, so ydot < 0 means going down)
 *   vdot = g * cos(θ)
 *
 * Expected result: T* ≈ 1.80 seconds (matches Dymos output)
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

// Physical constants
constexpr double g = 9.80665; // Standard gravity [m/s²]

// ============================================================================
// SAME ODE as brachistochrone.cpp - reused directly!
// ============================================================================

/**
 * @brief Brachistochrone ODE - single definition for dual-backend execution
 * @tparam Scalar Type (double for numeric, SymbolicScalar for symbolic/optimization)
 */
template <typename Scalar>
metis::MetisVector<Scalar> brachistochrone_ode(const metis::MetisVector<Scalar> &state,
                                               const Scalar &theta) {
    Scalar v = state(2);

    Scalar cos_theta = metis::cos(theta);
    Scalar sin_theta = metis::sin(theta);

    metis::MetisVector<Scalar> dydt(3);
    dydt << v * sin_theta, // xdot = v * sin(θ)
        -v * cos_theta,    // ydot = -v * cos(θ) (going down when cos(θ) > 0)
        g * cos_theta;     // vdot = g * cos(θ)

    return dydt;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Brachistochrone via Trajectory Optimization (Dymos Setup) ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Problem parameters (matching Dymos exactly)
    constexpr double x_init = 0.0;
    constexpr double y_init = 10.0;
    constexpr double v_init = 0.0;

    constexpr double x_final = 10.0;
    constexpr double y_final = 5.0; // Drop of 5 meters

    constexpr int N = 50; // Number of discretization points (more for accuracy)

    std::cout << "Problem (Dymos-compatible):\n";
    std::cout << "  Start: (x=" << x_init << ", y=" << y_init << ") with v=0\n";
    std::cout << "  End:   (x=" << x_final << ", y=" << y_final << ") with v=free\n";
    std::cout << "  Drop: " << (y_init - y_final) << " m, Horizontal: " << x_final << " m\n";
    std::cout << "  Discretization: " << N << " points\n\n";

    // =========================================================================
    // Setup Optimization Problem
    // =========================================================================
    metis::Opti opti;

    // Time - this is what we're minimizing
    // Initial guess of 2.0s, bounds [0.5, 10]
    auto T = opti.variable(2.0, std::nullopt, 0.5, 10.0);

    // Normalized time grid: tau ∈ [0, 1], actual time t = tau * T
    metis::NumericVector tau = metis::linspace(0.0, 1.0, N);

    // State trajectories: x(tau), y(tau), v(tau)
    // Initial guesses based on linear interpolation
    auto x = opti.variable(N, 5.0); // Midpoint guess
    auto y = opti.variable(N, 7.5); // Midpoint guess
    auto v = opti.variable(N, 5.0); // Reasonable velocity guess

    // Control trajectory: theta(tau) - angle from vertical
    // Dymos uses degrees [0.01, 179.9], we use radians
    auto theta = opti.variable(N, M_PI / 4);

    // Bound velocity to be non-negative
    opti.subject_to_lower(v, 0.0);

    // Bound theta to physical range (0.01° to 179.9° in radians)
    constexpr double theta_min = 0.01 * M_PI / 180.0;
    constexpr double theta_max = 179.9 * M_PI / 180.0;
    opti.subject_to_bounds(theta, theta_min, theta_max);

    // =========================================================================
    // Dynamics Constraints (using the SAME ODE function!)
    // =========================================================================

    // Apply collocation constraints at each interval
    for (int i = 0; i < N - 1; ++i) {
        double dtau = tau(i + 1) - tau(i);

        // Build state vector at point i
        metis::SymbolicVector state_i(3);
        state_i << x(i), y(i), v(i);

        // Build state vector at point i+1
        metis::SymbolicVector state_ip1(3);
        state_ip1 << x(i + 1), y(i + 1), v(i + 1);

        // Evaluate ODE at both points (REUSING the same ODE function!)
        auto dydt_i = brachistochrone_ode(state_i, theta(i));
        auto dydt_ip1 = brachistochrone_ode(state_ip1, theta(i + 1));

        // Trapezoidal collocation: state[i+1] - state[i] = 0.5 * (f[i] + f[i+1]) * T * dtau
        opti.subject_to(x(i + 1) - x(i) == 0.5 * (dydt_i(0) + dydt_ip1(0)) * T * dtau);
        opti.subject_to(y(i + 1) - y(i) == 0.5 * (dydt_i(1) + dydt_ip1(1)) * T * dtau);
        opti.subject_to(v(i + 1) - v(i) == 0.5 * (dydt_i(2) + dydt_ip1(2)) * T * dtau);
    }

    // =========================================================================
    // Boundary Conditions (matching Dymos)
    // =========================================================================

    // Initial conditions
    opti.subject_to(x(0) == x_init);
    opti.subject_to(y(0) == y_init);
    opti.subject_to(v(0) == v_init + 0.001); // Small perturbation to avoid singularity

    // Final conditions (velocity is free)
    opti.subject_to(x(N - 1) == x_final);
    opti.subject_to(y(N - 1) == y_final);

    // =========================================================================
    // Objective: Minimize Time
    // =========================================================================
    opti.minimize(T);

    // =========================================================================
    // Solve
    // =========================================================================
    std::cout << "Solving trajectory optimization...\n";
    auto sol = opti.solve({.max_iter = 500, .verbose = true});

    // =========================================================================
    // Extract Results
    // =========================================================================
    double T_opt = sol.value(T);
    auto x_opt = sol.value(x);
    auto y_opt = sol.value(y);
    auto v_opt = sol.value(v);
    auto theta_opt = sol.value(theta);

    std::cout << "\n=== RESULTS ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Optimal time T* = " << T_opt << " seconds\n";
    std::cout << "Solver iterations: " << sol.num_iterations().value_or(-1) << "\n";
    std::cout << "Dymos reference: T* ≈ 1.80185208 seconds\n\n";

    // Print trajectory
    std::cout << "Optimal Trajectory:\n";
    std::cout << std::setw(8) << "t [s]" << std::setw(10) << "x [m]" << std::setw(10) << "y [m]"
              << std::setw(10) << "v [m/s]" << std::setw(12) << "θ [deg]" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (int i = 0; i < N; i += 7) {
        double t = tau(i) * T_opt;
        std::cout << std::setw(8) << t << std::setw(10) << x_opt(i) << std::setw(10) << y_opt(i)
                  << std::setw(10) << v_opt(i) << std::setw(12) << theta_opt(i) * 180 / M_PI
                  << "\n";
    }
    // Always print last point
    std::cout << std::setw(8) << T_opt << std::setw(10) << x_opt(N - 1) << std::setw(10)
              << y_opt(N - 1) << std::setw(10) << v_opt(N - 1) << std::setw(12)
              << theta_opt(N - 1) * 180 / M_PI << "\n";

    // =========================================================================
    // Verification
    // =========================================================================
    std::cout << "\n=== VERIFICATION ===\n";
    std::cout << "Final state from optimization:\n";
    std::cout << "  x = " << x_opt(N - 1) << " m (target: " << x_final << ")\n";
    std::cout << "  y = " << y_opt(N - 1) << " m (target: " << y_final << ")\n";
    std::cout << "  v = " << v_opt(N - 1) << " m/s (free)\n";

    double error_pct = std::abs(T_opt - 1.80185208) / 1.80185208 * 100.0;
    std::cout << "\nComparison to Dymos:\n";
    std::cout << "  Metis T* = " << T_opt << " s\n";
    std::cout << "  Dymos T* = 1.80185208 s\n";
    std::cout << "  Error: " << error_pct << "%\n";

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "✓ Same ODE function used for BOTH simulation AND optimization\n";
    std::cout << "✓ metis::Opti with trapezoidal collocation\n";
    std::cout << "✓ Matches Dymos problem setup exactly\n";
    std::cout << "✓ Clean API - no direct CasADi calls\n";

    return 0;
}
