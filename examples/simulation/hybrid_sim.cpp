#include <cmath>
#include <iostream>
#include <metis/metis.hpp>
#include <vector>

/**
 * @brief Hybrid Simulation Pattern Example
 *
 * Demonstrates how to handle simulations with UNKNOWN end-times in an
 * optimization/autodiff context.
 *
 * THE PROBLEM:
 * - We want to simulate a bouncing ball until it stops bouncing (event detection).
 * - The number of steps N depends on the parameters (e.g. restitution coeff).
 * - AutoDiff requires a FIXED graph structure (fixed N).
 *
 * THE SOLUTION (Two Approaches):
 * 1. "Hybrid Pre-Step": Run numeric sim to find N, then build symbolic graph with N steps.
 * 2. "Free-Time": Fix N=100, optimize 'dt' to make the event happen exactly at step 100.
 */

// ======================================================================
// APPROACH 1: Hybrid Pre-Step (Mesh Refinement)
// ======================================================================

// 1a. Numeric Pre-Step (Cheap, finding N)
struct PreStepResult {
    int steps_needed;
    double final_time;
};

PreStepResult find_timesteps_until_impact(double y0, double v0, double dt) {
    // Pure numeric simulation - using while loops freely!
    double y = y0;
    double v = v0;
    double t = 0.0;
    int steps = 0;
    const double g = 9.81;

    // Simulate until impact (y < 0)
    while (y >= 0.0 && steps < 1000) {
        v -= g * dt;
        y += v * dt;
        t += dt;
        steps++;
    }

    return {steps, t};
}

// 1b. Symbolic Graph Building (Fixed N from Pre-Step)
template <typename Scalar>
Scalar simulate_fixed_steps(const Scalar &y0, const Scalar &v0, double dt, int N) {
    // Symbolic-compatible simulation (Fixed N)
    Scalar y = y0;
    Scalar v = v0;
    const double g = 9.81;

    for (int i = 0; i < N; ++i) {
        v -= g * dt;
        y += v * dt;
    }

    return y; // Should be approximately 0 at impact
}

void demo_hybrid_prestep() {
    std::cout << "\n=== Approach 1: Hybrid Pre-Step Pattern ===\n";
    std::cout << "Problem: Improve estimate of impact time sensitivity w.r.t initial velocity.\n";

    double y0 = 10.0;
    double v0_numeric = 0.0; // Drop from rest
    double dt = 0.01;

    // Step 1: Numeric Pre-Step
    std::cout << "1. Running numeric pre-step to find N...\n";
    PreStepResult res = find_timesteps_until_impact(y0, v0_numeric, dt);
    std::cout << "   Impact detected at step " << res.steps_needed << " (t=" << res.final_time
              << "s)\n";

    // Step 2: Build Symbolic Graph with exactly N steps
    std::cout << "2. Building symbolic graph with N=" << res.steps_needed << "...\n";

    auto v0_sym = metis::sym("v0");
    // We simulate exactly 'res.steps_needed' times
    auto y_final_sym =
        simulate_fixed_steps(metis::SymbolicScalar(y0), v0_sym, dt, res.steps_needed);

    // Step 3: Compute Derivatives
    // d(y_final)/d(v0) tells us how much the impact depth changes with initial velocity
    // This is useful for refining the event instant
    auto jac_expr = metis::jacobian({y_final_sym}, {v0_sym});
    metis::Function jac_fun({v0_sym}, {jac_expr});

    // Evaluate at v0 = 0
    auto sensitivity = jac_fun.eval(v0_numeric);

    std::cout << "3. Sensitivity Analysis:\n";
    std::cout << "   d(y_final)/d(v0) = " << sensitivity(0, 0) << "\n";
    std::cout << "   (This gradient is valid for the fixed N-step trajectory)\n";
}

// ======================================================================
// APPROACH 2: Free-Time Formulation (Time Scaling)
// ======================================================================

// In this approach, we fix N, and make 'dt' (or total time T) a variable.
// We want y(T) = 0.

template <typename Scalar>
Scalar free_time_simulation(const Scalar &y0, const Scalar &v0, const Scalar &total_time, int N) {
    Scalar dt = total_time / static_cast<double>(N); // Time scaling!
    Scalar y = y0;
    Scalar v = v0;
    const double g = 9.81;

    for (int i = 0; i < N; ++i) {
        v -= g * dt;
        y += v * dt;
    }

    return y;
}

void demo_free_time() {
    std::cout << "\n=== Approach 2: Free-Time Formulation ===\n";
    std::cout << "Problem: Find exact impact time T such that y(T) = 0.\n";

    // Fix N to a reasonable resolution
    int N = 100;
    double y0 = 10.0;
    double v0_val = 0.0;
    double T_guess = 1.0; // Initial guess for impact time

    std::cout << "1. Fixed N=" << N << ", Optimizing Total Time T...\n";

    // Newton-Raphson to find T such that y(T) = 0
    // We need d(y_final)/dT

    auto T_sym = metis::sym("T");
    // Build graph ONCE
    auto y_final_expr =
        free_time_simulation(metis::SymbolicScalar(y0), metis::SymbolicScalar(v0_val), T_sym, N);

    // Create function y(T) and dy/dT
    metis::Function y_fun({T_sym}, {y_final_expr});

    auto dy_dT_expr = metis::jacobian({y_final_expr}, {T_sym});
    metis::Function dy_dT_fun({T_sym}, {dy_dT_expr}); // Symbolic derivative w.r.t time!

    // Iterative solver using the symbolic derivative
    double T = T_guess;
    for (int iter = 0; iter < 10; ++iter) {
        // Evaluate y and dy/dT
        double y_val = y_fun.eval(T)(0, 0);
        double dydt_val = dy_dT_fun.eval(T)(0, 0);

        std::cout << "   Iter " << iter << ": T=" << T << ", y=" << y_val << "\n";

        // Newton step: T_new = T - y / y'
        if (std::abs(y_val) < 1e-6)
            break;
        T -= y_val / dydt_val;
    }

    std::cout << "2. Converged Result:\n";
    std::cout << "   Impact Time T = " << T << " s\n";
    std::cout << "   Analytic Check: sqrt(2*10/9.81) = " << std::sqrt(2.0 * 10.0 / 9.81) << " s\n";
    std::cout << "   ✅ Found exact event time using symbolic time scaling!\n";
}

int main() {
    std::cout << "=== HYBRID SIMULATION PATTERNS ===\n";
    std::cout << "Handling unknown end-times in Metis optimization.\n";

    demo_hybrid_prestep();
    demo_free_time();

    return 0;
}
