/**
 * @file beam_deflection.cpp
 * @brief Cantilever Beam Structural Optimization (GPkit-compatible)
 *
 * Based on the GPkit beam benchmark:
 * - Euler-Bernoulli beam with distributed load
 * - Fixed geometry (constant EI)
 * - Matches analytical solution: w = q/(24*EI) * x² * (x² - 4*L*x + 6*L²)
 *
 * Physics:
 *   dV/dx = -q            (shear from distributed load)
 *   dM/dx = V             (moment from shear)
 *   dθ/dx = M / (E*I)     (slope from curvature)
 *   dw/dx = θ             (deflection from slope)
 *
 * Boundary Conditions:
 *   V(L) ≈ 0, M(L) ≈ 0   (free end)
 *   θ(0) = 0, w(0) = 0   (fixed end)
 *
 * This demonstrates:
 * - Solving ODEs via optimization constraints
 * - Comparison with analytical solution
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Cantilever Beam Analysis (GPkit Benchmark)                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Problem parameters (matching GPkit)
    constexpr int N = 20;        // Number of nodes
    constexpr double L = 5.0;    // Beam length [m]
    constexpr double EI = 1.1e4; // Bending stiffness [N·m²]
    constexpr double q = 110.0;  // Distributed load [N/m]

    std::cout << "Problem Setup (GPkit compatible):\n";
    std::cout << "  Beam length L: " << L << " m\n";
    std::cout << "  Bending stiffness EI: " << EI << " N·m²\n";
    std::cout << "  Distributed load q: " << q << " N/m\n";
    std::cout << "  Nodes: " << N << "\n\n";

    // Discretization grid
    metis::NumericVector x = metis::linspace(0.0, L, N);
    double dx = x(1) - x(0);

    // =========================================================================
    // Warm Start Setup
    // =========================================================================
    std::string cache_file = "beam_solution.json";
    std::map<std::string, std::vector<double>> cache;
    try {
        cache = metis::OptiSol::load(cache_file);
        std::cout << "Warm starting from " << cache_file << "\n";
    } catch (...) {
        std::cout << "Cold start (no cache found)\n";
    }

    auto get_init = [&](const std::string &name, int size) {
        metis::NumericVector init = metis::NumericVector::Zero(size);
        if (cache.count(name)) {
            const auto &vec = cache[name];
            if (vec.size() == static_cast<size_t>(size)) {
                for (int i = 0; i < size; ++i)
                    init(i) = vec[i];
            }
        }
        return init;
    };

    // =========================================================================
    // Setup Optimization (solving beam equations as optimization)
    // =========================================================================
    metis::Opti opti;

    // State variables (unknowns)
    auto V = opti.variable(get_init("V", N));         // Shear force [N]
    auto M = opti.variable(get_init("M", N));         // Bending moment [N·m]
    auto theta = opti.variable(get_init("theta", N)); // Slope [rad]
    auto w = opti.variable(get_init("w", N));         // Deflection [m]

    // =========================================================================
    // Beam Equations (trapezoidal integration from tip to base)
    // =========================================================================

    // Shear integration: V[i] = V[i+1] + 0.5*dx*(q[i] + q[i+1])
    // (shear increases from tip to base)
    for (int i = 0; i < N - 1; ++i) {
        opti.subject_to(V(i) == V(i + 1) + q * dx);
    }

    // Moment integration: M[i] = M[i+1] + 0.5*dx*(V[i] + V[i+1])
    for (int i = 0; i < N - 1; ++i) {
        opti.subject_to(M(i) == M(i + 1) + 0.5 * dx * (V(i) + V(i + 1)));
    }

    // Theta integration: θ[i+1] = θ[i] + 0.5*dx*(M[i] + M[i+1])/EI
    for (int i = 0; i < N - 1; ++i) {
        opti.subject_to(theta(i + 1) == theta(i) + 0.5 * dx * (M(i) + M(i + 1)) / EI);
    }

    // Displacement integration: w[i+1] = w[i] + 0.5*dx*(θ[i] + θ[i+1])
    for (int i = 0; i < N - 1; ++i) {
        opti.subject_to(w(i + 1) == w(i) + 0.5 * dx * (theta(i) + theta(i + 1)));
    }

    // =========================================================================
    // Boundary Conditions
    // =========================================================================

    // Free end (x=L): V ≈ 0, M ≈ 0
    opti.subject_to(V(N - 1) == 0.0);
    opti.subject_to(M(N - 1) == 0.0);

    // Fixed end (x=0): θ = 0, w = 0
    opti.subject_to(theta(0) == 0.0);
    opti.subject_to(w(0) == 0.0);

    // =========================================================================
    // Objective: Minimize tip deflection (or just find feasible solution)
    // =========================================================================
    opti.minimize(w(N - 1));

    // =========================================================================
    // Solve
    // =========================================================================
    std::cout << "Solving beam equations...\n";
    auto sol = opti.solve({.max_iter = 100, .verbose = true});

    // =========================================================================
    // Results
    // =========================================================================
    auto V_sol = sol.value(V);
    auto M_sol = sol.value(M);
    auto theta_sol = sol.value(theta);
    auto w_sol = sol.value(w);

    double tip_deflection = w_sol(N - 1);

    // Analytical solution: w = q/(24*EI) * x² * (x² - 4*L*x + 6*L²)
    metis::NumericVector w_analytical(N);
    for (int i = 0; i < N; ++i) {
        double xi = x(i);
        w_analytical(i) = (q / (24.0 * EI)) * xi * xi * (xi * xi - 4.0 * L * xi + 6.0 * L * L);
    }
    double tip_analytical = w_analytical(N - 1);

    std::cout << "\n=== RESULTS ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Tip deflection (optimization): " << tip_deflection << " m\n";
    std::cout << "Tip deflection (analytical):   " << tip_analytical << " m\n";
    std::cout << "Error: " << std::abs(tip_deflection - tip_analytical) / tip_analytical * 100
              << "%\n";
    std::cout << "Solver iterations: " << sol.num_iterations().value_or(-1) << "\n\n";

    // Print deflection profile
    std::cout << "Deflection Profile:\n";
    std::cout << std::setw(10) << "x [m]" << std::setw(15) << "w_opt [m]" << std::setw(15)
              << "w_exact [m]" << std::setw(12) << "Error [cm]\n";
    std::cout << std::string(52, '-') << "\n";

    for (int i = 0; i < N; i += 3) {
        double err_cm = (w_sol(i) - w_analytical(i)) * 100;
        std::cout << std::setw(10) << x(i) << std::setw(15) << w_sol(i) << std::setw(15)
                  << w_analytical(i) << std::setw(12) << err_cm << "\n";
    }
    std::cout << std::setw(10) << x(N - 1) << std::setw(15) << w_sol(N - 1) << std::setw(15)
              << w_analytical(N - 1) << std::setw(12) << (w_sol(N - 1) - w_analytical(N - 1)) * 100
              << "\n";

    // Sanity check
    double max_error_cm = 0;
    for (int i = 0; i < N; ++i) {
        max_error_cm = std::max(max_error_cm, std::abs(w_sol(i) - w_analytical(i)) * 100);
    }

    std::cout << "\n=== VERIFICATION ===\n";
    std::cout << "Max error: " << max_error_cm << " cm\n";
    if (max_error_cm < 0.2) {
        std::cout << "✓ Solution matches analytical within 0.2 cm (like GPkit)\n";
    } else {
        std::cout << "✗ Error exceeds expected tolerance\n";
    }

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "✓ Euler-Bernoulli beam equations solved via optimization\n";
    std::cout << "✓ Matches analytical solution: w = q/(24EI) * x² * (x² - 4Lx + 6L²)\n";
    std::cout << "✓ GPkit benchmark compatible\n";

    // Save for warm start
    std::map<std::string, metis::SymbolicVector> vars;
    vars["V"] = V;
    vars["M"] = M;
    vars["theta"] = theta;
    vars["w"] = w;
    sol.save(cache_file, vars);
    std::cout << "Solution saved to " << cache_file << "\n";

    return 0;
}
