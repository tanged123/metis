/**
 * @file drag_optimization.cpp
 * @brief Aerodynamic Optimization: Maximum L/D and Minimum Drag
 *
 * Uses the SAME compute_drag function from drag_coefficient.cpp
 * to demonstrate optimization of aerodynamic quantities.
 *
 * ULTIMATE CLEAN PATTERN (C++20):
 * Use 'auto' for function parameters (Abbreviated Function Templates).
 * This allows mixing 'double' constants and 'SymbolicScalar' variables naturally.
 * The return type is deduced automatically.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

// ============================================================================
// SHARED PHYSICS FUNCTIONS (C++20 Auto Templates)
// ============================================================================

// Drag Polar: Cd = Cd0 + k * (Cl - Cl0)²
// Drag Force: D = 0.5 * rho * v² * S * Cd
// Using 'auto' allows the compiler to handle mixed inputs (double * Symbolic -> Symbolic)
auto compute_drag(auto rho, auto v, auto S, auto Cd0, auto k, auto Cl, auto Cl0) {
    auto q = 0.5 * rho * metis::pow(v, 2.0);
    auto Cd = Cd0 + k * metis::pow(Cl - Cl0, 2.0);
    return q * S * Cd;
}

// Lift Force: L = 0.5 * rho * v² * S * Cl
auto compute_lift(auto rho, auto v, auto S, auto Cl) {
    auto q = 0.5 * rho * metis::pow(v, 2.0);
    return q * S * Cl;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Aerodynamic Optimization Examples                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "Demonstrating: 'Write Physics Once, Use Everywhere' (C++20 Style)\n\n";

    // Aircraft parameters (constants as standard doubles)
    const double rho = 1.225;
    const double S = 16.0;
    const double Cd0 = 0.02;
    const double k = 0.04;
    const double Cl0 = 0.1;
    const double W = 10000.0;

    // =========================================================================
    // Problem 1: Find Cl for Maximum L/D
    // =========================================================================
    std::cout << "=== Problem 1: Maximum L/D Ratio ===\n";
    {
        metis::Opti opti;
        auto Cl = opti.variable(0.5);
        opti.subject_to_bounds(Cl, 0.1, 2.0);

        // CLEANEST: Just pass the variables!
        // compute_drag(double, double, double, double, double, Symbolic, double)
        // Works because double * Symbolic -> Symbolic via Metis operator overloads.
        auto D = compute_drag(rho, 100.0, S, Cd0, k, Cl, Cl0);
        auto L = compute_lift(rho, 100.0, S, Cl);

        opti.minimize(D / L);

        auto sol = opti.solve({.max_iter = 100, .verbose = false});

        double Cl_opt = sol.value(Cl);
        double Cl_analytical = std::sqrt((Cd0 + k * Cl0 * Cl0) / k);

        std::cout << "  Cl* (Optimization): " << Cl_opt << "\n";
        std::cout << "  Cl* (Analytical):   " << Cl_analytical << "\n";
    }
    std::cout << "\n";

    // =========================================================================
    // Problem 2: Minimum Drag at Cruise (L = W)
    // =========================================================================
    std::cout << "=== Problem 2: Minimum Drag at Cruise (L = W) ===\n";
    {
        metis::Opti opti;
        auto V = opti.variable(50.0);
        auto Cl = opti.variable(0.5);

        opti.subject_to_bounds(V, 20.0, 150.0);
        opti.subject_to_bounds(Cl, 0.1, 1.8);

        // Mixing V (Symbolic) and numeric constants works seamlessly
        auto L = compute_lift(rho, V, S, Cl);
        auto D = compute_drag(rho, V, S, Cd0, k, Cl, Cl0);

        opti.subject_to(L == W);
        opti.minimize(D);

        auto sol = opti.solve({.max_iter = 100, .verbose = false});

        std::cout << "  V*  : " << sol.value(V) << " m/s\n";
        std::cout << "  Dmin: " << sol.value(D) << " N\n";
        std::cout << "  L   : " << sol.value(L) << " N (Target: " << W << ")\n";
    }
    std::cout << "\n";

    // =========================================================================
    // Problem 3: Minimum Power (Best Endurance)
    // =========================================================================
    std::cout << "=== Problem 3: Minimum Power (P = D*V) ===\n";
    {
        metis::Opti opti;
        auto V = opti.variable(40.0);
        auto Cl = opti.variable(0.8);

        opti.subject_to_bounds(V, 20.0, 150.0);
        opti.subject_to_bounds(Cl, 0.1, 1.8);

        auto L = compute_lift(rho, V, S, Cl);
        auto D = compute_drag(rho, V, S, Cd0, k, Cl, Cl0);

        opti.subject_to(L == W);
        opti.minimize(D * V);

        auto sol = opti.solve({.max_iter = 100, .verbose = false});

        std::cout << "  V* (Endurance): " << sol.value(V) << " m/s\n";
        std::cout << "  Power Min:      " << sol.value(D * V) / 1000.0 << " kW\n";
    }

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "✓ Used 'auto' parameters for Physics Functions\n";
    std::cout << "✓ Mixed doubles and SymbolicScalars automatically handled\n";
    std::cout << "✓ No helper lambda or explicit template types needed!\n";

    return 0;
}
