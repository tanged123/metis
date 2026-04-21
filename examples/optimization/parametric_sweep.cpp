/**
 * @file parametric_sweep.cpp
 * @brief Example: Parametric Sweep using solve_sweep()
 *
 * Demonstrates how to efficiently solve an optimization problem
 * across a range of parameter values with automatic warm-starting.
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

int main() {
    std::cout << "========================================\n";
    std::cout << "Parametric Sweep Example\n";
    std::cout << "========================================\n\n";

    // =========================================================================
    // Problem: Optimal velocity for minimum flight time
    //
    // Simple model: minimize time = distance / velocity
    // Subject to: power = drag ≤ power_available
    //
    // Sweep over air density (altitude effect)
    // =========================================================================

    metis::Opti opti;

    // Parameters (can be swept)
    auto rho = opti.parameter(1.225); // Air density [kg/m³]

    // Constants
    const double distance = 100e3;     // 100 km
    const double S = 16.0;             // Wing area [m²]
    const double Cd0 = 0.02;           // Parasitic drag coeff
    const double power_avail = 150000; // Available power [W]

    // Decision variable
    auto V = opti.variable(50.0); // Velocity [m/s]

    // Objective: minimize time
    auto time = distance / V;
    opti.minimize(time);

    // Constraints
    auto drag = 0.5 * rho * V * V * S * Cd0;
    auto power_req = drag * V;
    opti.subject_to(power_req <= power_avail);
    opti.subject_to(V >= 10.0); // Min velocity

    // =========================================================================
    // Parametric Sweep: Vary air density (simulating altitude)
    // =========================================================================
    std::vector<double> rho_values;
    for (double r = 1.225; r >= 0.4; r -= 0.1) {
        rho_values.push_back(r);
    }

    std::cout << "Sweeping air density from " << rho_values.front() << " to " << rho_values.back()
              << " kg/m³\n\n";

    auto result = opti.solve_sweep(rho, rho_values, {.verbose = false});

    // =========================================================================
    // Results
    // =========================================================================
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(12) << "rho [kg/m³]" << std::setw(12) << "V* [m/s]" << std::setw(12)
              << "Time [min]" << std::setw(10) << "Iters\n";
    std::cout << std::string(46, '-') << "\n";

    for (size_t i = 0; i < result.size(); ++i) {
        if (!result.converged[i]) {
            std::cout << std::setw(12) << result.param_values[i] << "  (failed)\n";
            continue;
        }
        double V_opt = result.solutions[i]->value(V);
        double time_min = (distance / V_opt) / 60.0;

        std::cout << std::setw(12) << result.param_values[i] << std::setw(12) << V_opt
                  << std::setw(12) << time_min << std::setw(10) << result.iterations[i] << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Sweep Complete! (" << result.size() << " solves)\n";
    std::cout << "========================================\n";

    return 0;
}
