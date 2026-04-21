#include <iostream>
#include <metis/metis.hpp>

/**
 * @brief Demonstrate complex branching logic in Metis
 *
 * Standard C++ if/else doesn't work with symbolic types because conditions
 * can't be evaluated at graph-building time. Instead, we use metis::where()
 * which creates computational graphs that branch at runtime.
 */

// Example 1: Simple If-Else → metis::where()
template <typename Scalar> Scalar absolute_value(const Scalar &x) {
    // C++ way (DOESN'T work with symbolic):
    // if (x < 0) return -x;
    // else return x;

    // Metis way (works with both numeric and symbolic):
    return metis::where(x < 0.0, -x, x);
}

// Example 2: If-Else-If-Else → Nested where()
template <typename Scalar> Scalar piecewise_function(const Scalar &x) {
    // Compute: f(x) = { -1  if x < -1
    //                 {  x  if -1 <= x <= 1
    //                 {  1  if x > 1

    // Traditional C++ (DOESN'T work):
    // if (x < -1.0) return -1.0;
    // else if (x > 1.0) return 1.0;
    // else return x;

    // Metis way: chain where() calls
    return metis::where(x < -1.0, Scalar(-1.0), metis::where(x > 1.0, Scalar(1.0), x));
}

// Example 3: Switch-Case Logic → select() function (CLEAN!)
template <typename Scalar> Scalar flight_regime(const Scalar &mach) {
    // Determine drag coefficient based on Mach number
    // Mach < 0.3: incompressible (Cd = 0.02)
    // 0.3 <= Mach < 0.8: subsonic (Cd = 0.025)
    // 0.8 <= Mach < 1.2: transonic (Cd = 0.05)
    // Mach >= 1.2: supersonic (Cd = 0.03)
    // Much cleaner than nested where() calls!

    return metis::select({mach < 0.3, mach < 0.8, mach < 1.2},
                         {Scalar(0.02), Scalar(0.025), Scalar(0.05)},
                         Scalar(0.03)); // default: supersonic
}

// Example 3b: Old nested way (for comparison)
template <typename Scalar> Scalar flight_regime_nested(const Scalar &mach) {
    // Same logic, but harder to read with nested where()
    return metis::where(mach < 0.3, Scalar(0.02),
                        metis::where(mach < 0.8, Scalar(0.025),
                                     metis::where(mach < 1.2, Scalar(0.05), Scalar(0.03))));
}

// Example 4: Complex Multi-Step Branch Logic
// Helper functions for complex calculations
template <typename Scalar> Scalar turbulent_drag(const Scalar &reynolds, const Scalar &velocity) {
    // Multi-step calculation for turbulent flow
    auto cf = 0.074 / metis::pow(reynolds, 0.2);                       // Friction coefficient
    auto correction = 1.0 + 0.144 * metis::pow(velocity / 343.0, 2.0); // Compressibility
    return cf * correction;
}

template <typename Scalar> Scalar laminar_drag(const Scalar &reynolds) {
    // Simpler calculation for laminar flow
    return 1.328 / metis::sqrt(reynolds);
}

// Main function using helper functions in branches
template <typename Scalar>
Scalar skin_friction_coefficient(const Scalar &reynolds, const Scalar &velocity) {
    // Branching with complex multi-step logic in each branch
    Scalar Re_transition = 5e5;

    return metis::where(reynolds > Re_transition,
                        turbulent_drag(reynolds, velocity), // Complex calculation
                        laminar_drag(reynolds));            // Simpler calculation

    // Both branches are ALWAYS evaluated (computational graph requirement)
    // The condition just selects which result to use
}

// Example 4: Complex Physics - Atmosphere Model
template <typename Scalar> Scalar atmospheric_temperature(const Scalar &altitude_m) {
    // ISA (International Standard Atmosphere) temperature model
    // Troposphere (0-11km): T = T0 - L*h  (linear decrease)
    // Stratosphere (11-20km): T = T_tropopause (constant)

    Scalar T0 = 288.15; // Sea level temp (K)
    Scalar L = 0.0065;  // Lapse rate (K/m)
    Scalar h_tropopause = 11000.0;
    Scalar T_tropopause = T0 - L * h_tropopause;

    // Branch based on altitude
    return metis::where(altitude_m < h_tropopause,
                        T0 - L * altitude_m, // Troposphere
                        T_tropopause);       // Stratosphere
}

// Example 5: Multi-variable conditional - Aerodynamic stall
template <typename Scalar>
Scalar lift_coefficient(const Scalar &alpha_deg, const Scalar &reynolds) {
    // Lift coefficient with stall behavior
    // Cl = a * alpha  (linear region, |alpha| < 15°)
    // Cl = Cl_max * sign(alpha)  (stalled, |alpha| >= 15°)

    Scalar a = 0.1; // Lift slope (per degree)
    Scalar alpha_stall = 15.0;
    Scalar Cl_max = 1.5;

    Scalar alpha_abs = metis::where(alpha_deg < 0.0, -alpha_deg, alpha_deg);
    Scalar sign_alpha = metis::where(alpha_deg < 0.0, Scalar(-1.0), Scalar(1.0));

    // Reynolds number affects Cl_max
    Scalar Cl_max_corrected = Cl_max * metis::where(reynolds < 1e5,
                                                    Scalar(0.8), // Low Re penalty
                                                    Scalar(1.0));

    return metis::where(alpha_abs < alpha_stall,
                        a * alpha_deg,                  // Linear region
                        Cl_max_corrected * sign_alpha); // Stalled
}

// Example 6: Combining logical operators
template <typename Scalar>
Scalar safe_division(const Scalar &numerator, const Scalar &denominator) {
    // Avoid division by zero with epsilon threshold
    Scalar epsilon = 1e-10;

    // Check if denominator is "close to zero" using absolute value
    Scalar denom_abs = metis::where(denominator < 0.0, -denominator, denominator);
    Scalar is_small = denom_abs < epsilon;

    return metis::where(is_small,
                        Scalar(0.0),              // Return 0 if denom ≈ 0
                        numerator / denominator); // Safe division
}

int main() {
    std::cout << "=== Complex Branching Logic Examples ===\n\n";

    // Test 1: Absolute Value
    std::cout << "1. Absolute Value:\n";
    std::cout << "   |3.5| = " << absolute_value(3.5) << "\n";
    std::cout << "   |-2.7| = " << absolute_value(-2.7) << "\n";

    // Test 2: Piecewise Function
    std::cout << "\n2. Piecewise Clipping:\n";
    std::cout << "   f(-2.0) = " << piecewise_function(-2.0) << " (should be -1)\n";
    std::cout << "   f(0.5) = " << piecewise_function(0.5) << " (should be 0.5)\n";
    std::cout << "   f(3.0) = " << piecewise_function(3.0) << " (should be 1)\n";

    // Test 3: Flight Regime (with select!)
    std::cout << "\n3. Flight Regime Drag (using select):\n";
    std::cout << "   Mach 0.2: Cd = " << flight_regime(0.2) << "\n";
    std::cout << "   Mach 0.5: Cd = " << flight_regime(0.5) << "\n";
    std::cout << "   Mach 0.95: Cd = " << flight_regime(0.95) << "\n";
    std::cout << "   Mach 2.0: Cd = " << flight_regime(2.0) << "\n";

    std::cout << "   ✅ select() is much cleaner than nested where()!\n";

    // Test 4: Complex Branch Logic
    std::cout << "\n4. Skin Friction (Complex Multi-Step Branches):\n";
    std::cout << "   Laminar (Re=1e5): Cf = " << skin_friction_coefficient(1e5, 50.0) << "\n";
    std::cout << "   Turbulent (Re=1e6): Cf = " << skin_friction_coefficient(1e6, 50.0) << "\n";
    std::cout << "   💡 Each branch can have multiple calculation steps!\n";

    // Test 5: Atmospheric Model
    std::cout << "\n5. ISA Temperature:\n";
    std::cout << "   Sea level: T = " << atmospheric_temperature(0.0) << " K\n";
    std::cout << "   5000m: T = " << atmospheric_temperature(5000.0) << " K\n";
    std::cout << "   11000m: T = " << atmospheric_temperature(11000.0) << " K\n";
    std::cout << "   15000m: T = " << atmospheric_temperature(15000.0) << " K\n";

    // Test 6: Symbolic Mode - Generate derivatives
    std::cout << "\n6. Symbolic Mode - Automatic Differentiation:\n";

    auto alpha_sym = metis::sym("alpha");
    auto reynolds_sym = metis::sym("reynolds");

    auto Cl_expr = lift_coefficient(alpha_sym, metis::SymbolicScalar(2e5)); // Fix reynolds

    // Compute dCl/dalpha (lift curve slope with stall)
    auto dCl_dalpha = metis::jacobian({Cl_expr}, {alpha_sym});
    metis::Function Cl_slope({alpha_sym}, {dCl_dalpha});

    std::cout << "   Lift curve slope at alpha=5°: " << Cl_slope.eval(5.0)(0, 0) << "\n";
    std::cout << "   Lift curve slope at alpha=20°: " << Cl_slope.eval(20.0)(0, 0)
              << " (stalled!)\n";

    // Test 7: Safe Division
    std::cout << "\n7. Safe Division:\n";
    std::cout << "   10 / 2 = " << safe_division(10.0, 2.0) << "\n";
    std::cout << "   5 / 1e-15 = " << safe_division(5.0, 1e-15) << " (protected!)\n";

    std::cout << "\n✅ All branching examples completed!\n";
    std::cout
        << "💡 Key takeaway: Use metis::where() instead of if/else for symbolic compatibility\n";

    return 0;
}
