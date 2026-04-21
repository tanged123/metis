/**
 * @file brachistochrone.cpp
 * @brief Brachistochrone ODE Example (OpenMDAO/Dymos style)
 *
 * Demonstrates a single ODE definition that works for both numeric and symbolic modes.
 *
 * ODE Formulation:
 *   xdot = v * sin(θ)      [horizontal velocity]
 *   ydot = -v * cos(θ)     [vertical velocity, y positive down]
 *   vdot = g * cos(θ)      [acceleration along wire]
 *
 * This example shows:
 * 1. Single templated ODE for numeric AND symbolic execution
 * 2. Numeric simulation with solve_ivp
 * 3. Symbolic gradients computed automatically
 * 4. No direct CasADi or Eigen namespace usage
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

// Physical constants
constexpr double g = 9.80665; // Standard gravity [m/s²]

// ============================================================================
// Unified ODE Definition (works for BOTH numeric and symbolic)
// ============================================================================

/**
 * @brief Brachistochrone ODE - single definition for dual-backend execution
 *
 * State vector: [x, y, v]
 * Control input: theta (wire angle from vertical)
 *
 * @tparam Scalar Type (double for numeric, SymbolicScalar for symbolic)
 */
template <typename Scalar>
metis::MetisVector<Scalar> brachistochrone_ode(const metis::MetisVector<Scalar> &state,
                                               const Scalar &theta) {
    Scalar v = state(2);

    // Use metis:: math functions for dual-backend support
    Scalar cos_theta = metis::cos(theta);
    Scalar sin_theta = metis::sin(theta);

    metis::MetisVector<Scalar> dydt(3);
    dydt << v * sin_theta, // xdot = v * sin(θ)
        -v * cos_theta,    // ydot = -v * cos(θ)
        g * cos_theta;     // vdot = g * cos(θ)

    return dydt;
}

// ============================================================================
// Main: Demonstrates both numeric and symbolic execution paths
// ============================================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Brachistochrone Problem - Unified ODE Demo                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\nODE Formulation (single definition, dual execution):" << std::endl;
    std::cout << "  xdot = v * sin(θ)" << std::endl;
    std::cout << "  ydot = -v * cos(θ)" << std::endl;
    std::cout << "  vdot = g * cos(θ)" << std::endl;

    // =========================================================================
    // PART 1: Numeric Mode
    // =========================================================================
    std::cout << "\n--- Numeric Mode ---" << std::endl;

    metis::NumericVector state(3);
    state << 0.5, 0.3, 5.0;  // x=0.5m, y=0.3m, v=5m/s
    double theta = M_PI / 4; // 45 degrees

    auto dydt = brachistochrone_ode(state, theta);

    std::cout << "State [x, y, v] = [" << state.transpose() << "]" << std::endl;
    std::cout << "θ = " << theta * 180 / M_PI << "°" << std::endl;
    std::cout << "  xdot = " << dydt(0) << " m/s" << std::endl;
    std::cout << "  ydot = " << dydt(1) << " m/s" << std::endl;
    std::cout << "  vdot = " << dydt(2) << " m/s²" << std::endl;

    // =========================================================================
    // PART 2: Symbolic Mode (same ODE function!)
    // =========================================================================
    std::cout << "\n--- Symbolic Mode ---" << std::endl;

    // sym_vec_pair gives us both:
    //   state_sym (SymbolicVector) - works with templated ODE
    //   state_mx  (raw MX)         - required for jacobian/Function (CasADi needs original symbol)
    auto [state_sym, state_mx] = metis::sym_vec_pair("state", 3);
    auto theta_sym = metis::sym("theta");

    // Call the SAME ODE function with symbolic types!
    auto dydt_sym = brachistochrone_ode(state_sym, theta_sym);

    std::cout << "Symbolic state derivatives:" << std::endl;
    std::cout << "  xdot = " << dydt_sym(0) << std::endl;
    std::cout << "  ydot = " << dydt_sym(1) << std::endl;
    std::cout << "  vdot = " << dydt_sym(2) << std::endl;

    // =========================================================================
    // PART 3: Automatic Jacobian Computation
    // =========================================================================
    std::cout << "\n--- Automatic Jacobians (no manual partials!) ---" << std::endl;

    // Jacobian: use state_mx (original symbol) for inputs, to_mx for outputs
    auto jac = metis::jacobian({metis::to_mx(dydt_sym)}, {state_mx, theta_sym});

    std::cout << "Jacobian d[xdot,ydot,vdot]/d[x,y,v,θ]:" << std::endl;
    std::cout << jac << std::endl;

    // Create function to evaluate Jacobian numerically
    metis::Function jac_fn({state_mx, theta_sym}, {jac});

    // Evaluate at our test point
    auto jac_numeric = jac_fn.eval(state, theta);

    std::cout << "\nJacobian evaluated at state=" << state.transpose()
              << ", θ=" << theta * 180 / M_PI << "°:" << std::endl;
    std::cout << jac_numeric << std::endl;

    // Verify against analytical values
    double v_val = state(2);
    double c = metis::cos(theta);
    double s = metis::sin(theta);
    std::cout << "\nPartials verification (column 3 = d/dv, column 4 = d/dθ):" << std::endl;
    std::cout << "  ∂xdot/∂v = sin(θ) = " << s << std::endl;
    std::cout << "  ∂xdot/∂θ = v·cos(θ) = " << v_val * c << std::endl;
    std::cout << "  ∂ydot/∂v = -cos(θ) = " << -c << std::endl;
    std::cout << "  ∂ydot/∂θ = v·sin(θ) = " << v_val * s << std::endl;
    std::cout << "  ∂vdot/∂θ = -g·sin(θ) = " << -g * s << std::endl;

    // =========================================================================
    // PART 4: ODE Integration with Time-Varying Control
    // =========================================================================
    std::cout << "\n--- ODE Integration (solve_ivp) ---" << std::endl;

    // Problem: Bead slides from (x=0, y=10) towards (x=10, y=5)
    // Drop of 5 meters, horizontal distance of 10 meters
    std::cout << "Problem: Start at (0, 10), target (10, 5)" << std::endl;
    std::cout << "Drop: 5m, Horizontal: 10m" << std::endl;

    double theta_start = 0.3;           // Start fairly steep (closer to vertical)
    double theta_end = M_PI / 2 - 0.05; // End nearly horizontal
    double T_total = 2.0;               // Longer time for the larger distance

    // Initial state: x=0, y=10, v=0.01 (small initial velocity to avoid singularity)
    metis::NumericVector y0(3);
    y0 << 0.0, 10.0, 0.01;

    // ODE wrapper for solve_ivp (includes time-varying theta)
    auto ode_with_control = [=](double t, const metis::NumericVector &y) {
        double theta_t = theta_start + (theta_end - theta_start) * t / T_total;
        return brachistochrone_ode(y, theta_t);
    };

    auto sol = metis::solve_ivp(ode_with_control, {0.0, T_total}, y0, 100);

    std::cout << "\nTrajectory with θ(t) from " << theta_start * 180 / M_PI << "° to "
              << theta_end * 180 / M_PI << "°:" << std::endl;
    std::cout << std::setw(8) << "t" << std::setw(10) << "x" << std::setw(10) << "y"
              << std::setw(10) << "v" << std::endl;
    std::cout << std::string(38, '-') << std::endl;

    for (int i = 0; i < sol.y.cols(); i += 10) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(8) << sol.t(i) << std::setw(10) << sol.y(0, i) << std::setw(10)
                  << sol.y(1, i) << std::setw(10) << sol.y(2, i) << std::endl;
    }

    std::cout << "\nFinal: x=" << sol.y(0, sol.y.cols() - 1)
              << " m, y=" << sol.y(1, sol.y.cols() - 1) << " m, v=" << sol.y(2, sol.y.cols() - 1)
              << " m/s" << std::endl;

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "✓ Single ODE definition used for BOTH numeric AND symbolic" << std::endl;
    std::cout << "✓ Automatic Jacobian computation (no manual partials)" << std::endl;
    std::cout << "✓ ODE integration with solve_ivp" << std::endl;
    std::cout << "✓ No direct CasADi or Eigen namespace usage" << std::endl;

    return 0;
}
