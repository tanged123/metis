#include <iostream>
#include <metis/metis.hpp>
#include <vector>

// Physics model: Drag Coefficient
// Cd = Cd0 + k * (Cl - Cl0)^2
// Drag = 0.5 * rho * v^2 * S * Cd
//
// C++20 "Abbreviated Function Template" (auto params)
// Allows implicit mixing of double constants and Symbolic variables!
auto compute_drag(auto rho, auto v, auto S, auto Cd0, auto k, auto Cl, auto Cl0) {
    auto q = 0.5 * rho * metis::pow(v, 2.0);
    auto Cd = Cd0 + k * metis::pow(Cl - Cl0, 2.0);
    return q * S * Cd;
}

int main() {
    // Numeric Mode
    double rho = 1.225;
    double v = 50.0;
    double S = 10.0;
    double Cd0 = 0.02;
    double k = 0.04;
    double Cl = 0.5;
    double Cl0 = 0.1;

    double drag_numeric = compute_drag(rho, v, S, Cd0, k, Cl, Cl0);
    std::cout << "Numeric Drag: " << drag_numeric << " N" << std::endl;

    // Symbolic Mode
    auto v_sym = metis::sym("v");
    auto Cl_sym = metis::sym("Cl");

    // Call with mixed types (double constants + Symbolic variables)
    // No explicit casts needed thanks to 'auto' params!
    auto drag_sym = compute_drag(rho, v_sym, S, Cd0, k, Cl_sym, Cl0);

    // Create function: f(v, Cl) -> drag
    metis::Function drag_fun({v_sym, Cl_sym}, {drag_sym});

    // Evaluate symbolic function at numeric point
    double drag_eval = drag_fun.eval(v, Cl)(0, 0);
    std::cout << "Symbolic Drag (evaluated): " << drag_eval << " N" << std::endl;

    // --- Automatic Differentiation ---
    std::cout << "\nComputing Jacobian (Automatic Differentiation)..." << std::endl;

    // Concatenate inputs into a single vector for Jacobian computation
    // J = d(drag)/d[v, Cl]
    metis::SymbolicScalar J_sym = metis::jacobian({drag_sym}, {v_sym, Cl_sym});

    // Create Jacobian function: f(v, Cl) -> [dDrag/dv, dDrag/dCl]
    metis::Function J_fun({v_sym, Cl_sym}, {J_sym});

    // Evaluate Jacobian at operating point
    auto J = J_fun.eval(v, Cl);
    std::cout << "Jacobian [dDrag/dv, dDrag/dCl]: " << J << std::endl;

    // Verification (Analytic/Numeric check)
    // Drag = 0.5 * rho * v^2 * S * (Cd0 + k * (Cl - Cl0)^2)
    // dDrag/dv = rho * v * S * Cd
    // dDrag/dCl = q * S * (2 * k * (Cl - Cl0))
    // Let's verify manually
    double Cd = Cd0 + k * std::pow(Cl - Cl0, 2.0);
    double q = 0.5 * rho * std::pow(v, 2.0);
    double dDrag_dv = rho * v * S * Cd;
    double dDrag_dCl = q * S * (2.0 * k * (Cl - Cl0));

    std::cout << "Analytic Check: [" << dDrag_dv << ", " << dDrag_dCl << "]" << std::endl;

    return 0;
}
