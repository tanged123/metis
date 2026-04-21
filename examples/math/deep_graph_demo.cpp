#include <iostream>
#include <metis/metis.hpp>

/**
 * @brief Deep Graph Visualization Demo
 *
 * Demonstrates the difference between shallow (MX-based) and deep (SX-based)
 * graph visualization. Deep graphs show all primitive operations by expanding
 * nested function calls.
 */

int main() {
    std::cout << "=== Deep Graph Visualization Demo ===\n\n";

    // ============================================================
    // Example 1: Simple expression - compare shallow vs deep
    // ============================================================
    std::cout << "1. Simple Expression: sin(x)^2 + cos(x)^2\n";

    auto x = metis::sym("x");
    auto expr1 = metis::sin(x) * metis::sin(x) + metis::cos(x) * metis::cos(x);

    // Shallow graph (MX-based) - existing function
    metis::export_graph_dot(expr1, "shallow_trig", "ShallowTrig");
    metis::export_graph_html(expr1, "shallow_trig", "ShallowTrig");
    std::cout << "   Shallow: shallow_trig.html\n";

    // Create a Function and export deep graph
    metis::Function f1("trig_identity", {x}, {expr1});
    metis::export_graph_deep(f1.casadi_function(), "deep_trig", metis::DeepGraphFormat::HTML,
                             "DeepTrig");
    metis::export_graph_deep(f1.casadi_function(), "deep_trig", metis::DeepGraphFormat::DOT,
                             "DeepTrig");
    std::cout << "   Deep:    deep_trig.html\n\n";

    // ============================================================
    // Example 2: Nested function calls - this is where deep shines
    // ============================================================
    std::cout << "2. Nested Functions: Demonstrates function expansion\n";

    auto y = metis::sym("y");
    auto z = metis::sym("z");

    // Inner function: computes magnitude
    auto magnitude = metis::sqrt(x * x + y * y + z * z);
    metis::Function mag_fn("magnitude", {x, y, z}, {magnitude});

    // Outer function: normalizes a vector using the inner magnitude function
    // Call the magnitude function symbolically
    auto mag_result = mag_fn(x, y, z);
    auto mag_call = mag_result[0](0, 0); // Get the scalar result

    auto nx = x / mag_call;
    auto ny = y / mag_call;
    auto nz = z / mag_call;

    // Create function that uses the nested call
    metis::Function normalize_fn("normalize", {x, y, z}, {nx, ny, nz});

    // Shallow graph - shows magnitude as opaque function call
    metis::export_graph_html(nx, "shallow_normalize", "ShallowNormalize");
    std::cout << "   Shallow: shallow_normalize.html (magnitude appears as single node)\n";

    // Deep graph - expands magnitude function to show sqrt, sq, add operations
    metis::export_graph_deep(normalize_fn.casadi_function(), "deep_normalize",
                             metis::DeepGraphFormat::HTML, "DeepNormalize");
    std::cout << "   Deep:    deep_normalize.html (shows sqrt, *, +, / operations)\n\n";

    // ============================================================
    // Example 3: Physics simulation - 2-body gravitational dynamics
    // ============================================================
    std::cout << "3. Two-Body Gravitational Dynamics\n";

    // State: position (x, y, z) and velocity (vx, vy, vz)
    auto px = metis::sym("px");
    auto py = metis::sym("py");
    auto pz = metis::sym("pz");
    auto vx = metis::sym("vx");
    auto vy = metis::sym("vy");
    auto vz = metis::sym("vz");

    // Gravitational parameter
    auto mu = metis::sym("mu");

    // Compute r^3 for gravitational acceleration
    auto r_squared = px * px + py * py + pz * pz;
    auto r = metis::sqrt(r_squared);
    auto r_cubed = r * r_squared;

    // Gravitational acceleration: a = -mu/r^3 * r_vec
    auto ax = -mu * px / r_cubed;
    auto ay = -mu * py / r_cubed;
    auto az = -mu * pz / r_cubed;

    // State derivative: [v, a]
    metis::Function dynamics("gravity_dynamics", {px, py, pz, vx, vy, vz, mu},
                             {vx, vy, vz, ax, ay, az});

    // Deep graph shows all the mathematical operations
    metis::export_graph_deep(dynamics.casadi_function(), "deep_gravity",
                             metis::DeepGraphFormat::HTML, "GravityDynamics");
    metis::export_graph_deep(dynamics.casadi_function(), "deep_gravity",
                             metis::DeepGraphFormat::DOT, "GravityDynamics");
    std::cout << "   Deep:    deep_gravity.html\n";
    std::cout << "   Shows: sqrt, sq (square), *, /, - operations\n\n";

    // ============================================================
    // Example 4: Jacobian of dynamics - automatic differentiation
    // ============================================================
    std::cout << "4. Jacobian of Gravitational Dynamics (Automatic Differentiation)\n";

    // Compute Jacobian of acceleration w.r.t. position using metis::jacobian
    // which takes vectors of symbolic scalars
    auto jacobian_a_r = metis::jacobian({ax, ay, az}, {px, py, pz});

    metis::Function jac_fn("gravity_jacobian", {px, py, pz, mu}, {jacobian_a_r});
    metis::export_graph_deep(jac_fn.casadi_function(), "deep_gravity_jacobian",
                             metis::DeepGraphFormat::HTML, "GravityJacobian");
    std::cout << "   Deep:    deep_gravity_jacobian.html\n";
    std::cout << "   Shows the full derivative computation graph\n\n";

    // ============================================================
    // Example 5: Control system - PID controller
    // ============================================================
    std::cout << "5. PID Controller\n";

    auto error = metis::sym("error");
    auto error_integral = metis::sym("error_int");
    auto error_derivative = metis::sym("error_dot");
    auto Kp = metis::sym("Kp");
    auto Ki = metis::sym("Ki");
    auto Kd = metis::sym("Kd");

    // PID output: u = Kp*e + Ki*integral(e) + Kd*de/dt
    auto u = Kp * error + Ki * error_integral + Kd * error_derivative;

    // With saturation (smooth clamp)
    auto u_max = metis::sym("u_max");
    auto u_saturated = u_max * metis::tanh(u / u_max);

    metis::Function pid_fn("pid_saturated",
                           {error, error_integral, error_derivative, Kp, Ki, Kd, u_max},
                           {u_saturated});

    metis::export_graph_deep(pid_fn.casadi_function(), "deep_pid", metis::DeepGraphFormat::HTML,
                             "PIDController");
    std::cout << "   Deep:    deep_pid.html\n";
    std::cout << "   Shows: *, +, tanh, / operations\n\n";

    // ============================================================
    // Summary
    // ============================================================
    std::cout << "=== Generated Files ===\n";
    std::cout << "Shallow graphs (MX-based, may have opaque function nodes):\n";
    std::cout << "  - shallow_trig.html\n";
    std::cout << "  - shallow_normalize.html\n\n";

    std::cout << "Deep graphs (SX-based, all operations expanded):\n";
    std::cout << "  - deep_trig.html\n";
    std::cout << "  - deep_normalize.html\n";
    std::cout << "  - deep_gravity.html\n";
    std::cout << "  - deep_gravity_jacobian.html\n";
    std::cout << "  - deep_pid.html\n\n";

    std::cout << "Open any .html file in a browser for interactive visualization.\n";
    std::cout << "Features: pan (drag), zoom (scroll), click nodes for details.\n\n";

    std::cout << "Node colors in deep graphs:\n";
    std::cout << "  - Green ellipse:  Symbolic inputs (x, y, mu, etc.)\n";
    std::cout << "  - Orange ellipse: Constants (0, 1, numeric values)\n";
    std::cout << "  - Light blue box: Arithmetic (+, -, *, /, neg)\n";
    std::cout << "  - Plum box:       Trigonometric (sin, cos, tan, etc.)\n";
    std::cout << "  - Pink box:       Power/Exp (sqrt, sq, exp, log, pow)\n";
    std::cout << "  - Gold circle:    Output nodes\n";

    return 0;
}
