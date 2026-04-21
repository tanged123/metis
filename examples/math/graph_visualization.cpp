#include <iostream>
#include <metis/metis.hpp>

/**
 * @brief Graph Visualization Example: Electric Motor Engineering
 *
 * Demonstrates graph visualization with realistic electromagnetic motor modeling.
 * Shows how to visualize complex computational graphs used in motor control.
 */

// Brushless DC Motor Model (simplified PMSM)
template <typename Scalar> struct MotorModel {
    // Motor parameters (as scalars for templating)
    Scalar Rs;     // Stator resistance [Ohm]
    Scalar Ld;     // d-axis inductance [H]
    Scalar Lq;     // q-axis inductance [H]
    Scalar lambda; // Permanent magnet flux linkage [Wb]
    Scalar p;      // Pole pairs
    Scalar J;      // Rotor inertia [kg*m^2]
    Scalar B;      // Viscous friction [N*m*s/rad]

    // Electromagnetic torque: T_e = (3/2) * p * [lambda*iq + (Ld - Lq)*id*iq]
    Scalar electromagnetic_torque(Scalar id, Scalar iq) const {
        Scalar torque_pm = lambda * iq;                 // PM torque component
        Scalar torque_reluctance = (Ld - Lq) * id * iq; // Reluctance torque
        return 1.5 * p * (torque_pm + torque_reluctance);
    }

    // d-axis voltage equation: Vd = Rs*id + Ld*did/dt - omega_e*Lq*iq
    Scalar voltage_d(Scalar id, Scalar iq, Scalar did_dt, Scalar omega_e) const {
        return Rs * id + Ld * did_dt - omega_e * Lq * iq;
    }

    // q-axis voltage equation: Vq = Rs*iq + Lq*diq/dt + omega_e*(Ld*id + lambda)
    Scalar voltage_q(Scalar id, Scalar iq, Scalar diq_dt, Scalar omega_e) const {
        return Rs * iq + Lq * diq_dt + omega_e * (Ld * id + lambda);
    }

    // Mechanical dynamics: J*domega/dt = T_e - T_load - B*omega
    Scalar mechanical_dynamics(Scalar id, Scalar iq, Scalar omega, Scalar T_load) const {
        Scalar T_e = electromagnetic_torque(id, iq);
        return (T_e - T_load - B * omega) / J;
    }

    // Total electrical power: P = 1.5 * (Vd*id + Vq*iq)
    Scalar electrical_power(Scalar Vd, Scalar Vq, Scalar id, Scalar iq) const {
        return 1.5 * (Vd * id + Vq * iq);
    }

    // Efficiency: eta = P_mech / P_elec
    Scalar efficiency(Scalar T_e, Scalar omega, Scalar P_elec) const {
        Scalar P_mech = T_e * omega;
        // Use softplus-based smooth max to avoid division by zero
        Scalar P_elec_safe = metis::log(1.0 + metis::exp(P_elec - 0.001)) + 0.001;
        return P_mech / P_elec_safe;
    }
};

int main() {
    std::cout << "=== Metis Graph Visualization: Electric Motor Model ===\n\n";

    // Create symbolic variables for motor state
    auto id = metis::sym("id");           // d-axis current [A]
    auto iq = metis::sym("iq");           // q-axis current [A]
    auto omega = metis::sym("omega");     // Mechanical angular velocity [rad/s]
    auto omega_e = metis::sym("omega_e"); // Electrical angular velocity [rad/s]
    auto did_dt = metis::sym("did_dt");   // d-current derivative
    auto diq_dt = metis::sym("diq_dt");   // q-current derivative
    auto T_load = metis::sym("T_load");   // Load torque [N*m]

    // Motor parameters (symbolic for generality)
    auto Rs = metis::sym("Rs");
    auto Ld = metis::sym("Ld");
    auto Lq = metis::sym("Lq");
    auto lambda = metis::sym("lambda");
    auto p = metis::sym("p");
    auto J = metis::sym("J");
    auto B = metis::sym("B");

    // Build motor model
    MotorModel<metis::SymbolicScalar> motor{Rs, Ld, Lq, lambda, p, J, B};

    // ============================================================
    // Graph 1: Electromagnetic Torque
    // ============================================================
    std::cout << "1. Electromagnetic Torque Expression\n";
    auto T_e = motor.electromagnetic_torque(id, iq);
    metis::export_graph_dot(T_e, "graph_em_torque", "ElectromagneticTorque");
    metis::render_graph("graph_em_torque.dot", "graph_em_torque.pdf");
    metis::export_graph_html(T_e, "graph_em_torque", "ElectromagneticTorque");
    std::cout << "   T_e = 1.5 * p * [lambda*iq + (Ld-Lq)*id*iq]\n";
    std::cout << "   -> graph_em_torque.pdf / .html\n\n";

    // ============================================================
    // Graph 2: Voltage Equations (Park Transform)
    // ============================================================
    std::cout << "2. Park Transform Voltage Equations\n";
    auto Vd = motor.voltage_d(id, iq, did_dt, omega_e);
    auto Vq = motor.voltage_q(id, iq, diq_dt, omega_e);
    auto V_magnitude = metis::sqrt(Vd * Vd + Vq * Vq);

    metis::export_graph_dot(V_magnitude, "graph_voltage", "VoltageMagnitude");
    metis::render_graph("graph_voltage.dot", "graph_voltage.pdf");
    metis::export_graph_html(V_magnitude, "graph_voltage", "VoltageMagnitude");
    std::cout << "   |V| = sqrt(Vd^2 + Vq^2)\n";
    std::cout << "   -> graph_voltage.pdf / .html\n\n";

    // ============================================================
    // Graph 3: Complete Motor Dynamics (ODE RHS)
    // ============================================================
    std::cout << "3. Motor Mechanical Dynamics (ODE)\n";
    auto domega_dt = motor.mechanical_dynamics(id, iq, omega, T_load);

    metis::export_graph_dot(domega_dt, "graph_dynamics", "MechanicalDynamics");
    metis::render_graph("graph_dynamics.dot", "graph_dynamics.pdf");
    metis::export_graph_html(domega_dt, "graph_dynamics", "MechanicalDynamics");
    std::cout << "   d(omega)/dt = (T_e - T_load - B*omega) / J\n";
    std::cout << "   -> graph_dynamics.pdf / .html\n\n";

    // ============================================================
    // Graph 4: Electrical Power with Losses
    // ============================================================
    std::cout << "4. Electrical Power\n";
    auto P_elec = motor.electrical_power(Vd, Vq, id, iq);

    metis::export_graph_dot(P_elec, "graph_power", "ElectricalPower");
    metis::render_graph("graph_power.dot", "graph_power.pdf");
    metis::export_graph_html(P_elec, "graph_power", "ElectricalPower");
    std::cout << "   P = 1.5 * (Vd*id + Vq*iq)\n";
    std::cout << "   -> graph_power.pdf / .html\n\n";

    // ============================================================
    // Create callable metis::Function for torque
    // ============================================================
    std::cout << "5. Creating metis::Function for Jacobian\n";
    metis::Function torque_fn("torque", {id, iq, lambda, Ld, Lq, p}, {T_e});

    // Compute Jacobian of torque w.r.t. currents
    auto dT_dq = metis::jacobian({T_e}, {id, iq});
    metis::export_graph_dot(dT_dq, "graph_jacobian", "TorqueJacobian");
    metis::render_graph("graph_jacobian.dot", "graph_jacobian.pdf");
    metis::export_graph_html(dT_dq, "graph_jacobian", "TorqueJacobian");
    std::cout << "   dT/d[id, iq] Jacobian computed symbolically\n";
    std::cout << "   -> graph_jacobian.pdf / .html\n\n";

    // ============================================================
    // Evaluate numerically
    // ============================================================
    std::cout << "6. Numeric Evaluation\n";
    // Typical PMSM parameters:
    // Rs=0.5Ω, Ld=Lq=8.5mH, lambda=0.175Wb, p=4, J=0.089kg*m^2, B=0.005
    auto T_numeric = torque_fn.eval(0.0,    // id = 0 (MTPA control)
                                    10.0,   // iq = 10A
                                    0.175,  // lambda
                                    0.0085, // Ld
                                    0.0085, // Lq
                                    4.0     // pole pairs
    );
    std::cout << "   T_e(id=0, iq=10A) = " << T_numeric(0, 0) << " N*m\n";
    std::cout << "   Expected: 1.5 * 4 * 0.175 * 10 = 10.5 N*m\n\n";

    std::cout << "=== Complete! ===\n";
    std::cout << "View PDF graphs:  xdg-open graph_dynamics.pdf\n";
    std::cout << "View HTML graphs: xdg-open graph_dynamics.html (interactive pan/zoom)\n";

    return 0;
}
