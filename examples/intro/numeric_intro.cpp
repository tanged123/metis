#include <chrono>
#include <iostream>
#include <metis/metis.hpp>

/**
 * Numeric-Specific Example
 *
 * Demonstrates high-performance numeric execution using Metis standard types.
 * We implement a simple bouncing ball simulation.
 */

// Generic Physics Function
// Using "Scalar" template allows this to be reused for Symbolic later if needed.
template <typename Scalar> void step_physics(metis::MetisMatrix<Scalar> &state, const Scalar &dt) {
    // State: [y, vy]
    Scalar y = state(0);
    Scalar vy = state(1);

    Scalar g = 9.81;

    // Euler Integration
    y += vy * dt;
    vy -= g * dt;

    // Bounce Logic
    // usage: where(condition, if_true, if_false)
    // When y < 0, reverse velocity with damping
    vy = metis::where(y < 0.0, -0.8 * vy, vy);
    y = metis::where(y < 0.0, 0.0, y);

    // Write back
    state(0) = y;
    state(1) = vy;
}

int main() {
    std::cout << "--- Numeric Bouncing Ball Simulation ---\n";

    // Initialize State using Eigen syntax
    // NumericMatrix is standard Eigen::MatrixXd
    metis::NumericMatrix state(2, 1);
    state << 10.0, 0.0; // Initial height 10m, velocity 0

    double dt = 0.001;
    int steps = 10000;

    std::cout << "Initial State: " << state.transpose() << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Simulation Loop
    for (int i = 0; i < steps; ++i) {
        step_physics(state, dt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Final State (" << steps << " steps): " << state.transpose() << "\n";
    std::cout << "Simulation Time: " << elapsed.count() * 1000.0 << " ms\n";
    std::cout << "Throughput: " << steps / elapsed.count() << " steps/sec\n";

    // --- Symbolic Verification ---
    std::cout << "\n--- Symbolic Mode (Code Gen / Verification) ---\n";

    // 1. Create Symbols
    // step_physics expects Eigen structure for symbolic matrices
    auto state_mx = metis::sym("state", 2);
    auto state_sym = metis::to_eigen(state_mx); // Convert CasADi vector to Eigen<MX>

    // We need to keep the original symbols for the Function inputs
    auto state_next = state_sym; // Copy for modification

    auto dt_sym = metis::sym("dt"); // Scalar

    // 2. Build Graph (Reuse the exact same physics code!)
    step_physics(state_next, dt_sym);
    // Note: step_physics modifies state_next in-place.
    // The graph in state_next now represents the state *after* one step.

    // 3. Compile to Function: next_state = f(current_state, dt)
    // Inputs must be the original symbolic primitives (state_mx, dt_sym)
    // Outputs are the expressions (state_next)
    metis::Function step_fn({state_mx, dt_sym}, {state_next});

    // 4. Verify against Numeric
    // Let's test a specific condition where we bounce
    metis::NumericMatrix test_state(2, 1);
    test_state << -1.0, -5.0; // Below ground, moving down
    double test_dt = 0.1;

    // Numeric execution
    metis::NumericMatrix num_res = test_state;
    step_physics(num_res, test_dt);

    // Symbolic evaluation
    auto sym_res_vec = step_fn(test_state, test_dt);
    auto sym_res = sym_res_vec[0];

    std::cout << "Test State (Numeric):  " << num_res.transpose() << "\n";
    std::cout << "Test State (Symbolic): " << sym_res.transpose() << "\n";

    // Check diff
    double err = (num_res - sym_res).norm();
    if (err < 1e-9) {
        std::cout << "✅ Symbolic Graph matches Numeric Implementation perfectly.\n";
    } else {
        std::cout << "❌ Mismatch detected! Error: " << err << "\n";
    }

    return 0;
}
