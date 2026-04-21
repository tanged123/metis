#include <iostream>
#include <metis/metis.hpp>

/**
 * @brief Demonstrate loop patterns in Metis
 *
 * KEY INSIGHT: Loop indices and bounds must be STRUCTURAL (known at graph-building time),
 * but the VALUES being computed can be symbolic.
 *
 * ✅ WORKS: for (int i = 0; i < 10; i++) { result += symbolic_value(i); }
 * ❌ DOESN'T WORK: while (symbolic_value > threshold) { ... }
 *                  (condition depends on symbolic value, not structural)
 */

// Example 1: Simple For Loop - Accumulation
template <typename Scalar> Scalar sum_of_squares(int n) {
    // Loop index 'i' is structural (int)
    // Values being summed can be symbolic
    Scalar sum = 0.0;
    for (int i = 1; i <= n; ++i) {
        Scalar value = static_cast<Scalar>(i);
        sum += value * value;
    }
    return sum;
}

// Example 2: Nested For Loops - Matrix Operations
template <typename Scalar>
metis::MetisMatrix<Scalar> matrix_multiply_manual(const metis::MetisMatrix<Scalar> &A,
                                                  const metis::MetisMatrix<Scalar> &B) {
    // Nested loops for matrix multiplication
    // Loop bounds are structural (matrix dimensions)
    int m = A.rows();
    int n = A.cols();
    int p = B.cols();

    metis::MetisMatrix<Scalar> C(m, p);
    C.setZero();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return C;
}

// Example 3: Iterative Algorithm - Newton-Raphson
template <typename Scalar> Scalar newton_raphson_sqrt(const Scalar &x, int max_iterations = 10) {
    // Fixed number of iterations (structural)
    // Values being refined are symbolic
    Scalar guess = x / 2.0; // Initial guess

    for (int i = 0; i < max_iterations; ++i) {
        // Newton's method: x_new = (x_old + x/x_old) / 2
        guess = (guess + x / guess) / 2.0;
    }

    return guess;
}

// Example 4: While Loop Replacement - Fixed Iterations
template <typename Scalar> Scalar exponential_series(const Scalar &x, int n_terms = 20) {
    // C++ while (error > tol): ❌ NO! (error is symbolic)
    // Replace with fixed iterations: ✅ YES

    Scalar result = 1.0; // e^x ≈ 1 + x + x^2/2! + x^3/3! + ...
    Scalar term = 1.0;

    for (int i = 1; i <= n_terms; ++i) {
        term *= x / static_cast<Scalar>(i);
        result += term;
    }

    return result;
}

// Example 5: Element-wise Operations with Loops
template <typename Scalar>
metis::MetisVector<Scalar> apply_sigmoid_loop(const metis::MetisVector<Scalar> &input) {
    // Process each element in a loop
    int n = input.size();
    metis::MetisVector<Scalar> output(n);

    for (int i = 0; i < n; ++i) {
        // Apply sigmoid: 1 / (1 + exp(-x))
        output(i) = 1.0 / (1.0 + metis::exp(-input(i)));
    }

    return output;
}

// Example 6: Conditional Accumulation
template <typename Scalar>
Scalar selective_sum(const metis::MetisVector<Scalar> &values,
                     const metis::MetisVector<Scalar> &thresholds) {
    // Sum only values above thresholds
    // Use where() for conditional logic inside loop
    Scalar sum = 0.0;

    for (int i = 0; i < values.size(); ++i) {
        // Add value if above threshold, else add 0
        sum += metis::where(values(i) > thresholds(i), values(i), Scalar(0.0));
    }

    return sum;
}

// Example 7: Multi-Stage Pipeline
template <typename Scalar>
Scalar aerodynamic_load_distribution(int n_panels, const Scalar &alpha, const Scalar &velocity) {
    // Compute total lift from distributed panels
    Scalar total_lift = 0.0;
    Scalar chord = 1.0; // Chord length

    for (int i = 0; i < n_panels; ++i) {
        // Position along wing (structural)
        double x_pos = static_cast<double>(i) / n_panels;

        // Local angle of attack varies with position (symbolic)
        Scalar local_alpha = alpha * (1.0 - 0.1 * x_pos);

        // Local lift coefficient (symbolic)
        Scalar cl_local = 2.0 * M_PI * local_alpha;

        // Panel area (structural)
        Scalar panel_area = chord / n_panels;

        // Dynamic pressure (symbolic)
        Scalar q = 0.5 * 1.225 * velocity * velocity;

        // Lift from this panel
        Scalar lift_panel = q * panel_area * cl_local;
        total_lift += lift_panel;
    }

    return total_lift;
}

// Example 8: Break/Continue Pattern
template <typename Scalar>
Scalar find_first_above_threshold(const metis::MetisVector<Scalar> &values,
                                  const Scalar &threshold) {
    // C++ break/continue: ❌ NO! (depends on symbolic condition)
    // Workaround: Use select() to accumulate result

    Scalar result = -1.0; // Default if not found
    Scalar found = 0.0;   // Flag: 0 = not found yet, 1 = found

    for (int i = 0; i < values.size(); ++i) {
        // If we haven't found it yet AND this value is above threshold
        Scalar is_match = values(i) > threshold;
        Scalar new_find = (1.0 - found) * is_match; // Only if not found yet

        // Update result if we found a new match
        result = metis::where(new_find > 0.5, static_cast<Scalar>(i), result);

        // Update found flag
        found = metis::where(new_find > 0.5, Scalar(1.0), found);
    }

    return result;
}

// ======================================================================
// IMPORTANT: Dynamic Loops (While, Break, Continue)
// ======================================================================

// Example 9a: Dynamic While Loop - NUMERIC MODE ONLY
double simulate_until_convergence_numeric(double x0, double tolerance = 1e-6) {
    // ✅ This works in NUMERIC mode!
    // Loop continues until convergence (dynamic condition)

    double x = x0;
    double error = 1000.0;
    int iteration = 0;
    int max_iter = 1000; // Safety limit

    while (error > tolerance && iteration < max_iter) {
        double x_new = 0.5 * (x + 2.0 / x); // Newton's method for sqrt(2)
        error = std::abs(x_new - x);
        x = x_new;
        iteration++;
    }

    return x;
}

// Example 9b: Symbolic Version - FIXED iterations
template <typename Scalar>
Scalar simulate_fixed_iterations_symbolic(const Scalar &x0, int n_iterations = 10) {
    // ✅ This works in SYMBOLIC mode!
    // Fixed number of iterations (structural)

    Scalar x = x0;

    for (int i = 0; i < n_iterations; ++i) {
        x = 0.5 * (x + 2.0 / x); // Same update, but fixed iterations
    }

    return x;
}

// Example 10: Hybrid Approach - Unroll converged numeric into symbolic
template <typename Scalar> Scalar hybrid_simulation(const Scalar &x0) {
    // Strategy: Run numeric version ONCE to determine iteration count,
    // then use that fixed count for symbolic version

    // In practice, you'd determine n from numeric experiments
    // For now, we know sqrt(2) converges in ~5 iterations
    int n_iterations = 5;

    return simulate_fixed_iterations_symbolic(x0, n_iterations);
}

// Example 11: Time-stepping with Early Exit (Numeric Only)
struct SimulationResult {
    double final_value;
    int steps_taken;
    bool converged;
};

SimulationResult adaptive_timestep_simulation(double x0, double dt = 0.01) {
    // ✅ NUMERIC MODE: Dynamic timestep with convergence check

    double x = x0;
    double time = 0.0;
    int steps = 0;
    int max_steps = 10000;

    // Simulate until steady state OR max time
    while (steps < max_steps) {
        // Simple damped oscillator: dx/dt = -0.5*x
        double dx_dt = -0.5 * x;
        x += dx_dt * dt;
        time += dt;
        steps++;

        // Check convergence
        if (std::abs(x) < 1e-4) {
            return {x, steps, true};
        }
    }

    return {x, steps, false}; // Didn't converge
}

// Example 12: Event Detection (Numeric Only)
double find_zero_crossing_numeric(double (*func)(double), double x_start, double x_end) {
    // ✅ NUMERIC MODE: Binary search for zero crossing
    // Cannot do this symbolically because break condition is dynamic

    double a = x_start;
    double b = x_end;
    double tolerance = 1e-6;
    int max_iter = 50;

    for (int i = 0; i < max_iter; ++i) {
        double mid = (a + b) / 2.0;
        double f_mid = func(mid);

        // Early exit if close enough
        if (std::abs(f_mid) < tolerance) {
            return mid;
        }

        // Bisection
        if (f_mid * func(a) < 0) {
            b = mid;
        } else {
            a = mid;
        }
    }

    return (a + b) / 2.0;
}

int main() {
    std::cout << "=== Loop Patterns in Metis ===\n\n";

    // Test 1: Sum of Squares
    std::cout << "1. Simple For Loop - Sum of Squares (1² + 2² + ... + 5²):\n";
    std::cout << "   Numeric: " << sum_of_squares<double>(5) << "\n";
    std::cout << "   Expected: " << (1 + 4 + 9 + 16 + 25) << "\n";

    // Test 2: Matrix Multiply
    std::cout << "\n2. Nested For Loops - Matrix Multiplication:\n";
    metis::MetisMatrix<double> A(2, 3);
    A << 1, 2, 3, 4, 5, 6;
    metis::MetisMatrix<double> B(3, 2);
    B << 7, 8, 9, 10, 11, 12;
    auto C = matrix_multiply_manual(A, B);
    std::cout << "   A =\n" << A << "\n";
    std::cout << "   B =\n" << B << "\n";
    std::cout << "   C = A*B =\n" << C << "\n";
    std::cout << "   (Compare with Eigen: " << (A * B).sum() << " vs Manual: " << C.sum() << ")\n";

    // Test 3: Newton-Raphson
    std::cout << "\n3. Iterative Algorithm - Newton-Raphson sqrt(16):\n";
    std::cout << "   Result: " << newton_raphson_sqrt(16.0, 5) << "\n";
    std::cout << "   Expected: 4.0\n";

    // Test 4: Exponential Series
    std::cout << "\n4. While Loop Replacement - Exponential Series e^1:\n";
    std::cout << "   Result: " << exponential_series(1.0, 20) << "\n";
    std::cout << "   Expected (e): " << std::exp(1.0) << "\n";

    // Test 5: Element-wise Sigmoid
    std::cout << "\n5. Element-wise Operations:\n";
    metis::MetisVector<double> input(3);
    input << 0.0, 1.0, -1.0;
    auto output = apply_sigmoid_loop(input);
    std::cout << "   Input: " << input.transpose() << "\n";
    std::cout << "   Sigmoid: " << output.transpose() << "\n";

    // Test 6: Conditional Sum
    std::cout << "\n6. Conditional Accumulation:\n";
    metis::MetisVector<double> values(5);
    values << 1.0, 5.0, 3.0, 7.0, 2.0;
    metis::MetisVector<double> thresholds(5);
    thresholds << 2.0, 4.0, 4.0, 6.0, 3.0;
    double selective = selective_sum(values, thresholds);
    std::cout << "   Values: " << values.transpose() << "\n";
    std::cout << "   Thresholds: " << thresholds.transpose() << "\n";
    std::cout << "   Sum (if value > threshold): " << selective << "\n";
    std::cout << "   (5 + 7 = 12)\n";

    // Test 7: Symbolic Mode - Aerodynamic Load
    std::cout << "\n7. Symbolic Mode - Distributed Loads with Derivatives:\n";
    auto alpha_sym = metis::sym("alpha");
    auto v_sym = metis::sym("v");

    auto lift_expr = aerodynamic_load_distribution(10, alpha_sym, v_sym);

    // Compute dLift/dv (sensitivity to velocity)
    auto dL_dv = metis::jacobian({lift_expr}, {v_sym});
    metis::Function dL_dv_fun({alpha_sym, v_sym}, {dL_dv});

    double alpha_val = 0.1; // ~5.7 degrees
    double v_val = 50.0;
    auto sensitivity = dL_dv_fun.eval(alpha_val, v_val);

    std::cout << "   Lift sensitivity dL/dv at α=0.1 rad, v=50 m/s:\n";
    std::cout << "   " << sensitivity(0, 0) << " N/(m/s)\n";
    std::cout << "   💡 Loops work great with symbolic differentiation!\n";

    // Test 8: Find First
    std::cout << "\n8. Break/Continue Pattern - Find First Above Threshold:\n";
    metis::MetisVector<double> search_vals(5);
    search_vals << 1.0, 3.0, 5.0, 7.0, 9.0;
    double found_idx = find_first_above_threshold(search_vals, 4.5);
    std::cout << "   Values: " << search_vals.transpose() << "\n";
    std::cout << "   First index > 4.5: " << found_idx
              << " (value: " << search_vals(static_cast<int>(found_idx)) << ")\n";

    // Test 9: Dynamic While Loop (Numeric Only)
    std::cout << "\n9. Dynamic While Loop - NUMERIC MODE ONLY:\n";
    double sqrt2_dynamic = simulate_until_convergence_numeric(2.0);
    std::cout << "   sqrt(2) with dynamic convergence: " << sqrt2_dynamic << "\n";
    std::cout << "   Expected: " << std::sqrt(2.0) << "\n";
    std::cout << "   ⚠️  This ONLY works in numeric mode!\n";

    // Test 10: Symbolic Version with Fixed Iterations
    std::cout << "\n10. Symbolic Mode with Fixed Iterations:\n";
    auto x_sym = metis::sym("x");
    auto sqrt_expr = simulate_fixed_iterations_symbolic(x_sym, 5);
    metis::Function sqrt_fun({x_sym}, {sqrt_expr});
    double sqrt2_symbolic = sqrt_fun.eval(2.0)(0, 0);
    std::cout << "   sqrt(2) with fixed 5 iterations (symbolic compatible): " << sqrt2_symbolic
              << "\n";
    std::cout << "   ✅ This works in BOTH numeric and symbolic modes!\n";

    // Test 11: Adaptive Timestep (Numeric Only)
    std::cout << "\n11. Adaptive Time-Stepping - NUMERIC MODE ONLY:\n";
    auto result = adaptive_timestep_simulation(10.0);
    std::cout << "   Initial: x=10.0, damped oscillator\n";
    std::cout << "   Final value: " << result.final_value << "\n";
    std::cout << "   Steps taken: " << result.steps_taken << "\n";
    std::cout << "   Converged: " << (result.converged ? "Yes" : "No") << "\n";
    std::cout << "   ⚠️  Early exit based on convergence - numeric mode only!\n";

    // Test 12: Zero Crossing (Numeric Only)
    std::cout << "\n12. Event Detection - NUMERIC MODE ONLY:\n";
    auto parabola = [](double x) { return x * x - 2.0; };
    double zero = find_zero_crossing_numeric(parabola, 0.0, 3.0);
    std::cout << "   Find zero of f(x) = x² - 2 in [0, 3]\n";
    std::cout << "   Zero at x = " << zero << "\n";
    std::cout << "   Check: f(" << zero << ") = " << parabola(zero) << "\n";
    std::cout << "   ⚠️  Binary search with early exit - numeric mode only!\n";

    std::cout << "\n✅ All loop examples completed!\n";
    std::cout << "\n💡 KEY TAKEAWAYS:\n";
    std::cout << "   STRUCTURAL vs DYNAMIC loops:\n";
    std::cout << "   \n";
    std::cout << "   ✅ STRUCTURAL (works in symbolic mode):\n";
    std::cout << "      for (int i = 0; i < N; i++)  // N is int/const\n";
    std::cout << "      - Loop bounds known at graph-building time\n";
    std::cout << "      - Values computed can be symbolic\n";
    std::cout << "   \n";
    std::cout << "   ⚠️  DYNAMIC (numeric mode ONLY):\n";
    std::cout << "      while (error > tolerance)    // error is runtime value\n";
    std::cout << "      if (...) break;              // dynamic condition\n";
    std::cout << "      - Loop continues until runtime condition\n";
    std::cout << "      - Cannot build computational graph\n";
    std::cout << "   \n";
    std::cout << "   💡 SOLUTION:\n";
    std::cout << "      - Pure simulation: Use dynamic loops freely!\n";
    std::cout << "      - Optimization/AutoDiff: Use fixed iterations\n";
    std::cout << "      - Hybrid: Run numeric to determine N, then use fixed N symbolically\n";

    return 0;
}
