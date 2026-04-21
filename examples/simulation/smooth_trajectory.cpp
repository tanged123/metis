/**
 * @file smooth_trajectory.cpp
 * @brief Graduation Example: Smooth Trajectory Optimization
 *
 * Demonstrates Phase 3 Metis features:
 * - Discrete integration (trapezoidal, Simpson)
 * - Squared curvature regularization
 * - Rotation matrices (Euler angles)
 * - Surrogate models (sigmoid blending)
 * - Finite difference coefficients
 *
 * Problem: Find a smooth 2D path from start to goal that:
 * 1. Minimizes path length
 * 2. Minimizes curvature (smoothness constraint)
 * 3. Avoids obstacle using smooth blending
 */

#include <iostream>
#include <metis/metis.hpp>
#include <numbers>

using namespace metis;
using Scalar = double;
using Vec = MetisVector<Scalar>;

int main() {
    std::cout << "=== Smooth Trajectory Optimization ===" << std::endl;
    std::cout << "Demonstrating Phase 3 Metis features\n" << std::endl;

    // --- 1. Define waypoints for the trajectory ---
    // A curved path that avoids an obstacle at (1.5, 0.5)
    constexpr int N = 11;          // Number of waypoints
    Vec t = linspace(0.0, 1.0, N); // Parameter

    // Initial straight-line path (will be curved)
    Vec x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x(i) = 3.0 * t(i); // Start (0,0) to (3, 0)
        // Add curve to avoid obstacle
        y(i) = std::sin(std::numbers::pi * t(i)) * 0.8; // Curved path
    }

    std::cout << "1. Path waypoints:" << std::endl;
    std::cout << "   Start: (" << x(0) << ", " << y(0) << ")" << std::endl;
    std::cout << "   End:   (" << x(N - 1) << ", " << y(N - 1) << ")" << std::endl;

    // --- 2. Compute path length using discrete integration ---
    std::cout << "\n2. Path Length Computation (Discrete Integration)" << std::endl;

    // dx/dt, dy/dt using finite differences
    auto dx = diff(x);
    auto dy = diff(y);
    auto dt = diff(t);

    // Speed at each segment: sqrt(dx^2 + dy^2)
    Vec speed(N - 1);
    for (int i = 0; i < N - 1; ++i) {
        speed(i) = std::sqrt(dx(i) * dx(i) + dy(i) * dy(i));
    }

    // Integrate speed to get arc length using different methods
    auto arc_trapz = integrate_discrete_intervals(speed, t.head(N - 1), true, "trapezoidal");
    Scalar total_length_trapz = arc_trapz.sum();

    std::cout << "   Path length (Trapezoidal): " << total_length_trapz << std::endl;

    // --- 3. Smoothness analysis using squared curvature ---
    std::cout << "\n3. Smoothness Analysis (Squared Curvature)" << std::endl;

    // Curvature penalty for x(t) and y(t)
    auto curv_x = integrate_discrete_squared_curvature(x, t, "simpson");
    auto curv_y = integrate_discrete_squared_curvature(y, t, "simpson");

    Scalar total_curvature = curv_x.sum() + curv_y.sum();
    std::cout << "   Total squared curvature: " << total_curvature << std::endl;
    std::cout << "   (Lower = smoother trajectory)" << std::endl;

    // --- 4. Heading computation using rotation matrices ---
    std::cout << "\n4. Heading Computation (Euler Angles)" << std::endl;

    Vec heading(N - 1);
    for (int i = 0; i < N - 1; ++i) {
        heading(i) = std::atan2(dy(i), dx(i)); // Heading angle
    }

    std::cout << "   Heading at start: " << heading(0) * 180.0 / std::numbers::pi << " deg"
              << std::endl;
    std::cout << "   Heading at middle: " << heading(N / 2) * 180.0 / std::numbers::pi << " deg"
              << std::endl;

    // Create rotation matrix for the heading (2D)
    auto R_start = rotation_matrix_2d(heading(0));
    std::cout << "   Rotation matrix at start:\n" << R_start << std::endl;

    // Validate it's a proper rotation matrix
    bool valid = is_valid_rotation_matrix(R_start);
    std::cout << "   Is valid rotation: " << (valid ? "Yes" : "No") << std::endl;

    // --- 5. Obstacle avoidance using smooth blending ---
    std::cout << "\n5. Obstacle Avoidance (Sigmoid Blending)" << std::endl;

    // Obstacle at (1.5, 0.0) with radius 0.3
    Scalar obs_x = 1.5, obs_y = 0.0, obs_r = 0.3;

    Vec distance_to_obs(N);
    Vec obstacle_cost(N);
    for (int i = 0; i < N; ++i) {
        distance_to_obs(i) =
            std::sqrt((x(i) - obs_x) * (x(i) - obs_x) + (y(i) - obs_y) * (y(i) - obs_y));

        // Smooth penalty: high when close, zero when far
        // Using sigmoid blend: penalty = sigmoid((obs_r - dist) / scale)
        Scalar margin = distance_to_obs(i) - obs_r;
        obstacle_cost(i) = sigmoid(-margin * 10.0); // Sharp transition at margin=0
    }

    Scalar total_obstacle_cost = obstacle_cost.sum();
    std::cout << "   Min distance to obstacle: " << distance_to_obs.minCoeff() << std::endl;
    std::cout << "   Obstacle penalty (sigmoid): " << total_obstacle_cost << std::endl;
    std::cout << "   (Should be ~0 since path avoids obstacle)" << std::endl;

    // --- 6. Finite difference coefficients for derivative estimation ---
    std::cout << "\n6. Finite Difference Coefficients" << std::endl;

    // Get 1st derivative coefficients for 3-point stencil
    Vec stencil(3);
    stencil << -1.0, 0.0, 1.0; // Central difference stencil
    auto fd_coeff = finite_difference_coefficients(stencil, 0.0, 1);

    std::cout << "   3-point central difference coefficients:" << std::endl;
    std::cout << "   " << fd_coeff.transpose() << std::endl;
    std::cout << "   (Should be [-0.5, 0, 0.5])" << std::endl;

    // --- 7. Summary: Optimization objective ---
    std::cout << "\n7. Combined Optimization Objective" << std::endl;

    Scalar w_length = 1.0;
    Scalar w_smooth = 0.1;
    Scalar w_obstacle = 100.0;

    Scalar objective = w_length * total_length_trapz + w_smooth * total_curvature +
                       w_obstacle * total_obstacle_cost;

    std::cout << "   Length term:   " << w_length * total_length_trapz << std::endl;
    std::cout << "   Smooth term:   " << w_smooth * total_curvature << std::endl;
    std::cout << "   Obstacle term: " << w_obstacle * total_obstacle_cost << std::endl;
    std::cout << "   ---------------------" << std::endl;
    std::cout << "   Total objective: " << objective << std::endl;

    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}
