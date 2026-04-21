#include <cmath>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

/**
 * Scattered Data Interpolation Demo
 *
 * Demonstrates interpolation of unstructured (scattered) point cloud data
 * using Radial Basis Functions (RBF). Unlike gridded interpolation, scattered
 * interpolation works with arbitrary point locations.
 */
int main() {
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "=== Scattered Data Interpolation Demo ===\n";

    // =========================================================================
    // 1. Basic 1D Scattered Interpolation
    // =========================================================================
    std::cout << "\n--- 1D Scattered Interpolation ---\n";
    {
        // Non-uniformly spaced data points
        metis::NumericVector x(8);
        metis::NumericVector y(8);
        x << 0.0, 0.3, 0.7, 1.2, 2.0, 2.8, 3.5, 4.0;
        y << 0.0, 0.29, 0.64, 1.0, 0.91, 0.33, -0.35, -0.75; // sin(x)

        metis::ScatteredInterpolator interp(x, y);

        std::cout << "Data points: " << x.transpose() << "\n";
        std::cout << "Values:      " << y.transpose() << "\n";

        // Query at some test points
        double q1 = 1.5, q2 = 2.5;
        double v1 = interp(q1);
        double v2 = interp(q2);

        std::cout << "\nQuery at x=" << q1 << ": " << v1 << " (exact sin: " << std::sin(q1)
                  << ")\n";
        std::cout << "Query at x=" << q2 << ": " << v2 << " (exact sin: " << std::sin(q2) << ")\n";
        std::cout << "Reconstruction error: " << interp.reconstruction_error() << "\n";
    }

    // =========================================================================
    // 2. 2D Scattered Interpolation
    // =========================================================================
    std::cout << "\n--- 2D Scattered Interpolation ---\n";
    {
        // Simple 2D test: z = x + y
        int n_points = 16;
        metis::NumericMatrix points(n_points, 2);
        metis::NumericVector values(n_points);

        // Scattered points in [0, 3] x [0, 3]
        double xs[] = {0.2, 0.8, 1.5, 2.3, 0.5, 1.2, 2.0, 2.8,
                       0.3, 1.0, 1.8, 2.5, 0.7, 1.4, 2.2, 2.9};
        double ys[] = {0.3, 0.7, 1.2, 0.4, 1.5, 2.0, 1.3, 0.9,
                       2.5, 1.8, 2.3, 1.6, 2.8, 0.2, 2.7, 1.1};

        for (int i = 0; i < n_points; ++i) {
            points(i, 0) = xs[i];
            points(i, 1) = ys[i];
            values(i) = xs[i] + ys[i]; // z = x + y
        }

        metis::ScatteredInterpolator interp(points, values, 30);

        std::cout << "Function: z = x + y\n";
        std::cout << "Input: " << n_points << " scattered (x, y) points\n";
        std::cout << "Reconstruction error: " << interp.reconstruction_error() << "\n";

        // Query at test point
        metis::NumericVector query(2);
        query << 1.5, 1.5;
        double result = interp(query);
        double expected = 3.0; // 1.5 + 1.5

        std::cout << "\nQuery at (1.5, 1.5):\n";
        std::cout << "  Interpolated: " << result << "\n";
        std::cout << "  Expected:     " << expected << "\n";
        std::cout << "  Error:        " << std::abs(result - expected) << "\n";
    }

    // =========================================================================
    // 3. Different RBF Kernels
    // =========================================================================
    std::cout << "\n--- RBF Kernel Comparison ---\n";
    {
        metis::NumericVector x(10);
        metis::NumericVector y(10);
        for (int i = 0; i < 10; ++i) {
            x(i) = static_cast<double>(i) * 0.5;
            y(i) = std::exp(-x(i) * 0.3) * std::sin(x(i));
        }

        double query_pt = 2.25;
        double exact = std::exp(-query_pt * 0.3) * std::sin(query_pt);

        metis::ScatteredInterpolator tps(x, y, 50, metis::RBFKernel::ThinPlateSpline);
        metis::ScatteredInterpolator mq(x, y, 50, metis::RBFKernel::Multiquadric);
        metis::ScatteredInterpolator gauss(x, y, 50, metis::RBFKernel::Gaussian);
        metis::ScatteredInterpolator linear(x, y, 50, metis::RBFKernel::Linear);

        std::cout << "Query at x=" << query_pt << " (exact: " << exact << ")\n";
        std::cout << "  Thin Plate Spline: " << tps(query_pt)
                  << " (error: " << std::abs(tps(query_pt) - exact) << ")\n";
        std::cout << "  Multiquadric:      " << mq(query_pt)
                  << " (error: " << std::abs(mq(query_pt) - exact) << ")\n";
        std::cout << "  Gaussian:          " << gauss(query_pt)
                  << " (error: " << std::abs(gauss(query_pt) - exact) << ")\n";
        std::cout << "  Linear:            " << linear(query_pt)
                  << " (error: " << std::abs(linear(query_pt) - exact) << ")\n";
    }

    // =========================================================================
    // 4. Symbolic Mode & Gradient
    // =========================================================================
    std::cout << "\n--- Symbolic Interpolation & Gradient ---\n";
    {
        metis::NumericVector x(10);
        metis::NumericVector y(10);
        for (int i = 0; i < 10; ++i) {
            x(i) = static_cast<double>(i);
            y(i) = x(i) * x(i); // y = x^2
        }

        metis::ScatteredInterpolator interp(x, y, 100);

        // Create symbolic function
        auto sym_x = metis::sym("x");
        auto sym_result = interp(sym_x);

        // Compute derivative
        auto grad = metis::jacobian(sym_result, sym_x);
        metis::Function df("df", {sym_x}, {grad});

        // Evaluate at x=3
        double query_pt = 3.0;
        auto result = df(query_pt);
        double gradient = result[0](0, 0);

        std::cout << "Function: y = x^2 (sampled as scattered points)\n";
        std::cout << "Query at x=" << query_pt << "\n";
        std::cout << "  Computed gradient: " << gradient << "\n";
        std::cout << "  Exact dy/dx = 2x:  " << 2.0 * query_pt << "\n";
    }

    std::cout << "\n=== Demo Complete ===\n";
    return 0;
}
