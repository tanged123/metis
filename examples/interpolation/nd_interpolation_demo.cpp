#include <cmath>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

/**
 * N-Dimensional Interpolation Demo
 *
 * Demonstrates 2D interpolation using both Linear and B-Spline methods,
 * including numeric evaluation and symbolic gradient computation.
 */
int main() {
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "=== N-Dimensional Interpolation Demo (2D) ===\n";

    // 1. Definition: z = sin(x) * cos(y)
    // Domain: x in [0, pi], y in [0, pi]

    // Grid generation
    int N = 10;
    metis::NumericVector x_grid = metis::NumericVector::LinSpaced(N, 0, M_PI);
    metis::NumericVector y_grid = metis::NumericVector::LinSpaced(N, 0, M_PI);

    // Create meshgrid values
    // metis::interpn assumes values(i, j) corresponds to grid[0][i], grid[1][j]
    metis::NumericMatrix values(N, N);
    for (int i = 0; i < N; ++i) {     // x index (rows)
        for (int j = 0; j < N; ++j) { // y index (cols)
            values(i, j) = std::sin(x_grid(i)) * std::cos(y_grid(j));
        }
    }

    std::vector<metis::NumericVector> grids = {x_grid, y_grid};

    // Flatten values for interpn (Eigen defaults to column-major, which matches CasADi/Fortran
    // order)
    metis::NumericVector values_flat =
        Eigen::Map<const metis::NumericVector>(values.data(), values.size());

    std::cout << "Function: z = sin(x) * cos(y)\n";
    std::cout << "Grid size: " << N << "x" << N << " (" << x_grid(0) << " to " << x_grid(N - 1)
              << ")\n";

    // 2. Linear Interpolation
    std::cout << "\n--- Linear Interpolation ---\n";
    // Query point at center of a cell
    double qx = M_PI / 4.0;                     // 45 deg
    double qy = M_PI / 3.0;                     // 60 deg
    double exact = std::sin(qx) * std::cos(qy); // 0.707 * 0.5 = 0.35355

    // interpn expects Matrix<Scalar, Dynamic, Dynamic> (n_points x n_dims)
    metis::NumericMatrix query(1, 2);
    query << qx, qy;

    double val_linear =
        metis::interpn(grids, values_flat, query, metis::InterpolationMethod::Linear)(0);
    std::cout << "Query point:   (" << qx << ", " << qy << ")\n";
    std::cout << "Exact value:     " << exact << "\n";
    std::cout << "Linear Interp:   " << val_linear << " (Error: " << std::abs(val_linear - exact)
              << ")\n";

    // 3. B-Spline Interpolation
    std::cout << "\n--- B-Spline Interpolation ---\n";
    double val_bspline =
        metis::interpn(grids, values_flat, query, metis::InterpolationMethod::BSpline)(0);
    std::cout << "BSpline Interp:  " << val_bspline << " (Error: " << std::abs(val_bspline - exact)
              << ")\n";
    std::cout << "Note: B-Spline typically offers higher accuracy for smooth functions.\n";

    // 4. Symbolic Interpolation & Differentiation
    std::cout << "\n--- Symbolic Interpolation & Gradient ---\n";
    auto x_sym = metis::sym("x");
    auto y_sym = metis::sym("y");

    // Create symbolic matrix for query (1 point, 2 dims) using SymbolicMatrix (Dynamic, Dynamic)
    metis::SymbolicMatrix q_sym(1, 2);
    q_sym(0, 0) = x_sym;
    q_sym(0, 1) = y_sym;

    // Interpolate symbolically (returns vector result)
    auto z_sym = metis::interpn(grids, values_flat, q_sym, metis::InterpolationMethod::BSpline);

    // Compute Gradient (Jacobian of output w.r.t input)
    // d(interp)/d[x,y]
    // To diff w.r.t. input variables, we pass variables as vector/matrix
    // jacobian(expr, vars). q_sym is SymbolicMatrix. Need to cast/use consistent types.
    // metis::jacobian handles vector<MX>.
    // q_sym matrix contains x_sym, y_sym.
    // We can pass {x_sym, y_sym} explicitly or use q_sym if converted.

    auto grad_z = metis::jacobian(z_sym(0), x_sym, y_sym);

    // Create callable function
    metis::Function f_grad("gradient", {x_sym, y_sym}, {grad_z});

    // Evaluate gradient numerically at query point
    auto grad_val = f_grad(qx, qy)[0];
    double dz_dx = grad_val(0, 0);
    double dz_dy = grad_val(0, 1);

    // Exact gradient:
    // dz/dx = cos(x)*cos(y)
    // dz/dy = sin(x)*(-sin(y))
    double exact_dz_dx = std::cos(qx) * std::cos(qy);
    double exact_dz_dy = std::sin(qx) * -std::sin(qy);

    std::cout << "Calculated Gradient: [" << dz_dx << ", " << dz_dy << "]\n";
    std::cout << "Exact Gradient:      [" << exact_dz_dx << ", " << exact_dz_dy << "]\n";
    std::cout << "Gradient Error:      [" << std::abs(dz_dx - exact_dz_dx) << ", "
              << std::abs(dz_dy - exact_dz_dy) << "]\n";

    std::cout << "\n=== Demo Complete ===\n";
    return 0;
}
