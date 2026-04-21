/**
 * @file sparsity_intro.cpp
 * @brief Demonstration of Metis Sparsity Introspection
 *
 * Shows how to analyze sparsity patterns of Jacobians and Hessians,
 * which is critical for understanding the structure of optimization problems.
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>

using namespace metis;

void print_section(const std::string &title) {
    std::cout << "\n=== " << title << " ===\n" << std::endl;
}

int main() {
    print_section("Sparsity Introspection Demo");

    // 1. Jacobian Sparsity of a Vector Function
    // -----------------------------------------
    // Let's analyze a simple physics-like function
    // f(x) = [x0^2, x0*x1, x2]
    {
        auto x = sym("x", 3);
        auto f = SymbolicScalar::vertcat({
            x(0) * x(0), // Depends only on x0
            x(0) * x(1), // Depends on x0 and x1
            x(2)         // Depends only on x2
        });

        auto sp = sparsity_of_jacobian(f, x);

        std::cout << "Function f(x) -> Jacobian J(x):\n";
        std::cout << "  f[0] = x0^2   (depends on x0)\n";
        std::cout << "  f[1] = x0*x1  (depends on x0, x1)\n";
        std::cout << "  f[2] = x2     (depends on x2)\n\n";

        std::cout << sp.to_string() << "\n";
    }

    // 2. Hessian Sparsity (The "Arrowhead" Pattern)
    // ---------------------------------------------
    // Common in optimal control: variables are coupled to neighbors
    // f = sum((x[i] - x[i+1])^2)
    {
        int N = 10;
        auto x = sym("x", N);

        SymbolicScalar f = 0;
        for (int i = 0; i < N - 1; ++i) {
            auto diff = x(i) - x(i + 1);
            f = f + diff * diff;
        }

        auto sp = sparsity_of_hessian(f, x);

        std::cout << "Tridiagonal Hessian (Chain Structure):\n";
        std::cout << sp.to_string() << "\n";
    }

    // 3. Block-Diagonal Structure
    // ---------------------------
    // Independent systems combined together
    {
        int N = 4;
        auto x = sym("x", N);
        auto y = sym("y", N);

        // System 1 depends only on x, System 2 depends only on y
        auto sys1 = SymbolicScalar::mtimes(to_mx(NumericMatrix::Random(N, N)), x);
        auto sys2 = SymbolicScalar::mtimes(to_mx(NumericMatrix::Random(N, N)), y);

        auto combined_out = SymbolicScalar::vertcat({sys1, sys2});
        auto combined_in = SymbolicScalar::vertcat({x, y});

        auto sp = sparsity_of_jacobian(combined_out, combined_in);

        std::cout << "Block-Diagonal System (Independent Subsystems):\n";
        std::cout << sp.to_string() << "\n";

        // Export to DOT and render to PDF
        std::cout << "Exporting 'block_diag.pdf' for visualization...\n";
        bool success = sp.visualize_spy("block_diag");
        if (success) {
            std::cout << "Successfully rendered block_diag.pdf\n";
        } else {
            std::cout << "Failed to render PDF (Graphviz 'dot' command might be missing)\n";
        }

        // Export to interactive HTML
        std::cout << "Exporting 'block_diag.html' for interactive visualization...\n";
        sp.export_spy_html("block_diag", "Block Diagonal System");
        std::cout << "Open block_diag.html in a browser for pan/zoom exploration!\n";
    }

    // Example 4: 2D Laplacian (5-point stencil)
    // Structure typically found in PDE discretization
    {
        std::cout << "\n2D Laplacian (5-point Stencil on 5x5 grid):\n";
        int N = 5;
        int n_vars = N * N;

        // Use sym_vec_pair to get both the vector for indexing and the raw MX for function
        // definition
        auto [x_vec, x_mx] = metis::sym_vec_pair("x", n_vars);
        std::vector<SymbolicScalar> eqs(n_vars);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int k = i * N + j;
                auto val = x_vec(k); // Access via Eigen vector
                // Neighbors: left, right, up, down
                if (j > 0)
                    val += x_vec(k - 1); // Left
                if (j < N - 1)
                    val += x_vec(k + 1); // Right
                if (i > 0)
                    val += x_vec(k - N); // Up
                if (i < N - 1)
                    val += x_vec(k + N); // Down
                eqs[k] = val;            // Simple linear dependency
            }
        }

        // Pass the raw MX symbol (x_mx) as the input
        metis::Function f_pde({x_mx}, {SymbolicScalar::vertcat(eqs)});
        auto sp = metis::get_jacobian_sparsity(f_pde, 0, 0);

        std::cout << "Sparsity: " << sp.n_rows() << "x" << sp.n_cols() << ", nnz=" << sp.nnz()
                  << " (density=" << sp.density() << "%)" << std::endl;
        std::cout << sp.to_string() << "\n";

        // Visualize
        std::cout << "Exporting 'laplacian_2d.pdf' and 'laplacian_2d.html'...\n";
        sp.visualize_spy("laplacian_2d");
        sp.export_spy_html("laplacian_2d", "2D Laplacian Sparsity");
        std::cout << "Open laplacian_2d.html for interactive exploration!\n";
    }

    // Example 5: NaN-Propagation Sparsity (Black-Box Detection)
    // ---------------------------------------------------------
    // When you have a black-box function (e.g., external code, non-traceable ops),
    // NaN-propagation detects sparsity by testing each input with NaN
    {
        print_section("NaN-Propagation Sparsity Detection");

        std::cout << "Black-box function: f[i] = x[i]^2 (diagonal Jacobian)\n\n";

        // Define a black-box numeric function
        auto square_fn = [](const NumericVector &x) {
            NumericVector y(x.size());
            for (int i = 0; i < x.size(); ++i) {
                y(i) = x(i) * x(i);
            }
            return y;
        };

        // Detect sparsity via NaN propagation
        auto sp = nan_propagation_sparsity(square_fn, 4, 4);

        std::cout << "Detected sparsity pattern:\n";
        std::cout << sp.to_string() << "\n";

        // Compare with symbolic detection
        std::cout << "Comparison: Symbolic sparsity should match...\n";
        auto x_sym = sym("x", 4);
        auto f_sym = x_sym * x_sym;
        auto sp_symbolic = sparsity_of_jacobian(f_sym, x_sym);

        if (sp == sp_symbolic) {
            std::cout << "✓ NaN-propagation matches symbolic sparsity!\n\n";
        } else {
            std::cout << "✗ Patterns differ (unexpected)\n\n";
        }

        // Another example: sparse coupling
        std::cout << "Sparse coupling example:\n";
        std::cout << "  f[0] = x[0] * x[1]  (depends on x0, x1)\n";
        std::cout << "  f[1] = x[2]         (depends on x2 only)\n\n";

        auto sparse_fn = [](const NumericVector &x) {
            NumericVector y(2);
            y(0) = x(0) * x(1);
            y(1) = x(2);
            return y;
        };

        auto sp_sparse = nan_propagation_sparsity(sparse_fn, 3, 2);
        std::cout << sp_sparse.to_string() << "\n";
    }
}
