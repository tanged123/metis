/**
 * @file linear_solve_policy_demo.cpp
 * @brief Demonstrate Metis linear solve backends and policy selection.
 *
 * This example shows four workflows:
 * 1. Default dense solve vs explicit dense direct backends.
 * 2. Sparse-direct solve on the same system stored as a sparse matrix.
 * 3. Iterative GMRES with a custom preconditioner hook.
 * 4. Symbolic solve with an explicit CasADi linear solver selection.
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <string>

using namespace metis;

namespace {

double residual_inf_norm(const NumericMatrix &A, const NumericMatrix &x, const NumericMatrix &b) {
    return (A * x - b).array().abs().maxCoeff();
}

void print_numeric_case(const std::string &label, const NumericMatrix &A, const NumericMatrix &b,
                        const NumericMatrix &x) {
    std::cout << label << "\n";
    std::cout << "  solution      = " << x.transpose() << "\n";
    std::cout << "  residual inf  = " << residual_inf_norm(A, x, b) << "\n\n";
}

} // namespace

int main() {
    std::cout << "=== Linear Solve Policy Demo ===\n\n";
    std::cout << std::fixed << std::setprecision(6);

    NumericMatrix A(4, 4);
    A << 10.0, 1.0, 0.0, 0.0, 1.0, 7.0, 1.0, 0.0, 0.0, 1.0, 6.0, 1.0, 0.0, 0.0, 1.0, 5.0;

    NumericVector b(4);
    b << 1.0, 2.0, 3.0, 4.0;

    std::cout << "System matrix A:\n" << A << "\n\n";
    std::cout << "Right-hand side b:\n" << b.transpose() << "\n\n";

    NumericVector x_default = solve(A, b);
    NumericVector x_lu = solve(A, b, LinearSolvePolicy::dense(DenseLinearSolver::PartialPivLU));
    NumericVector x_llt = solve(A, b, LinearSolvePolicy::dense(DenseLinearSolver::LLT));

    std::cout << "Case 1: dense direct backends\n";
    std::cout << "  default solve(A, b) uses ColPivHouseholderQR for dense numeric matrices\n\n";
    print_numeric_case("  default dense QR", A, b, x_default);
    print_numeric_case("  explicit PartialPivLU", A, b, x_lu);
    print_numeric_case("  explicit LLT (SPD only)", A, b, x_llt);

    SparseMatrix A_sparse = to_sparse(A);
    NumericVector x_sparse_default = solve(A_sparse, b);
    NumericVector x_sparse_qr =
        solve(A_sparse, b, LinearSolvePolicy::sparse_direct(SparseDirectLinearSolver::SparseQR));

    std::cout << "Case 2: sparse-direct backends\n";
    std::cout << "  sparse input defaults to SparseLU\n";
    std::cout << "  structural nonzeros in A_sparse = " << A_sparse.nonZeros() << "\n\n";
    print_numeric_case("  sparse default (SparseLU)", A, b, x_sparse_default);
    print_numeric_case("  explicit SparseQR", A, b, x_sparse_qr);

    NumericVector inv_diag = A.diagonal().cwiseInverse();
    int preconditioner_calls = 0;
    auto gmres_policy =
        LinearSolvePolicy::iterative(IterativeKrylovSolver::GMRES, IterativePreconditioner::None)
            .set_tolerance(1e-12)
            .set_max_iterations(200)
            .set_gmres_restart(8)
            .set_preconditioner_hook([&](const NumericVector &rhs) {
                ++preconditioner_calls;
                return (inv_diag.array() * rhs.array()).matrix();
            });
    NumericVector x_gmres = solve(A_sparse, b, gmres_policy);

    std::cout << "Case 3: iterative Krylov solve with a preconditioner hook\n";
    std::cout << "  backend       = GMRES\n";
    std::cout << "  restart       = 8\n";
    std::cout << "  tolerance     = 1e-12\n";
    std::cout << "  hook          = diagonal inverse apply(rhs)\n\n";
    print_numeric_case("  GMRES + custom preconditioner", A, b, x_gmres);
    std::cout << "  preconditioner calls = " << preconditioner_calls << "\n\n";

    SymbolicScalar A_mx = sym("A_sym", 2, 2);
    SymbolicScalar b_mx = sym("b_sym", 2, 1);
    SymbolicMatrix A_sym = to_eigen(A_mx);
    SymbolicVector b_sym = to_eigen(b_mx);
    auto symbolic_policy = LinearSolvePolicy();
    symbolic_policy.set_symbolic_solver("qr");

    SymbolicVector x_sym = solve(A_sym, b_sym, symbolic_policy);
    Function symbolic_solver("symbolic_linear_solve_demo", {A_mx, b_mx}, {x_sym});

    NumericMatrix A_small(2, 2);
    A_small << 2.0, 1.0, 1.0, 2.0;
    NumericVector b_small(2);
    b_small << 3.0, 3.0;
    NumericVector x_symbolic = symbolic_solver.eval(A_small, b_small);

    std::cout << "Case 4: symbolic solve with explicit CasADi linear solver selection\n";
    std::cout << "  symbolic solver = qr\n";
    std::cout << "  evaluated x     = " << x_symbolic.transpose() << "\n";
    std::cout << "  residual inf    = " << residual_inf_norm(A_small, x_symbolic, b_small)
              << "\n\n";

    std::cout << "Takeaway:\n";
    std::cout << "  - solve(A, b) keeps a simple default path\n";
    std::cout
        << "  - LinearSolvePolicy lets you switch dense, sparse-direct, and iterative backends\n";
    std::cout << "  - sparse input gets a sparse-aware default, and iterative solves can inject\n";
    std::cout << "    a custom preconditioner without changing the surrounding math code\n";

    return 0;
}
