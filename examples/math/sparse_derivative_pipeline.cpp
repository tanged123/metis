/**
 * @file sparse_derivative_pipeline.cpp
 * @brief Demonstrate sparse derivative value kernels and graph coloring reuse.
 *
 * This example builds a small trajectory-style residual with local coupling,
 * compiles sparse Jacobian/Hessian evaluators, and shows how the fixed sparsity
 * structure can be reused across multiple numeric evaluations.
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <string>
#include <vector>

using namespace metis;

namespace {

NumericMatrix dense_from_sparse_values(const SparsityPattern &sp, const NumericMatrix &values) {
    NumericMatrix dense = NumericMatrix::Zero(sp.n_rows(), sp.n_cols());
    auto [rows, cols] = sp.get_triplet();
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        dense(rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)]) = values(i);
    }
    return dense;
}

const char *mode_name(SparseJacobianMode mode) {
    switch (mode) {
    case SparseJacobianMode::Forward:
        return "forward";
    case SparseJacobianMode::Reverse:
        return "reverse";
    }
    return "unknown";
}

void print_jacobian_summary(const std::string &name, const SparseJacobianEvaluator &eval,
                            int dense_rows, int dense_cols) {
    const int dense_entries = dense_rows * dense_cols;
    std::cout << name << "\n";
    std::cout << "  dense size      : " << dense_rows << " x " << dense_cols << " = "
              << dense_entries << " entries\n";
    std::cout << "  structural nnz  : " << eval.nnz() << "\n";
    std::cout << "  forward colors  : " << eval.forward_coloring().n_colors() << "\n";
    std::cout << "  reverse colors  : " << eval.reverse_coloring().n_colors() << "\n";
    std::cout << "  preferred mode  : " << mode_name(eval.preferred_mode()) << "\n";
    std::cout << eval.sparsity().to_string() << "\n";
}

void print_hessian_summary(const std::string &name, const SparseHessianEvaluator &eval,
                           int dense_rows, int dense_cols) {
    const int dense_entries = dense_rows * dense_cols;
    std::cout << name << "\n";
    std::cout << "  dense size      : " << dense_rows << " x " << dense_cols << " = "
              << dense_entries << " entries\n";
    std::cout << "  structural nnz  : " << eval.nnz() << "\n";
    std::cout << "  star colors     : " << eval.coloring().n_colors() << "\n";
    std::cout << eval.sparsity().to_string() << "\n";
}

void evaluate_point(const std::string &label, const SparseJacobianEvaluator &ddefects_dx,
                    const SparseJacobianEvaluator &ddefects_du, const SparseHessianEvaluator &hxx,
                    const NumericMatrix &x, const NumericMatrix &u) {
    auto dx_nz = ddefects_dx.values(x, u);
    auto du_nz = ddefects_du.values(x, u);
    auto hxx_nz = hxx.values(x, u);

    NumericMatrix dx_dense = dense_from_sparse_values(ddefects_dx.sparsity(), dx_nz);
    NumericMatrix du_dense = dense_from_sparse_values(ddefects_du.sparsity(), du_nz);
    NumericMatrix hxx_dense = dense_from_sparse_values(hxx.sparsity(), hxx_nz);

    std::cout << label << "\n";
    std::cout << "  d(defects)/dx nonzero values : " << dx_nz.transpose() << "\n";
    std::cout << "  d(defects)/du nonzero values : " << du_nz.transpose() << "\n";
    std::cout << "  Hessian_xx nonzero values    : " << hxx_nz.transpose() << "\n\n";

    std::cout << "  Dense d(defects)/dx:\n" << dx_dense << "\n\n";
    std::cout << "  Dense d(defects)/du:\n" << du_dense << "\n\n";
    std::cout << "  Dense Hessian_xx:\n" << hxx_dense << "\n\n";
}

} // namespace

int main() {
    std::cout << "=== Sparse Derivative Pipeline Demo ===\n\n";

    constexpr int N = 6;
    constexpr double dt = 0.2;

    auto x = sym("x", N);
    auto u = sym("u", N - 1);

    std::vector<SymbolicScalar> defects_terms;
    defects_terms.reserve(N - 1);

    SymbolicScalar objective = 0;
    for (int k = 0; k < N - 1; ++k) {
        auto defect = x(k + 1) - x(k) - dt * (sin(x(k)) + 0.25 * u(k));
        defects_terms.push_back(defect);
        objective = objective + defect * defect + 0.05 * u(k) * u(k);
    }

    SymbolicScalar defects = SymbolicScalar::vertcat(defects_terms);
    Function terms("trajectory_terms", {x, u}, {defects, objective});

    auto ddefects_dx = sparse_jacobian(terms, 0, 0, "defects_dx_nz");
    auto ddefects_du = sparse_jacobian(terms, 0, 1, "defects_du_nz");
    auto hxx = sparse_hessian(terms, 1, 0, "objective_hxx_nz");

    std::cout << "The problem is intentionally local:\n";
    std::cout << "  defect[k] depends on x[k], x[k+1], and u[k]\n";
    std::cout << "This gives a banded state Jacobian, diagonal control Jacobian,\n";
    std::cout << "and sparse trajectory Hessian.\n\n";

    print_jacobian_summary("Sparse block: d(defects)/dx", ddefects_dx, N - 1, N);
    print_jacobian_summary("Sparse block: d(defects)/du", ddefects_du, N - 1, N - 1);
    print_hessian_summary("Sparse block: d2(objective)/dx2", hxx, N, N);

    NumericMatrix x0(N, 1);
    NumericMatrix u0(N - 1, 1);
    for (int i = 0; i < N; ++i) {
        x0(i, 0) = 0.15 * static_cast<double>(i);
        if (i < N - 1) {
            u0(i, 0) = (i % 2 == 0) ? 0.4 : -0.3;
        }
    }

    NumericMatrix x1 = x0;
    NumericMatrix u1 = u0;
    x1.array() += 0.1;
    u1.array() *= -1.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "The same sparse kernels are reused at each evaluation point.\n";
    std::cout << "Only the numeric nonzero values change.\n\n";

    evaluate_point("Evaluation A", ddefects_dx, ddefects_du, hxx, x0, u0);
    evaluate_point("Evaluation B", ddefects_dx, ddefects_du, hxx, x1, u1);

    std::cout << "Takeaway:\n";
    std::cout << "  - structure is compiled once\n";
    std::cout << "  - evaluations return only structural nonzeros\n";
    std::cout << "  - graph coloring reduces directional derivative sweeps\n";

    return 0;
}
