/**
 * @file test_sparsity.cpp
 * @brief Tests for SparsityPattern class and sparsity query functions
 */

#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/core/Sparsity.hpp>
#include <stdexcept>

using namespace metis;

namespace {

NumericMatrix dense_from_sparse_values(const SparsityPattern &sp, const NumericMatrix &values) {
    NumericMatrix dense = NumericMatrix::Zero(sp.n_rows(), sp.n_cols());
    auto [rows, cols] = sp.get_triplet();
    if (static_cast<size_t>(values.size()) != rows.size()) {
        throw std::runtime_error("dense_from_sparse_values: value count mismatch");
    }
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        dense(rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)]) = values(i);
    }
    return dense;
}

} // namespace

// ============================================================================
// SparsityPattern Construction Tests
// ============================================================================

TEST(SparsityTests, DefaultConstruction) {
    SparsityPattern sp;
    EXPECT_EQ(sp.n_rows(), 0);
    EXPECT_EQ(sp.n_cols(), 0);
    EXPECT_EQ(sp.nnz(), 0);
}

TEST(SparsityTests, FromCasadiSparsity) {
    casadi::Sparsity dense_sp = casadi::Sparsity::dense(3, 4);
    SparsityPattern sp(dense_sp);

    EXPECT_EQ(sp.n_rows(), 3);
    EXPECT_EQ(sp.n_cols(), 4);
    EXPECT_EQ(sp.nnz(), 12); // 3*4 = 12
    EXPECT_DOUBLE_EQ(sp.density(), 1.0);
}

TEST(SparsityTests, FromCasadiMX) {
    auto x = sym("x", 2, 2);
    SparsityPattern sp(x);

    EXPECT_EQ(sp.n_rows(), 2);
    EXPECT_EQ(sp.n_cols(), 2);
    EXPECT_EQ(sp.nnz(), 4);
}

// ============================================================================
// Element Access Tests
// ============================================================================

TEST(SparsityTests, HasNonzero) {
    casadi::Sparsity diag_sp = casadi::Sparsity::diag(3);
    SparsityPattern sp(diag_sp);

    EXPECT_EQ(sp.nnz(), 3);
    EXPECT_TRUE(sp.has_nz(0, 0));
    EXPECT_TRUE(sp.has_nz(1, 1));
    EXPECT_TRUE(sp.has_nz(2, 2));
    EXPECT_FALSE(sp.has_nz(0, 1));
    EXPECT_FALSE(sp.has_nz(1, 0));
}

TEST(SparsityTests, Nonzeros) {
    casadi::Sparsity diag_sp = casadi::Sparsity::diag(3);
    SparsityPattern sp(diag_sp);

    auto nz = sp.nonzeros();
    EXPECT_EQ(nz.size(), 3);

    // Diagonal elements should be present
    bool found_00 = false, found_11 = false, found_22 = false;
    for (const auto &[r, c] : nz) {
        if (r == 0 && c == 0)
            found_00 = true;
        if (r == 1 && c == 1)
            found_11 = true;
        if (r == 2 && c == 2)
            found_22 = true;
    }
    EXPECT_TRUE(found_00);
    EXPECT_TRUE(found_11);
    EXPECT_TRUE(found_22);
}

// ============================================================================
// Export Format Tests
// ============================================================================

TEST(SparsityTests, GetTriplet) {
    casadi::Sparsity diag_sp = casadi::Sparsity::diag(3);
    SparsityPattern sp(diag_sp);

    auto [rows, cols] = sp.get_triplet();
    EXPECT_EQ(rows.size(), 3);
    EXPECT_EQ(cols.size(), 3);

    // For diagonal, rows[i] == cols[i]
    for (size_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(rows[i], cols[i]);
    }
}

TEST(SparsityTests, GetCCS) {
    // 2x3 matrix with pattern:
    //   [* . *]
    //   [. * .]
    casadi::Sparsity sp = casadi::Sparsity::triplet(2, 3, {0, 1, 0}, {0, 1, 2});
    SparsityPattern pattern(sp);

    auto [col_ptr, row_idx] = pattern.get_ccs();

    // col_ptr should have n_cols + 1 = 4 elements
    EXPECT_EQ(col_ptr.size(), 4);
    EXPECT_EQ(row_idx.size(), 3); // 3 non-zeros
}

// ============================================================================
// Jacobian Sparsity Tests
// ============================================================================

TEST(SparsityTests, JacobianSparsity_Linear) {
    // f = A*x where A is 2x3
    // Jacobian df/dx = A, which is dense 2x3
    auto x = sym("x", 3);
    SymbolicScalar A = to_mx(Eigen::MatrixXd{{1, 2, 3}, {4, 5, 6}});
    auto f = SymbolicScalar::mtimes(A, x);

    auto sp = sparsity_of_jacobian(f, x);

    EXPECT_EQ(sp.n_rows(), 2);
    EXPECT_EQ(sp.n_cols(), 3);
    EXPECT_EQ(sp.nnz(), 6); // Dense
}

TEST(SparsityTests, JacobianSparsity_ElementWise) {
    // f[i] = x[i]^2, should have diagonal Jacobian
    auto x = sym("x", 3);
    auto f = x * x; // element-wise square

    auto sp = sparsity_of_jacobian(f, x);

    EXPECT_EQ(sp.n_rows(), 3);
    EXPECT_EQ(sp.n_cols(), 3);
    EXPECT_EQ(sp.nnz(), 3); // Diagonal

    // Check diagonal pattern
    EXPECT_TRUE(sp.has_nz(0, 0));
    EXPECT_TRUE(sp.has_nz(1, 1));
    EXPECT_TRUE(sp.has_nz(2, 2));
    EXPECT_FALSE(sp.has_nz(0, 1));
}

TEST(SparsityTests, JacobianSparsity_Scalar) {
    // f = x*y, Jacobian [y, x]
    auto x = sym("x");
    auto y = sym("y");
    auto vars = SymbolicScalar::vertcat({x, y});
    auto f = x * y;

    auto sp = sparsity_of_jacobian(f, vars);

    EXPECT_EQ(sp.n_rows(), 1);
    EXPECT_EQ(sp.n_cols(), 2);
    EXPECT_EQ(sp.nnz(), 2); // Both x and y influence f
}

// ============================================================================
// Hessian Sparsity Tests
// ============================================================================

TEST(SparsityTests, HessianSparsity_Quadratic) {
    // f = x^2 + y^2 + z^2
    // Hessian = diag([2, 2, 2])
    auto x = sym("x");
    auto y = sym("y");
    auto z = sym("z");
    auto vars = SymbolicScalar::vertcat({x, y, z});
    auto f = x * x + y * y + z * z;

    auto sp = sparsity_of_hessian(f, vars);

    EXPECT_EQ(sp.n_rows(), 3);
    EXPECT_EQ(sp.n_cols(), 3);
    EXPECT_EQ(sp.nnz(), 3); // Diagonal

    EXPECT_TRUE(sp.has_nz(0, 0));
    EXPECT_TRUE(sp.has_nz(1, 1));
    EXPECT_TRUE(sp.has_nz(2, 2));
    EXPECT_FALSE(sp.has_nz(0, 1));
}

TEST(SparsityTests, HessianSparsity_CrossTerms) {
    // f = x*y
    // Hessian = [[0, 1], [1, 0]]
    auto x = sym("x");
    auto y = sym("y");
    auto vars = SymbolicScalar::vertcat({x, y});
    auto f = x * y;

    auto sp = sparsity_of_hessian(f, vars);

    EXPECT_EQ(sp.n_rows(), 2);
    EXPECT_EQ(sp.n_cols(), 2);
    EXPECT_EQ(sp.nnz(), 2); // Off-diagonal

    EXPECT_FALSE(sp.has_nz(0, 0)); // d²f/dx² = 0
    EXPECT_FALSE(sp.has_nz(1, 1)); // d²f/dy² = 0
    EXPECT_TRUE(sp.has_nz(0, 1));  // d²f/dxdy = 1
    EXPECT_TRUE(sp.has_nz(1, 0));  // d²f/dydx = 1
}

// ============================================================================
// Visualization Tests
// ============================================================================

TEST(SparsityTests, ToString_Diagonal) {
    casadi::Sparsity diag_sp = casadi::Sparsity::diag(3);
    SparsityPattern sp(diag_sp);

    std::string str = sp.to_string();

    // Should contain dimensions and nnz
    EXPECT_NE(str.find("3x3"), std::string::npos);
    EXPECT_NE(str.find("nnz=3"), std::string::npos);

    // Should have some structure markers
    EXPECT_NE(str.find("*"), std::string::npos);
    EXPECT_NE(str.find("."), std::string::npos);
}

TEST(SparsityTests, ToString_Large) {
    // Test truncation for large matrices
    casadi::Sparsity large_sp = casadi::Sparsity::dense(100, 100);
    SparsityPattern sp(large_sp);

    std::string str = sp.to_string(10, 10); // Max 10x10 display

    // Should indicate truncation
    EXPECT_NE(str.find("100x100"), std::string::npos);
}

// ============================================================================
// Equality Tests
// ============================================================================

TEST(SparsityTests, Equality) {
    casadi::Sparsity sp1 = casadi::Sparsity::diag(3);
    casadi::Sparsity sp2 = casadi::Sparsity::diag(3);
    casadi::Sparsity sp3 = casadi::Sparsity::dense(3, 3);

    SparsityPattern p1(sp1);
    SparsityPattern p2(sp2);
    SparsityPattern p3(sp3);

    EXPECT_EQ(p1, p2);
    EXPECT_NE(p1, p3);
}

// ============================================================================
// Integration with metis::Function
// ============================================================================

TEST(SparsityTests, FunctionJacobianSparsity) {
    // Create a function f(x) = [x[0]^2, x[1]^2, x[2]^2]
    // Jacobian should be diagonal
    auto x = sym("x", 3);
    auto f = x * x;

    Function fn("square", {x}, {f});

    auto sp = get_jacobian_sparsity(fn);

    EXPECT_EQ(sp.n_rows(), 3);
    EXPECT_EQ(sp.n_cols(), 3);
    EXPECT_EQ(sp.nnz(), 3); // Diagonal
}

TEST(SparsityTests, FunctionJacobianSparsitySelectedInputOutput) {
    auto x = sym("x", 2, 1);
    auto p = sym("p", 2, 1);
    auto y0 = x + p;
    auto y1 = SymbolicScalar::vertcat({x(0) * p(0), x(1) + p(1) * p(1)});

    Function fn("mixed_io", {x, p}, {y0, y1});
    auto sp = get_jacobian_sparsity(fn, 1, 1);

    EXPECT_EQ(sp.n_rows(), 2);
    EXPECT_EQ(sp.n_cols(), 2);
    EXPECT_EQ(sp.nnz(), 2);
    EXPECT_TRUE(sp.has_nz(0, 0));
    EXPECT_TRUE(sp.has_nz(1, 1));
    EXPECT_FALSE(sp.has_nz(0, 1));
    EXPECT_FALSE(sp.has_nz(1, 0));
}

TEST(SparsityTests, SparseJacobianExpressionMatchesDenseJacobian) {
    auto x = sym("x", 3);
    auto f = SymbolicScalar::vertcat({x(0) * x(0) + x(1), 3.0 * x(2)});

    auto sparse = sparse_jacobian(f, x);
    Function dense_jac("dense_jac_expr", {x}, {SymbolicScalar::jacobian(f, x)});

    NumericMatrix xval(3, 1);
    xval << 2.0, -1.5, 4.0;

    auto sparse_values = sparse.values(xval);
    auto dense_values = dense_jac.eval(xval);
    auto reconstructed = dense_from_sparse_values(sparse.sparsity(), sparse_values);

    EXPECT_EQ(sparse.nnz(), 3);
    EXPECT_EQ(sparse_values.rows(), sparse.nnz());
    EXPECT_EQ(sparse_values.cols(), 1);
    EXPECT_LT(sparse.forward_coloring().n_colors(), sparse.forward_coloring().n_entries());
    EXPECT_LT(sparse.reverse_coloring().n_colors(), sparse.reverse_coloring().n_entries());
    EXPECT_TRUE(reconstructed.isApprox(dense_values, 1e-12));
}

TEST(SparsityTests, SparseJacobianFunctionMatchesDenseJacobian) {
    auto x = sym("x", 2, 1);
    auto p = sym("p", 2, 1);
    auto y0 = x + p;
    auto y1 = SymbolicScalar::vertcat({x(0) * p(0), x(1) + p(1) * p(1)});

    Function fn("mixed_sparse_jac", {x, p}, {y0, y1});
    Function dense_jac("dense_jac_fn", {x, p}, {SymbolicScalar::jacobian(y1, p)});
    auto sparse = sparse_jacobian(fn, 1, 1);

    NumericMatrix xval(2, 1);
    xval << 3.0, -2.0;
    NumericMatrix pval(2, 1);
    pval << 4.0, 5.0;

    auto sparse_values = sparse.values(xval, pval);
    auto dense_values = dense_jac.eval(xval, pval);
    auto reconstructed = dense_from_sparse_values(sparse.sparsity(), sparse_values);

    EXPECT_EQ(sparse.sparsity().nnz(), 2);
    EXPECT_EQ(sparse.preferred_coloring().colorvec().size(), 2u);
    EXPECT_TRUE(reconstructed.isApprox(dense_values, 1e-12));
}

TEST(SparsityTests, SparseHessianMatchesDenseHessian) {
    auto x = sym("x", 3);
    auto f = x(0) * x(1) + x(1) * x(1) + 0.5 * x(2) * x(2);

    auto sparse = sparse_hessian(f, x);
    Function dense_hess("dense_hess_expr", {x}, {SymbolicScalar::hessian(f, x)});

    NumericMatrix xval(3, 1);
    xval << 1.0, -2.0, 3.0;

    auto sparse_values = sparse.values(xval);
    auto dense_values = dense_hess.eval(xval);
    auto reconstructed = dense_from_sparse_values(sparse.sparsity(), sparse_values);

    EXPECT_EQ(sparse.nnz(), 4);
    EXPECT_EQ(sparse_values.rows(), sparse.nnz());
    EXPECT_EQ(sparse_values.cols(), 1);
    EXPECT_LT(sparse.coloring().n_colors(), sparse.coloring().n_entries());
    EXPECT_TRUE(reconstructed.isApprox(dense_values, 1e-12));
}

// ============================================================================
// NaN-Propagation Sparsity Tests
// ============================================================================

TEST(SparsityTests, NaNSparsity_Diagonal) {
    // f[i] = x[i]^2 -> diagonal Jacobian
    auto square_fn = [](const NumericVector &x) {
        NumericVector y(x.size());
        for (int i = 0; i < x.size(); ++i) {
            y(i) = x(i) * x(i);
        }
        return y;
    };

    auto sp = nan_propagation_sparsity(square_fn, 3, 3);

    EXPECT_EQ(sp.n_rows(), 3);
    EXPECT_EQ(sp.n_cols(), 3);
    EXPECT_EQ(sp.nnz(), 3); // Diagonal

    // Check diagonal pattern
    EXPECT_TRUE(sp.has_nz(0, 0));
    EXPECT_TRUE(sp.has_nz(1, 1));
    EXPECT_TRUE(sp.has_nz(2, 2));
    EXPECT_FALSE(sp.has_nz(0, 1));
    EXPECT_FALSE(sp.has_nz(1, 0));
}

TEST(SparsityTests, NaNSparsity_Dense) {
    // f[i] = sum(x) -> each output depends on all inputs
    auto sum_fn = [](const NumericVector &x) {
        NumericVector y(2);
        double s = x.sum();
        y(0) = s;
        y(1) = s * 2;
        return y;
    };

    auto sp = nan_propagation_sparsity(sum_fn, 3, 2);

    EXPECT_EQ(sp.n_rows(), 2);
    EXPECT_EQ(sp.n_cols(), 3);
    EXPECT_EQ(sp.nnz(), 6); // Dense: all outputs depend on all inputs

    // All entries should be non-zero
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_TRUE(sp.has_nz(i, j));
        }
    }
}

TEST(SparsityTests, NaNSparsity_Sparse) {
    // f[0] = x[0] * x[1]  -> depends on x[0], x[1]
    // f[1] = x[2]         -> depends only on x[2]
    auto sparse_fn = [](const NumericVector &x) {
        NumericVector y(2);
        y(0) = x(0) * x(1);
        y(1) = x(2);
        return y;
    };

    auto sp = nan_propagation_sparsity(sparse_fn, 3, 2);

    EXPECT_EQ(sp.n_rows(), 2);
    EXPECT_EQ(sp.n_cols(), 3);
    EXPECT_EQ(sp.nnz(), 3);

    // f[0] depends on x[0], x[1]
    EXPECT_TRUE(sp.has_nz(0, 0));
    EXPECT_TRUE(sp.has_nz(0, 1));
    EXPECT_FALSE(sp.has_nz(0, 2));

    // f[1] depends only on x[2]
    EXPECT_FALSE(sp.has_nz(1, 0));
    EXPECT_FALSE(sp.has_nz(1, 1));
    EXPECT_TRUE(sp.has_nz(1, 2));
}

TEST(SparsityTests, NaNSparsity_MatchesSymbolic) {
    // Create a symbolic function and compare sparsity patterns
    auto x = sym("x", 3);
    auto f = x * x; // Element-wise square, diagonal Jacobian

    // Get symbolic sparsity
    auto symbolic_sp = sparsity_of_jacobian(f, x);

    // Create metis::Function and get NaN-propagation sparsity
    Function fn("square", {x}, {f});
    auto nan_sp = nan_propagation_sparsity(fn);

    // They should match
    EXPECT_EQ(symbolic_sp.n_rows(), nan_sp.n_rows());
    EXPECT_EQ(symbolic_sp.n_cols(), nan_sp.n_cols());
    EXPECT_EQ(symbolic_sp.nnz(), nan_sp.nnz());
    EXPECT_EQ(symbolic_sp, nan_sp);
}

TEST(SparsityTests, NaNSparsity_WithReferencePoint) {
    // Test with a custom reference point
    auto fn = [](const NumericVector &x) {
        NumericVector y(2);
        y(0) = x(0) + x(1);
        y(1) = x(1) + x(2);
        return y;
    };

    NaNSparsityOptions opts;
    opts.reference_point = NumericVector{{1.0, 2.0, 3.0}};

    auto sp = nan_propagation_sparsity(fn, 3, 2, opts);

    EXPECT_EQ(sp.n_rows(), 2);
    EXPECT_EQ(sp.n_cols(), 3);
    EXPECT_EQ(sp.nnz(), 4);

    // f[0] = x[0] + x[1]
    EXPECT_TRUE(sp.has_nz(0, 0));
    EXPECT_TRUE(sp.has_nz(0, 1));
    EXPECT_FALSE(sp.has_nz(0, 2));

    // f[1] = x[1] + x[2]
    EXPECT_FALSE(sp.has_nz(1, 0));
    EXPECT_TRUE(sp.has_nz(1, 1));
    EXPECT_TRUE(sp.has_nz(1, 2));
}
