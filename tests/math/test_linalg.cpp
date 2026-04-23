#include "../utils/TestUtils.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisIO.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Linalg.hpp>

template <typename Scalar> void test_linalg_ops() {
    using Matrix = metis::MetisMatrix<Scalar>;
    using Vector = metis::MetisVector<Scalar>;

    // Test solve
    // A = [[2, 1], [1, 2]]
    // b = [3, 3]
    // x = [1, 1]
    Matrix A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;
    Vector b(2);
    b(0) = 3.0;
    b(1) = 3.0;

    auto x = metis::solve(A, b);

    // Test norm
    Vector v(2);
    v(0) = 3.0;
    v(1) = 4.0;
    auto n = metis::norm(v);

    // Test outer
    Vector v1(2);
    v1(0) = 1.0;
    v1(1) = 2.0;
    Vector v2(2);
    v2(0) = 3.0;
    v2(1) = 4.0;                   // v2 = [3, 4]
    auto M = metis::outer(v1, v2); // [[3, 4], [6, 8]]

    // Test dot
    auto d = metis::dot(v, v); // 3*3 + 4*4 = 25

    // Test cross (3D)
    Vector c1(3);
    c1 << 1.0, 0.0, 0.0;
    Vector c2(3);
    c2 << 0.0, 1.0, 0.0;
    auto c3 = metis::cross(c1, c2); // [0, 0, 1]

    // Test inner
    auto i_prod = metis::inner(v, v); // 25.0

    // Test pinv
    // A singular = [[1, 1], [2, 2]]
    Matrix A_sing(2, 2);
    A_sing << 1.0, 1.0, 2.0, 2.0;
    auto A_pinv = metis::pinv(A_sing);

    // Test extended norms
    Vector v_norm(3);
    v_norm << -1.0, 2.0, -3.0;

    auto n_1 = metis::norm(v_norm, metis::NormType::L1);    // 1+2+3 = 6
    auto n_inf = metis::norm(v_norm, metis::NormType::Inf); // 3

    // Existing inv / det tests
    auto A_inv = metis::inv(A); // inv([[2, 1], [1, 2]]) = 1/3 * [[2, -1], [-1, 2]]
    auto A_det = metis::det(A); // 4 - 1 = 3

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_NEAR(x(0), 1.0, 1e-6);
        EXPECT_NEAR(x(1), 1.0, 1e-6);
        EXPECT_DOUBLE_EQ(n, 5.0);
        EXPECT_DOUBLE_EQ(M(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(M(1, 1), 8.0);

        EXPECT_DOUBLE_EQ(d, 25.0);

        EXPECT_DOUBLE_EQ(c3(0), 0.0);
        EXPECT_DOUBLE_EQ(c3(1), 0.0);
        EXPECT_DOUBLE_EQ(c3(2), 1.0);

        EXPECT_DOUBLE_EQ(i_prod, 25.0);

        // Pinv of [[1, 1], [2, 2]]
        // SVD based, roughly [[0.1, 0.2], [0.1, 0.2]]
        // Check A * pinv * A = A
        auto recon = A_sing * A_pinv * A_sing;
        EXPECT_TRUE(recon.isApprox(A_sing, 1e-5));

        EXPECT_DOUBLE_EQ(n_1, 6.0);
        EXPECT_DOUBLE_EQ(n_inf, 3.0);

        EXPECT_NEAR(A_det, 3.0, 1e-9);
        EXPECT_NEAR(A_inv(0, 0), 2.0 / 3.0, 1e-9);
    } else {
        auto x_eval = metis::eval(x);
        EXPECT_NEAR(x_eval(0), 1.0, 1e-6);
        EXPECT_NEAR(x_eval(1), 1.0, 1e-6);

        EXPECT_DOUBLE_EQ(metis::eval(n), 5.0);

        auto M_eval = metis::eval(M);
        EXPECT_DOUBLE_EQ(M_eval(0, 0), 3.0);
        EXPECT_DOUBLE_EQ(M_eval(1, 1), 8.0);

        EXPECT_DOUBLE_EQ(metis::eval(d), 25.0);

        auto c3_eval = metis::eval(c3);
        EXPECT_NEAR(c3_eval(0), 0.0, 1e-9);
        EXPECT_NEAR(c3_eval(1), 0.0, 1e-9);
        EXPECT_NEAR(c3_eval(2), 1.0, 1e-9);

        EXPECT_DOUBLE_EQ(metis::eval(i_prod), 25.0);

        // Pinv test symbolic
        auto A_pinv_inv = metis::pinv(A);
        auto A_pinv_eval = metis::eval(A_pinv_inv);
        EXPECT_NEAR(A_pinv_eval(0, 0), 2.0 / 3.0, 1e-6);

        EXPECT_DOUBLE_EQ(metis::eval(n_1), 6.0);
        EXPECT_DOUBLE_EQ(metis::eval(n_inf), 3.0);

        auto A_inv_eval = metis::eval(A_inv);
        if (std::abs(A_inv_eval(0, 0) - (2.0 / 3.0)) > 1e-9) {
            metis::print("A_inv (Symbolic)", A_inv);
            metis::print("A_inv (Evaluated)", A_inv_eval);
        }
        EXPECT_NEAR(A_inv_eval(0, 0), 2.0 / 3.0, 1e-9);
    }
}

TEST(LinalgTests, Numeric) { test_linalg_ops<double>(); }

TEST(LinalgTests, Symbolic) { test_linalg_ops<metis::SymbolicScalar>(); }

TEST(LinalgTests, SolveDensePolicyBackends) {
    metis::NumericMatrix A(3, 3);
    A << 4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0;

    metis::NumericMatrix B(3, 2);
    B << 1.0, 2.0, 0.0, -1.0, 3.0, 1.0;

    metis::NumericMatrix x_qr = metis::solve(
        A, B, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::ColPivHouseholderQR));
    metis::NumericMatrix x_lu =
        metis::solve(A, B, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::PartialPivLU));
    metis::NumericMatrix x_llt =
        metis::solve(A, B, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::LLT));
    metis::NumericMatrix x_ldlt =
        metis::solve(A, B, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::LDLT));

    EXPECT_TRUE((A * x_qr).isApprox(B, 1e-10));
    EXPECT_TRUE((A * x_lu).isApprox(B, 1e-10));
    EXPECT_TRUE((A * x_llt).isApprox(B, 1e-10));
    EXPECT_TRUE((A * x_ldlt).isApprox(B, 1e-10));
}

TEST(LinalgTests, SolveSparseDirectBackends) {
    metis::NumericMatrix dense(3, 3);
    dense << 4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0;
    metis::SparseMatrix sparse = metis::to_sparse(dense);

    metis::NumericVector b(3);
    b << 1.0, 0.0, 3.0;

    metis::NumericVector x_default = metis::solve(sparse, b);
    metis::NumericVector x_sparse_lu = metis::solve(
        sparse, b,
        metis::LinearSolvePolicy::sparse_direct(metis::SparseDirectLinearSolver::SparseLU));
    metis::NumericVector x_sparse_qr = metis::solve(
        sparse, b,
        metis::LinearSolvePolicy::sparse_direct(metis::SparseDirectLinearSolver::SparseQR));
    metis::NumericVector x_sparse_ldlt = metis::solve(
        sparse, b,
        metis::LinearSolvePolicy::sparse_direct(metis::SparseDirectLinearSolver::SimplicialLDLT));
    metis::NumericVector x_dense_input_sparse_policy = metis::solve(
        dense, b,
        metis::LinearSolvePolicy::sparse_direct(metis::SparseDirectLinearSolver::SparseLU));

    EXPECT_TRUE((dense * x_default).isApprox(b, 1e-10));
    EXPECT_TRUE((dense * x_sparse_lu).isApprox(b, 1e-10));
    EXPECT_TRUE((dense * x_sparse_qr).isApprox(b, 1e-10));
    EXPECT_TRUE((dense * x_sparse_ldlt).isApprox(b, 1e-10));
    EXPECT_TRUE((dense * x_dense_input_sparse_policy).isApprox(b, 1e-10));
}

TEST(LinalgTests, SolveIterativeBackendsAndPreconditionerHook) {
    metis::NumericMatrix dense(4, 4);
    dense << 10.0, 1.0, 0.0, 0.0, 1.0, 7.0, 1.0, 0.0, 0.0, 1.0, 6.0, 1.0, 0.0, 0.0, 1.0, 5.0;
    metis::SparseMatrix sparse = metis::to_sparse(dense);

    metis::NumericVector b(4);
    b << 1.0, 2.0, 3.0, 4.0;

    auto bicg_policy = metis::LinearSolvePolicy::iterative(metis::IterativeKrylovSolver::BiCGSTAB,
                                                           metis::IterativePreconditioner::Diagonal)
                           .set_tolerance(1e-12)
                           .set_max_iterations(200);
    metis::NumericVector x_bicg = metis::solve(sparse, b, bicg_policy);
    EXPECT_TRUE((dense * x_bicg).isApprox(b, 1e-8));

    metis::NumericVector inv_diag = dense.diagonal().cwiseInverse();
    int preconditioner_calls = 0;
    auto gmres_policy = metis::LinearSolvePolicy::iterative(metis::IterativeKrylovSolver::GMRES,
                                                            metis::IterativePreconditioner::None)
                            .set_tolerance(1e-12)
                            .set_max_iterations(200)
                            .set_gmres_restart(8)
                            .set_preconditioner_hook([&](const metis::NumericVector &rhs) {
                                ++preconditioner_calls;
                                return (inv_diag.array() * rhs.array()).matrix();
                            });
    metis::NumericVector x_gmres = metis::solve(sparse, b, gmres_policy);
    EXPECT_GT(preconditioner_calls, 0);
    EXPECT_TRUE((dense * x_gmres).isApprox(b, 1e-8));
}

TEST(LinalgTests, SolveSymbolicPolicyAndValidationErrors) {
    metis::SymbolicMatrix A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 2.0;
    metis::SymbolicVector b(2);
    b(0) = 3.0;
    b(1) = 3.0;

    auto symbolic_policy = metis::LinearSolvePolicy();
    symbolic_policy.set_symbolic_solver("qr");
    auto x = metis::solve(A, b, symbolic_policy);
    auto x_eval = metis::eval(x);
    EXPECT_NEAR(x_eval(0), 1.0, 1e-10);
    EXPECT_NEAR(x_eval(1), 1.0, 1e-10);

    auto bad_symbolic_policy = metis::LinearSolvePolicy();
    bad_symbolic_policy.symbolic_options["mock_option"] = 1;
    EXPECT_THROW(metis::solve(A, b, bad_symbolic_policy), metis::InvalidArgument);

    auto bad_iterative = metis::LinearSolvePolicy::iterative().set_max_iterations(0);
    metis::NumericMatrix dense(2, 2);
    dense << 2.0, 1.0, 1.0, 2.0;
    metis::NumericVector rhs(2);
    rhs << 3.0, 3.0;
    EXPECT_THROW(metis::solve(dense, rhs, bad_iterative), metis::InvalidArgument);
}

TEST(LinalgTests, CoverageEdges) {
    // 1. Empty to_mx coverage
    metis::NumericMatrix empty(0, 0);
    casadi::MX empty_mx = metis::to_mx(empty);
    EXPECT_TRUE(empty_mx.is_empty());

    // 2. Numeric Norm Edges
    metis::NumericVector v(3);
    v << 1.0, 2.0, 2.0; // norm = 3

    // Frobenius (same as L2 for vectors)
    EXPECT_DOUBLE_EQ(metis::norm(v, metis::NormType::Frobenius), 3.0);

    // Default (invalid enum)
    EXPECT_DOUBLE_EQ(metis::norm(v, static_cast<metis::NormType>(999)), 3.0);

    // 3. Symbolic Norm Edges (Default branch)
    metis::SymbolicVector vs = metis::as_vector(metis::to_mx(v));
    auto n_def = metis::norm(vs, static_cast<metis::NormType>(999));
    EXPECT_DOUBLE_EQ(metis::eval(n_def), 3.0);
}

namespace {

void expect_eigen_residual(const metis::NumericMatrix &A, const metis::NumericVector &eigenvalues,
                           const metis::NumericMatrix &eigenvectors, double tol) {
    EXPECT_TRUE((A * eigenvectors).isApprox(eigenvectors * eigenvalues.asDiagonal(), tol));
}

std::array<double, 6> pack_symmetric_lower_column_major(const metis::NumericMatrix &A) {
    return {A(0, 0), A(1, 0), A(2, 0), A(1, 1), A(2, 1), A(2, 2)};
}

metis::NumericMatrix unpack_symmetric_lower_column_major(const std::array<double, 6> &packed) {
    metis::NumericMatrix A(3, 3);
    A << packed[0], packed[1], packed[2], packed[1], packed[3], packed[4], packed[2], packed[4],
        packed[5];
    return A;
}

} // namespace

TEST(LinalgTests, EigSymmetricNumeric) {
    metis::NumericMatrix A(2, 2);
    A << 2.0, 1.0, 1.0, 2.0;

    auto decomp = metis::eig_symmetric(A);

    ASSERT_EQ(decomp.eigenvalues.size(), 2);
    ASSERT_EQ(decomp.eigenvectors.rows(), 2);
    ASSERT_EQ(decomp.eigenvectors.cols(), 2);
    EXPECT_NEAR(decomp.eigenvalues(0), 1.0, 1e-12);
    EXPECT_NEAR(decomp.eigenvalues(1), 3.0, 1e-12);
    expect_eigen_residual(A, decomp.eigenvalues, decomp.eigenvectors, 1e-12);
    EXPECT_TRUE((decomp.eigenvectors.transpose() * decomp.eigenvectors)
                    .isApprox(metis::NumericMatrix::Identity(2, 2), 1e-12));
}

TEST(LinalgTests, EigNumericRealSpectrum) {
    metis::NumericMatrix A(2, 2);
    A << 1.0, 1.0, 0.0, 2.0;

    auto decomp = metis::eig(A);

    ASSERT_EQ(decomp.eigenvalues.size(), 2);
    EXPECT_NEAR(decomp.eigenvalues(0), 1.0, 1e-12);
    EXPECT_NEAR(decomp.eigenvalues(1), 2.0, 1e-12);
    expect_eigen_residual(A, decomp.eigenvalues, decomp.eigenvectors, 1e-12);
}

TEST(LinalgTests, EigRejectsComplexSpectrum) {
    metis::NumericMatrix A(2, 2);
    A << 0.0, -1.0, 1.0, 0.0;

    EXPECT_THROW(metis::eig(A), metis::InvalidArgument);
}

TEST(LinalgTests, EigSymmetricRejectsNonsymmetricNumeric) {
    metis::NumericMatrix A(2, 2);
    A << 1.0, 2.0, 0.0, 1.0;

    EXPECT_THROW(metis::eig_symmetric(A), metis::InvalidArgument);
}

TEST(LinalgTests, EigSymmetricSymbolic3x3) {
    auto t = metis::sym("t");
    metis::SymbolicMatrix A(3, 3);
    A << 4.0, t, 0.0, t, 2.0, 0.0, 0.0, 0.0, 1.0;

    auto decomp = metis::eig_symmetric(A);
    metis::Function f({t}, {metis::to_mx(decomp.eigenvalues), metis::to_mx(decomp.eigenvectors)});
    auto outputs = f(0.5);

    ASSERT_EQ(outputs.size(), 2);
    metis::NumericVector eigenvalues = outputs[0].col(0);
    const metis::NumericMatrix eigenvectors = outputs[1];

    EXPECT_NEAR(eigenvalues(0), 1.0, 1e-9);
    EXPECT_NEAR(eigenvalues(1), 3.0 - 0.5 * std::sqrt(5.0), 1e-9);
    EXPECT_NEAR(eigenvalues(2), 3.0 + 0.5 * std::sqrt(5.0), 1e-9);

    metis::NumericMatrix A_eval(3, 3);
    A_eval << 4.0, 0.5, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 1.0;
    expect_eigen_residual(A_eval, eigenvalues, eigenvectors, 1e-8);
    EXPECT_TRUE((eigenvectors.transpose() * eigenvectors)
                    .isApprox(metis::NumericMatrix::Identity(3, 3), 1e-8));
}

TEST(LinalgTests, EigSymbolicRejectsGeneralMatrices) {
    auto A_mx = metis::sym("A", 2, 2);
    auto A = metis::to_eigen(A_mx);

    EXPECT_THROW(metis::eig(A), metis::InvalidArgument);
}

TEST(LinalgTests, InvSymmetric3x3ExplicitPackedOrderMatchesInv) {
    const std::array<metis::NumericMatrix, 4> cases = [] {
        std::array<metis::NumericMatrix, 4> mats;

        mats[0].resize(3, 3);
        mats[0] << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

        mats[1].resize(3, 3);
        mats[1] << 4.0, 1.0, 1.5, 1.0, 5.0, 2.0, 1.5, 2.0, 6.0;

        mats[2].resize(3, 3);
        mats[2] << 7.0, -2.5, 3.0, -2.5, 8.0, 1.5, 3.0, 1.5, 9.0;

        mats[3].resize(3, 3);
        mats[3] << 3.5, -0.4, 0.8, -0.4, 2.2, -0.6, 0.8, -0.6, 4.1;

        return mats;
    }();

    for (const auto &A : cases) {
        const auto A_inv = metis::inv(A);
        const auto packed_expected = pack_symmetric_lower_column_major(A_inv);

        const auto packed_actual_tuple =
            metis::inv_symmetric_3x3_explicit(A(0, 0), A(1, 1), A(2, 2), A(0, 1), A(1, 2), A(0, 2));
        const std::array<double, 6> packed_actual = {
            std::get<0>(packed_actual_tuple), std::get<1>(packed_actual_tuple),
            std::get<2>(packed_actual_tuple), std::get<3>(packed_actual_tuple),
            std::get<4>(packed_actual_tuple), std::get<5>(packed_actual_tuple)};

        for (std::size_t i = 0; i < packed_actual.size(); ++i) {
            EXPECT_NEAR(packed_actual[i], packed_expected[i], 1e-12);
        }

        EXPECT_TRUE(unpack_symmetric_lower_column_major(packed_actual).isApprox(A_inv, 1e-12));
    }
}

// =============================================================================
// Sparse Matrix Tests
// =============================================================================

TEST(LinalgTests, SparseFromTriplets) {
    std::vector<metis::SparseTriplet> triplets;
    triplets.emplace_back(0, 0, 1.0);
    triplets.emplace_back(1, 1, 2.0);
    triplets.emplace_back(2, 2, 3.0);

    auto sp = metis::sparse_from_triplets(3, 3, triplets);

    EXPECT_EQ(sp.rows(), 3);
    EXPECT_EQ(sp.cols(), 3);
    EXPECT_EQ(sp.nonZeros(), 3);
    EXPECT_DOUBLE_EQ(sp.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(sp.coeff(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(sp.coeff(2, 2), 3.0);
    EXPECT_DOUBLE_EQ(sp.coeff(0, 1), 0.0);
}

TEST(LinalgTests, ToSparse) {
    metis::NumericMatrix dense(3, 3);
    dense << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;

    auto sp = metis::to_sparse(dense);

    EXPECT_EQ(sp.nonZeros(), 3);
    EXPECT_DOUBLE_EQ(sp.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(sp.coeff(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(sp.coeff(2, 2), 3.0);
}

TEST(LinalgTests, ToSparseWithTolerance) {
    metis::NumericMatrix dense(2, 2);
    dense << 1.0, 1e-10, 1e-10, 1.0;

    // With tol=0, small values are kept
    auto sp1 = metis::to_sparse(dense, 0.0);
    EXPECT_EQ(sp1.nonZeros(), 4);

    // With tol=1e-9, small values are ignored
    auto sp2 = metis::to_sparse(dense, 1e-9);
    EXPECT_EQ(sp2.nonZeros(), 2);
}

TEST(LinalgTests, ToDense) {
    std::vector<metis::SparseTriplet> triplets;
    triplets.emplace_back(0, 0, 5.0);
    triplets.emplace_back(1, 1, 10.0);

    auto sp = metis::sparse_from_triplets(2, 2, triplets);
    auto dense = metis::to_dense(sp);

    EXPECT_EQ(dense.rows(), 2);
    EXPECT_EQ(dense.cols(), 2);
    EXPECT_DOUBLE_EQ(dense(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(dense(1, 1), 10.0);
    EXPECT_DOUBLE_EQ(dense(0, 1), 0.0);
}

TEST(LinalgTests, SparseIdentity) {
    auto I = metis::sparse_identity(4);

    EXPECT_EQ(I.rows(), 4);
    EXPECT_EQ(I.cols(), 4);
    EXPECT_EQ(I.nonZeros(), 4);

    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(I.coeff(i, i), 1.0);
    }
}

// Regression tests: metis::solve on fixed-size inputs must preserve the
// compile-time shape across every DenseLinearSolver policy. Previously LLT and
// LDLT hard-coded the dynamic NumericMatrix for the Eigen solver template
// argument, which made the switch branches return inconsistent types and broke
// `auto` deduction for callers using Mat3/Vec3 etc.
TEST(LinalgTests, SolveFixedSizeColPivHouseholderQR) {
    metis::Mat3<double> A = metis::Mat3<double>::Identity() * 4.0;
    metis::Vec3<double> b;
    b << 1.0, 2.0, 3.0;
    auto x = metis::solve(
        A, b, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::ColPivHouseholderQR));
    static_assert(std::is_same_v<decltype(x), metis::Vec3<double>>,
                  "ColPivHouseholderQR solve should preserve fixed-size type");
    EXPECT_NEAR(x(0), 0.25, 1e-12);
    EXPECT_NEAR(x(1), 0.50, 1e-12);
    EXPECT_NEAR(x(2), 0.75, 1e-12);
}

TEST(LinalgTests, SolveFixedSizePartialPivLU) {
    metis::Mat3<double> A = metis::Mat3<double>::Identity() * 4.0;
    metis::Vec3<double> b;
    b << 1.0, 2.0, 3.0;
    auto x =
        metis::solve(A, b, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::PartialPivLU));
    static_assert(std::is_same_v<decltype(x), metis::Vec3<double>>,
                  "PartialPivLU solve should preserve fixed-size type");
    EXPECT_NEAR(x(0), 0.25, 1e-12);
    EXPECT_NEAR(x(1), 0.50, 1e-12);
    EXPECT_NEAR(x(2), 0.75, 1e-12);
}

TEST(LinalgTests, SolveFixedSizeFullPivLU) {
    metis::Mat3<double> A = metis::Mat3<double>::Identity() * 4.0;
    metis::Vec3<double> b;
    b << 1.0, 2.0, 3.0;
    auto x =
        metis::solve(A, b, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::FullPivLU));
    static_assert(std::is_same_v<decltype(x), metis::Vec3<double>>,
                  "FullPivLU solve should preserve fixed-size type");
    EXPECT_NEAR(x(0), 0.25, 1e-12);
    EXPECT_NEAR(x(1), 0.50, 1e-12);
    EXPECT_NEAR(x(2), 0.75, 1e-12);
}

TEST(LinalgTests, SolveFixedSizeLLT) {
    metis::Mat3<double> A = metis::Mat3<double>::Identity() * 4.0;
    metis::Vec3<double> b;
    b << 1.0, 2.0, 3.0;
    auto x = metis::solve(A, b, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::LLT));
    static_assert(std::is_same_v<decltype(x), metis::Vec3<double>>,
                  "LLT solve should preserve fixed-size type");
    EXPECT_NEAR(x(0), 0.25, 1e-12);
    EXPECT_NEAR(x(1), 0.50, 1e-12);
    EXPECT_NEAR(x(2), 0.75, 1e-12);
}

TEST(LinalgTests, SolveFixedSizeLDLT) {
    metis::Mat3<double> A = metis::Mat3<double>::Identity() * 4.0;
    metis::Vec3<double> b;
    b << 1.0, 2.0, 3.0;
    auto x = metis::solve(A, b, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::LDLT));
    static_assert(std::is_same_v<decltype(x), metis::Vec3<double>>,
                  "LDLT solve should preserve fixed-size type");
    EXPECT_NEAR(x(0), 0.25, 1e-12);
    EXPECT_NEAR(x(1), 0.50, 1e-12);
    EXPECT_NEAR(x(2), 0.75, 1e-12);
}

// Non-square least-squares regression: for A (3x2) and b (3x1), the solution
// should be a 2x1 vector (rows = A.cols, not A.rows). Catches the earlier
// Result typedef that used DerivedA::RowsAtCompileTime.
TEST(LinalgTests, SolveFixedSizeNonSquareLeastSquares) {
    Eigen::Matrix<double, 3, 2> A;
    A << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    metis::Vec3<double> b;
    b << 1.0, 2.0, 3.0;
    auto x = metis::solve(
        A, b, metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::ColPivHouseholderQR));
    static_assert(std::is_same_v<decltype(x), Eigen::Matrix<double, 2, 1>>,
                  "Non-square QR solve result must have A.cols rows");
    // Normal equations: x = (AᵀA)⁻¹ Aᵀ b = [[2,1],[1,2]]⁻¹ [4,5] = [1, 2]
    EXPECT_NEAR(x(0), 1.0, 1e-12);
    EXPECT_NEAR(x(1), 2.0, 1e-12);
}
