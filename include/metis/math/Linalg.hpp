#pragma once
/**
 * @file Linalg.hpp
 * @brief Linear algebra operations (solve, inverse, determinant, eigendecomposition, norms)
 * @see Arithmetic.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Arithmetic.hpp"
#include "metis/math/Logic.hpp"
#include "metis/math/Trig.hpp"
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <algorithm>
#include <array>
#include <casadi/casadi.hpp>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <unsupported/Eigen/IterativeSolvers>
#include <vector>

namespace metis {

// --- Conversion Helpers ---
// (Moved to MetisTypes.hpp: to_mx, to_eigen, as_vector)

/**
 * @brief Backend selection for linear system solves
 */
enum class LinearSolveBackend {
    Dense,          ///< Dense matrix factorization
    SparseDirect,   ///< Sparse direct factorization
    IterativeKrylov ///< Iterative Krylov subspace method
};

/**
 * @brief Dense linear solver algorithm
 */
enum class DenseLinearSolver {
    ColPivHouseholderQR, ///< Column-pivoted Householder QR (default, general)
    PartialPivLU,        ///< Partial-pivot LU (square only)
    FullPivLU,           ///< Full-pivot LU (square only)
    LLT,                 ///< Cholesky (SPD only)
    LDLT,                ///< LDLT (symmetric only)
};

/**
 * @brief Sparse direct solver algorithm
 */
enum class SparseDirectLinearSolver {
    SparseLU,       ///< Sparse LU factorization
    SparseQR,       ///< Sparse QR factorization
    SimplicialLLT,  ///< Simplicial Cholesky (SPD only)
    SimplicialLDLT, ///< Simplicial LDLT (symmetric only)
};

/**
 * @brief Iterative Krylov solver algorithm
 */
enum class IterativeKrylovSolver {
    BiCGSTAB, ///< Biconjugate gradient stabilized
    GMRES     ///< Generalized minimal residual
};

/**
 * @brief Preconditioner for iterative solvers
 */
enum class IterativePreconditioner {
    None,    ///< No preconditioning
    Diagonal ///< Jacobi (diagonal) preconditioner
};

/**
 * @brief Configuration for linear system solve backend and algorithm
 * @see solve
 */
struct LinearSolvePolicy {
    LinearSolveBackend backend = LinearSolveBackend::Dense;
    DenseLinearSolver dense_solver = DenseLinearSolver::ColPivHouseholderQR;
    SparseDirectLinearSolver sparse_direct_solver = SparseDirectLinearSolver::SparseLU;
    IterativeKrylovSolver iterative_solver = IterativeKrylovSolver::BiCGSTAB;
    IterativePreconditioner iterative_preconditioner = IterativePreconditioner::Diagonal;
    double tolerance = 1e-10;
    int max_iterations = 500;
    int gmres_restart = 30;
    std::function<NumericVector(const NumericVector &)> preconditioner_hook;
    std::string symbolic_linear_solver;
    casadi::Dict symbolic_options;

    static LinearSolvePolicy
    dense(DenseLinearSolver solver = DenseLinearSolver::ColPivHouseholderQR) {
        LinearSolvePolicy policy;
        policy.backend = LinearSolveBackend::Dense;
        policy.dense_solver = solver;
        return policy;
    }

    static LinearSolvePolicy
    sparse_direct(SparseDirectLinearSolver solver = SparseDirectLinearSolver::SparseLU) {
        LinearSolvePolicy policy;
        policy.backend = LinearSolveBackend::SparseDirect;
        policy.sparse_direct_solver = solver;
        return policy;
    }

    static LinearSolvePolicy
    iterative(IterativeKrylovSolver solver = IterativeKrylovSolver::BiCGSTAB,
              IterativePreconditioner preconditioner = IterativePreconditioner::Diagonal) {
        LinearSolvePolicy policy;
        policy.backend = LinearSolveBackend::IterativeKrylov;
        policy.iterative_solver = solver;
        policy.iterative_preconditioner = preconditioner;
        return policy;
    }

    LinearSolvePolicy &set_tolerance(double value) {
        tolerance = value;
        return *this;
    }

    LinearSolvePolicy &set_max_iterations(int value) {
        max_iterations = value;
        return *this;
    }

    LinearSolvePolicy &set_gmres_restart(int value) {
        gmres_restart = value;
        return *this;
    }

    LinearSolvePolicy &
    set_preconditioner_hook(std::function<NumericVector(const NumericVector &)> hook) {
        preconditioner_hook = std::move(hook);
        return *this;
    }

    LinearSolvePolicy &set_symbolic_solver(const std::string &solver,
                                           const casadi::Dict &opts = casadi::Dict()) {
        symbolic_linear_solver = solver;
        symbolic_options = opts;
        return *this;
    }
};

namespace detail {

class FunctionalPreconditioner {
  public:
    using Scalar = double;
    using RealScalar = double;
    using StorageIndex = int;
    enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

    FunctionalPreconditioner() = default;

    void set_apply(std::function<NumericVector(const NumericVector &)> apply) {
        apply_ = std::move(apply);
    }

    Eigen::Index rows() const noexcept { return size_; }
    Eigen::Index cols() const noexcept { return size_; }

    template <typename MatType> FunctionalPreconditioner &analyzePattern(const MatType &) {
        return *this;
    }

    template <typename MatType> FunctionalPreconditioner &factorize(const MatType &mat) {
        size_ = mat.cols();
        initialized_ = true;
        return *this;
    }

    template <typename MatType> FunctionalPreconditioner &compute(const MatType &mat) {
        return factorize(mat);
    }

    template <typename Rhs, typename Dest> void _solve_impl(const Rhs &b, Dest &x) const {
        NumericVector rhs = b;
        x = apply_ ? apply_(rhs) : rhs;
    }

    template <typename Rhs>
    inline const Eigen::Solve<FunctionalPreconditioner, Rhs>
    solve(const Eigen::MatrixBase<Rhs> &b) const {
        eigen_assert(initialized_ && "FunctionalPreconditioner is not initialized.");
        eigen_assert(size_ == b.rows() && "FunctionalPreconditioner::solve(): invalid rhs size");
        return Eigen::Solve<FunctionalPreconditioner, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info() const { return Eigen::Success; }

  private:
    std::function<NumericVector(const NumericVector &)> apply_;
    Eigen::Index size_ = 0;
    bool initialized_ = false;
};

inline void validate_linear_solve_dims(Eigen::Index a_rows, Eigen::Index a_cols,
                                       Eigen::Index b_rows, const std::string &context) {
    if (a_rows == 0 || a_cols == 0) {
        throw InvalidArgument(context + ": coefficient matrix must be non-empty");
    }
    if (b_rows != a_rows) {
        throw InvalidArgument(context + ": rhs row count must match coefficient matrix rows");
    }
}

inline void validate_square_required(Eigen::Index rows, Eigen::Index cols,
                                     const std::string &context, const std::string &solver_name) {
    if (rows != cols) {
        throw InvalidArgument(context + ": " + solver_name + " requires a square matrix");
    }
}

inline void validate_iterative_policy(const LinearSolvePolicy &policy, const std::string &context) {
    if (policy.tolerance <= 0.0) {
        throw InvalidArgument(context + ": tolerance must be positive");
    }
    if (policy.max_iterations <= 0) {
        throw InvalidArgument(context + ": max_iterations must be positive");
    }
    if (policy.gmres_restart <= 0) {
        throw InvalidArgument(context + ": gmres_restart must be positive");
    }
}

inline std::function<NumericVector(const NumericVector &)>
make_preconditioner(const SparseMatrix &A, const LinearSolvePolicy &policy) {
    if (policy.preconditioner_hook) {
        return policy.preconditioner_hook;
    }

    switch (policy.iterative_preconditioner) {
    case IterativePreconditioner::None:
        return [](const NumericVector &rhs) { return rhs; };
    case IterativePreconditioner::Diagonal: {
        NumericVector inv_diag(A.rows());
        const double eps = std::numeric_limits<double>::epsilon();
        for (int i = 0; i < A.rows(); ++i) {
            const double diag = A.coeff(i, i);
            inv_diag(i) = std::abs(diag) > eps ? 1.0 / diag : 1.0;
        }
        return [inv_diag](const NumericVector &rhs) { return inv_diag.array() * rhs.array(); };
    }
    }
    throw InvalidArgument("solve: unsupported iterative preconditioner");
}

template <typename MatrixLike> SparseMatrix dense_to_sparse(const MatrixLike &A) {
    SparseMatrix sparse = A.sparseView();
    sparse.makeCompressed();
    return sparse;
}

template <typename DerivedB>
auto solve_sparse_direct_numeric(const SparseMatrix &A, const Eigen::MatrixBase<DerivedB> &b,
                                 const LinearSolvePolicy &policy) {
    using Result = Eigen::Matrix<double, Eigen::Dynamic, DerivedB::ColsAtCompileTime>;

    SparseMatrix matrix = A;
    matrix.makeCompressed();

    switch (policy.sparse_direct_solver) {
    case SparseDirectLinearSolver::SparseLU: {
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(matrix);
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("solve: SparseLU factorization failed");
        }
        return solver.solve(b).eval();
    }
    case SparseDirectLinearSolver::SparseQR: {
        metis::SparseQR<SparseMatrix> solver;
        solver.compute(matrix);
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("solve: SparseQR factorization failed");
        }
        return solver.solve(b).eval();
    }
    case SparseDirectLinearSolver::SimplicialLLT: {
        validate_square_required(matrix.rows(), matrix.cols(), "solve", "SimplicialLLT");
        metis::SimplicialLLT<SparseMatrix> solver;
        solver.compute(matrix);
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("solve: SimplicialLLT factorization failed");
        }
        return solver.solve(b).eval();
    }
    case SparseDirectLinearSolver::SimplicialLDLT: {
        validate_square_required(matrix.rows(), matrix.cols(), "solve", "SimplicialLDLT");
        metis::SimplicialLDLT<SparseMatrix> solver;
        solver.compute(matrix);
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("solve: SimplicialLDLT factorization failed");
        }
        return solver.solve(b).eval();
    }
    }

    return Result();
}

template <typename Solver, typename DerivedB>
auto solve_iterative_with_solver(Solver &solver, const Eigen::MatrixBase<DerivedB> &b) {
    using Result = Eigen::Matrix<double, Eigen::Dynamic, DerivedB::ColsAtCompileTime>;

    Result x(b.rows(), b.cols());
    for (int col = 0; col < b.cols(); ++col) {
        NumericVector rhs = b.col(col);
        NumericVector sol = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("solve: iterative solver failed to converge");
        }
        x.col(col) = sol;
    }
    return x;
}

template <typename DerivedB>
auto solve_iterative_numeric(const SparseMatrix &A, const Eigen::MatrixBase<DerivedB> &b,
                             const LinearSolvePolicy &policy) {
    validate_square_required(A.rows(), A.cols(), "solve", "iterative Krylov backends");
    validate_iterative_policy(policy, "solve");

    FunctionalPreconditioner preconditioner;
    preconditioner.set_apply(make_preconditioner(A, policy));

    switch (policy.iterative_solver) {
    case IterativeKrylovSolver::BiCGSTAB: {
        Eigen::BiCGSTAB<SparseMatrix, FunctionalPreconditioner> solver;
        solver.setTolerance(policy.tolerance);
        solver.setMaxIterations(policy.max_iterations);
        solver.preconditioner() = preconditioner;
        solver.compute(A);
        return solve_iterative_with_solver(solver, b);
    }
    case IterativeKrylovSolver::GMRES: {
        Eigen::GMRES<SparseMatrix, FunctionalPreconditioner> solver;
        solver.setTolerance(policy.tolerance);
        solver.setMaxIterations(policy.max_iterations);
        solver.set_restart(policy.gmres_restart);
        solver.preconditioner() = preconditioner;
        solver.compute(A);
        return solve_iterative_with_solver(solver, b);
    }
    }

    using Result = Eigen::Matrix<double, Eigen::Dynamic, DerivedB::ColsAtCompileTime>;
    return Result();
}

template <typename DerivedA, typename DerivedB>
auto solve_dense_numeric(const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &b,
                         const LinearSolvePolicy &policy) {
    switch (policy.dense_solver) {
    case DenseLinearSolver::ColPivHouseholderQR:
        return A.colPivHouseholderQr().solve(b).eval();
    case DenseLinearSolver::PartialPivLU:
        validate_square_required(A.rows(), A.cols(), "solve", "PartialPivLU");
        return A.partialPivLu().solve(b).eval();
    case DenseLinearSolver::FullPivLU:
        validate_square_required(A.rows(), A.cols(), "solve", "FullPivLU");
        return A.fullPivLu().solve(b).eval();
    case DenseLinearSolver::LLT: {
        validate_square_required(A.rows(), A.cols(), "solve", "LLT");
        metis::LLT<typename DerivedA::PlainObject> solver(A.eval());
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("solve: LLT factorization failed");
        }
        return solver.solve(b).eval();
    }
    case DenseLinearSolver::LDLT: {
        validate_square_required(A.rows(), A.cols(), "solve", "LDLT");
        metis::LDLT<typename DerivedA::PlainObject> solver(A.eval());
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("solve: LDLT factorization failed");
        }
        return solver.solve(b).eval();
    }
    }

    return A.colPivHouseholderQr().solve(b).eval();
}

} // namespace detail

// --- solve(A, b) ---
/**
 * @brief Solves linear system Ax = b using the default backend policy.
 *
 * Numeric dense matrices default to column-pivoted Householder QR. Symbolic MX matrices keep
 * CasADi's default linear solver selection.
 *
 * @param A Coefficient matrix
 * @param b Right-hand side vector
 * @return Solution vector x
 */
template <typename DerivedA, typename DerivedB>
auto solve(const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &b) {
    return solve(A, b, LinearSolvePolicy());
}

/**
 * @brief Solves linear system Ax = b using an explicit backend policy.
 */
template <typename DerivedA, typename DerivedB>
auto solve(const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &b,
           const LinearSolvePolicy &policy) {
    using Scalar = typename DerivedA::Scalar;
    detail::validate_linear_solve_dims(A.rows(), A.cols(), b.rows(), "solve");

    if constexpr (std::is_floating_point_v<Scalar>) {
        // Unify the return type across dense/sparse/iterative backends so `auto`
        // deduction succeeds. Rows come from A (square systems preserve A's row
        // dim), cols come from b. Fixed-size inputs yield fixed-size results.
        using Result =
            Eigen::Matrix<Scalar, DerivedA::RowsAtCompileTime, DerivedB::ColsAtCompileTime>;
        switch (policy.backend) {
        case LinearSolveBackend::Dense:
            return Result(detail::solve_dense_numeric(A, b, policy));
        case LinearSolveBackend::SparseDirect:
            return Result(
                detail::solve_sparse_direct_numeric(detail::dense_to_sparse(A.eval()), b, policy));
        case LinearSolveBackend::IterativeKrylov:
            return Result(
                detail::solve_iterative_numeric(detail::dense_to_sparse(A.eval()), b, policy));
        }
        return Result(detail::solve_dense_numeric(A, b, policy));
    } else {
        SymbolicScalar A_mx = to_mx(A);
        SymbolicScalar b_mx = to_mx(b);
        if (policy.symbolic_linear_solver.empty()) {
            if (!policy.symbolic_options.empty()) {
                throw InvalidArgument(
                    "solve: symbolic_options require a non-empty symbolic_linear_solver");
            }
            SymbolicScalar x_mx = SymbolicScalar::solve(A_mx, b_mx);
            return to_eigen(x_mx);
        }
        SymbolicScalar x_mx = SymbolicScalar::solve(A_mx, b_mx, policy.symbolic_linear_solver,
                                                    policy.symbolic_options);
        return to_eigen(x_mx);
    }
}

/**
 * @brief Solve a numeric sparse linear system with a sparse-aware default backend.
 *
 * Sparse input defaults to `SparseLU`, since there was no previous sparse overload to preserve.
 */
template <typename DerivedB>
auto solve(const SparseMatrix &A, const Eigen::MatrixBase<DerivedB> &b) {
    return solve(A, b, LinearSolvePolicy::sparse_direct());
}

/**
 * @brief Solve a numeric sparse linear system using an explicit backend policy.
 */
template <typename DerivedB>
auto solve(const SparseMatrix &A, const Eigen::MatrixBase<DerivedB> &b,
           const LinearSolvePolicy &policy) {
    detail::validate_linear_solve_dims(A.rows(), A.cols(), b.rows(), "solve");

    switch (policy.backend) {
    case LinearSolveBackend::Dense:
        return detail::solve_dense_numeric(NumericMatrix(A), b, policy);
    case LinearSolveBackend::SparseDirect:
        return detail::solve_sparse_direct_numeric(A, b, policy);
    case LinearSolveBackend::IterativeKrylov:
        return detail::solve_iterative_numeric(A, b, policy);
    }

    return detail::solve_sparse_direct_numeric(A, b, policy);
}

// --- outer(x, y) ---
/**
 * @brief Computes outer product x * y^T
 * @param x First vector
 * @param y Second vector
 * @return Outer product matrix
 */
template <typename DerivedX, typename DerivedY>
auto outer(const Eigen::MatrixBase<DerivedX> &x, const Eigen::MatrixBase<DerivedY> &y) {
    // Eigen's outer product works efficiently for both numeric and symbolic scalars
    // because MX * MX (scalar mult) creates a standard multiplication node.
    return x * y.transpose();
}

// --- Dot Product ---
/**
 * @brief Computes dot product of two vectors
 * @param a First vector
 * @param b Second vector
 * @return Dot product
 */
template <typename DerivedA, typename DerivedB>
auto dot(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    return a.dot(b);
}

// --- Cross Product ---
/**
 * @brief Computes 3D cross product
 * @param a First 3-element vector
 * @param b Second 3-element vector
 * @return Cross product vector
 * @throws InvalidArgument if vectors are not 3 elements
 */
template <typename DerivedA, typename DerivedB>
auto cross(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    if (a.size() != 3 || b.size() != 3) {
        throw InvalidArgument("cross: both vectors must have exactly 3 elements");
    }
    // Manual implementation to support Dynamic vectors (Eigen::cross requires fixed size 3)
    using Scalar = typename DerivedA::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res(3);
    res(0) = a(1) * b(2) - a(2) * b(1);
    res(1) = a(2) * b(0) - a(0) * b(2);
    res(2) = a(0) * b(1) - a(1) * b(0);
    return res;
}

// --- Inverse ---
/**
 * @brief Computes matrix inverse
 * @param A Input matrix
 * @return Inverse of A
 */
template <typename Derived> auto inv(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return A.inverse().eval();
    } else {
        SymbolicScalar A_mx = to_mx(A);
        SymbolicScalar inv_mx = inv(A_mx);
        return to_eigen(inv_mx);
    }
}

// --- Determinant ---
/**
 * @brief Computes matrix determinant
 * @param A Input matrix
 * @return Determinant of A
 */
template <typename Derived> auto det(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return A.determinant();
    } else {
        SymbolicScalar A_mx = to_mx(A);
        return det(A_mx);
    }
}

// --- Inner Product ---
/**
 * @brief Computes inner product of two vectors (dot product)
 * @param x First vector
 * @param y Second vector
 * @return Inner product
 */
template <typename DerivedA, typename DerivedB>
auto inner(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    return metis::dot(a, b);
}

// --- Pseudo-Inverse ---
/**
 * @brief Computes Moore-Penrose pseudo-inverse
 * @param A Input matrix
 * @return Pseudo-inverse of A
 */
template <typename Derived> auto pinv(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        return A.completeOrthogonalDecomposition().pseudoInverse();
    } else {
        SymbolicScalar A_mx = to_mx(A);
        SymbolicScalar pinv_mx = SymbolicScalar::pinv(A_mx);
        return to_eigen(pinv_mx);
    }
}

// --- Norm Extensions ---

/**
 * @brief Norm type selection
 */
enum class NormType {
    L1,       ///< L1 (Manhattan) norm
    L2,       ///< L2 (Euclidean) norm
    Inf,      ///< Infinity (max absolute) norm
    Frobenius ///< Frobenius norm
};

/**
 * @brief Computes vector/matrix norm
 * @param x Input vector/matrix
 * @param type Norm type (L1, L2, Inf, Frobenius)
 * @return Norm value
 */
template <typename Derived>
auto norm(const Eigen::MatrixBase<Derived> &x, NormType type = NormType::L2) {
    using Scalar = typename Derived::Scalar;

    if constexpr (std::is_floating_point_v<Scalar>) {
        switch (type) {
        case NormType::L1:
            return x.template lpNorm<1>();
        case NormType::L2:
            return x.norm();
        case NormType::Inf:
            return x.template lpNorm<Eigen::Infinity>();
        case NormType::Frobenius:
            return x.norm(); // Frobenius equals L2 for vectors
        default:
            return x.norm();
        }
    } else {
        SymbolicScalar x_mx = to_mx(x);
        switch (type) {
        case NormType::L1:
            return SymbolicScalar::norm_1(x_mx);
        case NormType::L2:
            return SymbolicScalar::norm_2(x_mx);
        case NormType::Inf:
            return SymbolicScalar::norm_inf(x_mx);
        case NormType::Frobenius:
            return SymbolicScalar::norm_fro(x_mx);
        default:
            return SymbolicScalar::norm_2(x_mx);
        }
    }
}

// Backwards compatibility overload for just L2 norm (default handled above really, but if called
// without args, it works)

/**
 * @brief Result of eigendecomposition: eigenvalues and eigenvectors
 * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
 */
template <typename Scalar> struct EigenDecomposition {
    MetisVector<Scalar> eigenvalues;  ///< Eigenvalues in ascending order
    MetisMatrix<Scalar> eigenvectors; ///< Eigenvectors as columns
};

namespace detail {

template <typename Scalar> MetisVector<Scalar> normalize_vector(const MetisVector<Scalar> &v) {
    const auto v_norm = metis::norm(v);
    if constexpr (std::is_floating_point_v<Scalar>) {
        if (v_norm <= std::numeric_limits<Scalar>::epsilon()) {
            throw InvalidArgument("eigendecomposition: eigenvector construction failed");
        }
    }
    return v / v_norm;
}

template <typename Scalar>
MetisVector<Scalar> best_eigenvector_candidate(const std::array<MetisVector<Scalar>, 3> &cands) {
    const auto n01 = metis::dot(cands[0], cands[0]);
    const auto n02 = metis::dot(cands[1], cands[1]);
    const auto n12 = metis::dot(cands[2], cands[2]);

    if constexpr (std::is_floating_point_v<Scalar>) {
        const auto *best = &cands[0];
        Scalar best_norm = n01;
        if (n02 > best_norm) {
            best = &cands[1];
            best_norm = n02;
        }
        if (n12 > best_norm) {
            best = &cands[2];
        }
        return normalize_vector(*best);
    } else {
        auto best = metis::detail::select(n02 > n01, cands[1], cands[0]);
        auto best_norm = metis::where(n02 > n01, n02, n01);
        best = metis::detail::select(n12 > best_norm, cands[2], best);
        return normalize_vector(best);
    }
}

template <typename Scalar>
MetisVector<Scalar> symmetric_eigenvector_2x2(const MetisMatrix<Scalar> &A, const Scalar &lambda) {
    MetisVector<Scalar> first(2);
    first << A(0, 1), lambda - A(0, 0);

    MetisVector<Scalar> second(2);
    second << lambda - A(1, 1), A(1, 0);

    if constexpr (std::is_floating_point_v<Scalar>) {
        return normalize_vector(metis::dot(first, first) >= metis::dot(second, second) ? first
                                                                                       : second);
    } else {
        return normalize_vector(metis::detail::select(
            metis::dot(first, first) >= metis::dot(second, second), first, second));
    }
}

template <typename Scalar>
MetisVector<Scalar> symmetric_eigenvector_3x3(const MetisMatrix<Scalar> &A, const Scalar &lambda) {
    MetisMatrix<Scalar> shifted = A;
    shifted(0, 0) = shifted(0, 0) - lambda;
    shifted(1, 1) = shifted(1, 1) - lambda;
    shifted(2, 2) = shifted(2, 2) - lambda;

    const MetisVector<Scalar> r0 = shifted.row(0).transpose();
    const MetisVector<Scalar> r1 = shifted.row(1).transpose();
    const MetisVector<Scalar> r2 = shifted.row(2).transpose();

    return best_eigenvector_candidate<Scalar>(
        {metis::cross(r0, r1), metis::cross(r0, r2), metis::cross(r1, r2)});
}

template <typename Scalar> Scalar determinant_3x3(const MetisMatrix<Scalar> &A) {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
           A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
}

template <typename Scalar>
void sort_eigenpairs(MetisVector<Scalar> &eigenvalues, MetisMatrix<Scalar> &eigenvectors) {
    std::vector<Eigen::Index> order(static_cast<size_t>(eigenvalues.size()));
    std::iota(order.begin(), order.end(), Eigen::Index{0});
    std::sort(order.begin(), order.end(), [&](Eigen::Index lhs, Eigen::Index rhs) {
        return eigenvalues(lhs) < eigenvalues(rhs);
    });

    MetisVector<Scalar> sorted_values(eigenvalues.size());
    MetisMatrix<Scalar> sorted_vectors(eigenvectors.rows(), eigenvectors.cols());
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        sorted_values(i) = eigenvalues(order[static_cast<size_t>(i)]);
        sorted_vectors.col(i) = eigenvectors.col(order[static_cast<size_t>(i)]);
    }

    eigenvalues = std::move(sorted_values);
    eigenvectors = std::move(sorted_vectors);
}

template <typename Scalar>
EigenDecomposition<Scalar> eig_symmetric_symbolic(const MetisMatrix<Scalar> &A) {
    if (A.rows() != A.cols()) {
        throw InvalidArgument("eig_symmetric: input must be square");
    }

    if (A.rows() == 1) {
        EigenDecomposition<Scalar> result;
        result.eigenvalues.resize(1);
        result.eigenvalues(0) = A(0, 0);
        result.eigenvectors = MetisMatrix<Scalar>::Identity(1, 1);
        return result;
    }

    if (A.rows() == 2) {
        const Scalar trace = A(0, 0) + A(1, 1);
        const Scalar disc = metis::sqrt((A(0, 0) - A(1, 1)) * (A(0, 0) - A(1, 1)) +
                                        Scalar(4.0) * A(0, 1) * A(0, 1));

        EigenDecomposition<Scalar> result;
        result.eigenvalues.resize(2);
        result.eigenvalues(0) = Scalar(0.5) * (trace - disc);
        result.eigenvalues(1) = Scalar(0.5) * (trace + disc);

        result.eigenvectors.resize(2, 2);
        result.eigenvectors.col(0) = symmetric_eigenvector_2x2(A, result.eigenvalues(0));
        result.eigenvectors.col(1) = symmetric_eigenvector_2x2(A, result.eigenvalues(1));
        return result;
    }

    if (A.rows() == 3) {
        const Scalar q = (A(0, 0) + A(1, 1) + A(2, 2)) / Scalar(3.0);
        const Scalar p1 = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
        const Scalar a00 = A(0, 0) - q;
        const Scalar a11 = A(1, 1) - q;
        const Scalar a22 = A(2, 2) - q;
        const Scalar p2 = a00 * a00 + a11 * a11 + a22 * a22 + Scalar(2.0) * p1;

        if constexpr (std::is_floating_point_v<Scalar>) {
            if (p2 <= std::numeric_limits<Scalar>::epsilon()) {
                EigenDecomposition<Scalar> result;
                result.eigenvalues = MetisVector<Scalar>::Constant(3, q);
                result.eigenvectors = MetisMatrix<Scalar>::Identity(3, 3);
                return result;
            }
        }

        const auto has_spread = p2 > Scalar(0.0);
        const Scalar p = metis::where(has_spread, metis::sqrt(p2 / Scalar(6.0)), Scalar(1.0));

        MetisMatrix<Scalar> centered = A;
        centered(0, 0) = centered(0, 0) - q;
        centered(1, 1) = centered(1, 1) - q;
        centered(2, 2) = centered(2, 2) - q;
        const MetisMatrix<Scalar> B = centered / p;

        const Scalar r = metis::clamp(determinant_3x3(B) / Scalar(2.0), -1.0, 1.0);
        const Scalar phi = metis::where(has_spread, metis::acos(r) / Scalar(3.0), Scalar(0.0));

        constexpr double kTwoPiOverThree = 2.0943951023931954923;
        Scalar largest = q + Scalar(2.0) * p * metis::cos(phi);
        Scalar smallest = q + Scalar(2.0) * p * metis::cos(phi + Scalar(kTwoPiOverThree));
        Scalar middle = Scalar(3.0) * q - largest - smallest;

        largest = metis::where(has_spread, largest, q);
        middle = metis::where(has_spread, middle, q);
        smallest = metis::where(has_spread, smallest, q);

        EigenDecomposition<Scalar> result;
        result.eigenvalues.resize(3);
        result.eigenvalues << smallest, middle, largest;

        MetisMatrix<Scalar> vectors(3, 3);
        vectors.col(0) = symmetric_eigenvector_3x3(A, smallest);
        vectors.col(1) = symmetric_eigenvector_3x3(A, middle);
        vectors.col(2) = symmetric_eigenvector_3x3(A, largest);

        const MetisMatrix<Scalar> identity = MetisMatrix<Scalar>::Identity(3, 3);
        result.eigenvectors = metis::detail::select(has_spread, vectors, identity);
        return result;
    }

    throw InvalidArgument(
        "eig_symmetric: symbolic support is limited to 1x1, 2x2, and 3x3 matrices");
}

} // namespace detail

// --- Eigendecomposition ---
/**
 * @brief Computes the eigendecomposition of a square matrix with a real spectrum.
 *
 * Returns eigenvalues in ascending order and eigenvectors as columns of the returned matrix.
 * Numeric matrices use Eigen's general eigensolver. Symbolic MX matrices are not supported in the
 * general case because CasADi does not expose a compatible MX eigendecomposition.
 */
template <typename Derived> auto eig(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;

    if (A.rows() != A.cols()) {
        throw InvalidArgument("eig: input must be square");
    }

    if constexpr (std::is_floating_point_v<Scalar>) {
        using Matrix = MetisMatrix<Scalar>;
        using Complex = std::complex<Scalar>;
        using ComplexMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
        using ComplexVector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;

        if (A.rows() == 1) {
            EigenDecomposition<Scalar> result;
            result.eigenvalues.resize(1);
            result.eigenvalues(0) = A(0, 0);
            result.eigenvectors = Matrix::Identity(1, 1);
            return result;
        }

        Eigen::EigenSolver<Matrix> solver(A.eval());
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("eig: EigenSolver failed");
        }

        const ComplexVector values_complex = solver.eigenvalues();
        const ComplexMatrix vectors_complex = solver.eigenvectors();
        constexpr Scalar kImagTol = Scalar(1e-10);

        for (Eigen::Index i = 0; i < values_complex.size(); ++i) {
            if (std::abs(values_complex(i).imag()) > kImagTol) {
                throw InvalidArgument("eig: complex eigenvalues are not supported");
            }
        }

        for (Eigen::Index i = 0; i < vectors_complex.rows(); ++i) {
            for (Eigen::Index j = 0; j < vectors_complex.cols(); ++j) {
                if (std::abs(vectors_complex(i, j).imag()) > kImagTol) {
                    throw InvalidArgument("eig: complex eigenvectors are not supported");
                }
            }
        }

        EigenDecomposition<Scalar> result;
        result.eigenvalues = values_complex.real();
        result.eigenvectors = vectors_complex.real();
        detail::sort_eigenpairs(result.eigenvalues, result.eigenvectors);
        return result;
    } else {
        throw InvalidArgument(
            "eig: symbolic general eigendecomposition is not supported for CasADi MX; use "
            "eig_symmetric() for 1x1, 2x2, or 3x3 symmetric matrices");
    }
}

/**
 * @brief Computes the eigendecomposition of a symmetric matrix.
 *
 * Returns eigenvalues in ascending order and orthonormal eigenvectors as columns.
 * Numeric matrices delegate to Eigen's SelfAdjointEigenSolver. Symbolic MX matrices support only
 * 1x1, 2x2, and 3x3 symmetric inputs.
 */
template <typename Derived> auto eig_symmetric(const Eigen::MatrixBase<Derived> &A) {
    using Scalar = typename Derived::Scalar;

    if (A.rows() != A.cols()) {
        throw InvalidArgument("eig_symmetric: input must be square");
    }

    if constexpr (std::is_floating_point_v<Scalar>) {
        if (!A.isApprox(A.transpose(), 1e-12)) {
            throw InvalidArgument("eig_symmetric: numeric input must be symmetric");
        }

        using Matrix = MetisMatrix<Scalar>;
        Eigen::SelfAdjointEigenSolver<Matrix> solver(A.eval());
        if (solver.info() != Eigen::Success) {
            throw InvalidArgument("eig_symmetric: SelfAdjointEigenSolver failed");
        }

        EigenDecomposition<Scalar> result;
        result.eigenvalues = solver.eigenvalues();
        result.eigenvectors = solver.eigenvectors();
        detail::sort_eigenpairs(result.eigenvalues, result.eigenvectors);
        return result;
    } else {
        return detail::eig_symmetric_symbolic(A.eval());
    }
}

// --- Explicit 3x3 Symmetric Inverse (AeroSandbox Helper) ---
/**
 * @brief Explicit inverse of a symmetric 3x3 matrix.
 *
 * Input matrix:
 * [m11, m12, m13]
 * [m12, m22, m23]
 * [m13, m23, m33]
 *
 * Returns the six unique inverse coefficients in Eigen-compatible packed column-major order for
 * the lower triangle: `(0,0), (1,0), (2,0), (1,1), (2,1), (2,2)`, i.e.
 * `{a11, a12, a13, a22, a23, a33}` for
 *
 * [a11, a12, a13]
 * [a12, a22, a23]
 * [a13, a23, a33]
 */
template <typename T>
std::tuple<T, T, T, T, T, T> inv_symmetric_3x3_explicit(const T &m11, const T &m22, const T &m33,
                                                        const T &m12, const T &m23, const T &m13) {

    T det = m11 * (m33 * m22 - m23 * m23) - m12 * (m33 * m12 - m23 * m13) +
            m13 * (m23 * m12 - m22 * m13);

    T inv_det = 1.0 / det;

    T a11 = (m33 * m22 - m23 * m23) * inv_det;
    T a12 = (m13 * m23 - m33 * m12) * inv_det;
    T a13 = (m12 * m23 - m13 * m22) * inv_det;
    T a22 = (m33 * m11 - m13 * m13) * inv_det;
    T a23 = (m12 * m13 - m11 * m23) * inv_det;
    T a33 = (m11 * m22 - m12 * m12) * inv_det;

    return {a11, a12, a13, a22, a23, a33};
}

// --- Sparse Matrix Utilities ---
// NOTE: Sparse matrices are for NUMERIC data only. CasADi MX handles sparsity internally.
// For symbolic sparsity analysis, use metis::SparsityPattern.

/**
 * @brief Compile-time check for numeric scalar types
 *
 * This prevents confusing errors when someone tries to use sparse matrices
 * with symbolic types in templated code.
 */
template <typename Scalar>
constexpr bool is_numeric_scalar_v = std::is_floating_point_v<Scalar> || std::is_integral_v<Scalar>;

/**
 * @brief Create sparse matrix from triplets
 *
 * Triplets specify (row, col, value) entries. Duplicate entries are summed.
 *
 * @code
 * std::vector<metis::SparseTriplet> triplets;
 * triplets.emplace_back(0, 0, 1.0);
 * triplets.emplace_back(1, 1, 2.0);
 * auto sp = metis::sparse_from_triplets(2, 2, triplets);
 * @endcode
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param triplets Vector of (row, col, value) triplets
 * @return Compressed sparse matrix
 */
inline SparseMatrix sparse_from_triplets(int rows, int cols,
                                         const std::vector<SparseTriplet> &triplets) {
    SparseMatrix m(rows, cols);
    m.setFromTriplets(triplets.begin(), triplets.end());
    return m;
}

/**
 * @brief Convert dense matrix to sparse
 *
 * Elements with absolute value <= tol are treated as zero.
 *
 * @param dense Dense numeric matrix
 * @param tol Tolerance for zero (default 0.0 = exact zeros only)
 * @return Sparse matrix
 */
inline SparseMatrix to_sparse(const NumericMatrix &dense, double tol = 0.0) {
    SparseMatrix sparse = dense.sparseView(1.0, tol);
    sparse.makeCompressed();
    return sparse;
}

/**
 * @brief Convert sparse matrix to dense
 *
 * @param sparse Sparse matrix
 * @return Dense numeric matrix
 */
inline NumericMatrix to_dense(const SparseMatrix &sparse) { return NumericMatrix(sparse); }

/**
 * @brief Create identity sparse matrix
 *
 * @param n Size (n x n)
 * @return Sparse identity matrix
 */
inline SparseMatrix sparse_identity(int n) {
    SparseMatrix I(n, n);
    I.setIdentity();
    return I;
}

} // namespace metis
