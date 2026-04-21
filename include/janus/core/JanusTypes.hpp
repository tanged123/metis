/// @file MetisTypes.hpp
/// @brief Core type aliases for numeric and symbolic Eigen/CasADi interop
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <casadi/casadi.hpp>
#include <limits>

namespace Eigen {
/**
 * @brief NumTraits specialization for casadi::MX
 * Required for Eigen's operator<< and other scalar-dependent operations.
 */
template <> struct NumTraits<casadi::MX> : GenericNumTraits<casadi::MX> {
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 1
    };

    static inline int digits10() { return std::numeric_limits<double>::digits10; }
    static inline int min_exponent() { return std::numeric_limits<double>::min_exponent; }
    static inline int max_exponent() { return std::numeric_limits<double>::max_exponent; }
    static inline casadi::MX epsilon() { return std::numeric_limits<double>::epsilon(); }
    static inline casadi::MX dummy_precision() { return 1e-5; }
    static inline casadi::MX highest() { return std::numeric_limits<double>::max(); }
    static inline casadi::MX lowest() { return std::numeric_limits<double>::lowest(); }
};
} // namespace Eigen

namespace metis {

// --- Matrix Types ---
/**
 * @brief Dynamic-size matrix for both numeric and symbolic backends
 * @tparam Scalar Element type (double or casadi::MX)
 */
template <typename Scalar>
using MetisMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Dynamic-size column vector for both numeric and symbolic backends
 * @tparam Scalar Element type (double or casadi::MX)
 */
template <typename Scalar> using MetisVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// --- Fixed-Size Types ---
/**
 * @brief Fixed-size vectors and matrices for performance-critical code
 * @tparam Scalar Element type (double or casadi::MX)
 */
template <typename Scalar> using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
template <typename Scalar> using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
template <typename Scalar> using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

template <typename Scalar> using Mat2 = Eigen::Matrix<Scalar, 2, 2>;
template <typename Scalar> using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
template <typename Scalar> using Mat4 = Eigen::Matrix<Scalar, 4, 4>;

// Numeric Backend
using NumericScalar = double;                     ///< Numeric scalar type
using NumericMatrix = MetisMatrix<NumericScalar>; ///< Eigen::MatrixXd equivalent
using NumericVector = MetisVector<NumericScalar>; ///< Eigen::VectorXd equivalent

// Symbolic Backend
using SymbolicScalar = casadi::MX;                  ///< CasADi MX symbolic scalar
using SymbolicMatrix = MetisMatrix<SymbolicScalar>; ///< Eigen matrix of MX elements
using SymbolicVector = MetisVector<SymbolicScalar>; ///< Eigen vector of MX elements

// --- Sparse Numeric Types ---
/**
 * @brief Sparse matrix types for efficient storage of large, sparse numeric data
 * @note For symbolic sparsity analysis, use metis::SparsityPattern.
 * @see SparsityPattern
 */
using SparseMatrix = Eigen::SparseMatrix<double>; ///< Sparse numeric matrix (CSC)
using SparseTriplet = Eigen::Triplet<double>;     ///< (row, col, value) triplet

// --- Symbolic Variable Creation ---

/**
 * @brief Create a named symbolic scalar variable
 * @param name Name of the variable
 * @return SymbolicScalar (casadi::MX)
 */
inline SymbolicScalar sym(const std::string &name) { return casadi::MX::sym(name); }

/**
 * @brief Create a named symbolic matrix variable
 * @param name Name of the variable
 * @param rows Number of rows
 * @param cols Number of columns (default 1)
 * @return SymbolicScalar (casadi::MX) representing a matrix
 */
inline SymbolicScalar sym(const std::string &name, int rows, int cols = 1) {
    return casadi::MX::sym(name, rows, cols);
}

/**
 * @brief Create a named symbolic vector (returns SymbolicVector)
 *
 * @code
 * auto x = metis::sym_vector("x", 3);  // Returns SymbolicVector
 * @endcode
 *
 * @param name Name of the variable
 * @param size Number of elements
 * @return SymbolicVector with MX elements
 * @see sym_vec, sym_vec_pair
 */
inline SymbolicVector sym_vector(const std::string &name, int size) {
    casadi::MX mx = casadi::MX::sym(name, size, 1);
    SymbolicVector v(size);
    for (int i = 0; i < size; ++i) {
        v(i) = mx(i);
    }
    return v;
}

/**
 * @brief Create a symbolic vector preserving the CasADi primitive connection
 *
 * @code
 * auto state = metis::sym_vec("state", 3);
 * auto jac = metis::jacobian({metis::to_mx(dydt)}, {metis::to_mx(state)});
 * @endcode
 *
 * @param name Name of the variable
 * @param size Number of elements
 * @return SymbolicVector with MX elements from single underlying vector
 * @see sym_vector, sym_vec_pair
 */
inline SymbolicVector sym_vec(const std::string &name, int size) {
    // Delegate to sym_vector - they are equivalent
    return sym_vector(name, size);
}

/**
 * @brief Create symbolic vector and return both SymbolicVector and underlying MX
 *
 * @code
 * auto [state_vec, state_mx] = metis::sym_vec_pair("state", 3);
 * auto jac = metis::jacobian({metis::to_mx(dydt)}, {state_mx, theta});
 * @endcode
 *
 * @param name Name of the variable
 * @param size Number of elements
 * @return Pair of (SymbolicVector, underlying MX)
 * @see sym_vec
 */
inline std::pair<SymbolicVector, SymbolicScalar> sym_vec_pair(const std::string &name, int size) {
    casadi::MX mx = casadi::MX::sym(name, size, 1);
    SymbolicVector v(size);
    for (int i = 0; i < size; ++i) {
        v(i) = mx(i);
    }
    return {v, mx};
}

/**
 * @brief Get the underlying MX representation of a SymbolicVector
 *
 * This packs an Eigen container of MX elements back into a single CasADi MX
 * for use with metis::Function or metis::jacobian.
 *
 * @param v SymbolicVector to convert
 * @return Single casadi::MX representing the vector
 */
inline SymbolicScalar as_mx(const SymbolicVector &v) {
    casadi::MX m(v.size(), 1);
    for (int i = 0; i < v.size(); ++i) {
        m(i) = v(i);
    }
    return m;
}

// --- Universal Conversion Helpers ---

/**
 * @brief Convert Eigen matrix of MX (or numeric) to CasADi MX
 * @tparam Derived Eigen matrix type
 * @param e Input Eigen matrix
 * @return CasADi MX (dense)
 */
template <typename Derived> casadi::MX to_mx(const Eigen::MatrixBase<Derived> &e) {
    if (e.size() == 0)
        return casadi::MX(e.rows(), e.cols());

    // Create an MX of correct shape
    casadi::MX m(e.rows(), e.cols());
    // Fill it element-wise
    for (Eigen::Index i = 0; i < e.rows(); ++i) {
        for (Eigen::Index j = 0; j < e.cols(); ++j) {
            if constexpr (std::is_same_v<typename Derived::Scalar, casadi::MX>) {
                m(static_cast<int>(i), static_cast<int>(j)) = e(i, j);
            } else {
                m(static_cast<int>(i), static_cast<int>(j)) = casadi::MX(e(i, j));
            }
        }
    }
    return m;
}

/**
 * @brief Convert CasADi MX to Eigen matrix of MX
 * @param m Input CasADi MX
 * @return Eigen matrix (dynamic size)
 */
inline Eigen::Matrix<casadi::MX, Eigen::Dynamic, Eigen::Dynamic> to_eigen(const casadi::MX &m) {
    Eigen::Matrix<casadi::MX, Eigen::Dynamic, Eigen::Dynamic> e(m.size1(), m.size2());
    for (int i = 0; i < m.size1(); ++i) {
        for (int j = 0; j < m.size2(); ++j) {
            e(i, j) = m(i, j);
        }
    }
    return e;
}

/**
 * @brief Convert CasADi MX vector to SymbolicVector (Eigen container of MX)
 * @param m Input CasADi MX (column vector)
 * @return SymbolicVector (Eigen::Matrix<casadi::MX, Dynamic, 1>)
 */
inline SymbolicVector as_vector(const casadi::MX &m) {
    SymbolicVector v(m.size1());
    for (int i = 0; i < m.size1(); ++i) {
        v(i) = m(i);
    }
    return v;
}

/// @brief Backwards compatibility alias for as_vector
/// @param m Input CasADi MX (column vector)
/// @return SymbolicVector
/// @see as_vector
inline SymbolicVector to_eigen_vec(const casadi::MX &m) { return as_vector(m); }

/**
 * @brief Universal symbolic argument wrapper for Function inputs/outputs
 *
 * Allows automatic flattening of Eigen matrices and MX scalars into a single
 * casadi::MX type. Enables mixed initializer lists: {scalar_sym, matrix_sym}.
 *
 * @see Function
 */
class SymbolicArg {
  public:
    /**
     * @brief Construct from single symbolic scalar (MX)
     * @param s Symbolic scalar value
     */
    SymbolicArg(const SymbolicScalar &s) : mx_(s) {}

    /**
     * @brief Construct from Eigen matrix of symbolic scalars
     * @tparam Derived Eigen expression type
     * @param e Eigen matrix to flatten into MX
     */
    template <typename Derived> SymbolicArg(const Eigen::MatrixBase<Derived> &e) {
        if (e.size() == 0) {
            mx_ = casadi::MX(e.rows(), e.cols());
            return;
        }
        mx_ = casadi::MX(e.rows(), e.cols());
        for (Eigen::Index i = 0; i < e.rows(); ++i) {
            for (Eigen::Index j = 0; j < e.cols(); ++j) {
                mx_(static_cast<int>(i), static_cast<int>(j)) = e(i, j);
            }
        }
    }

    /**
     * @brief Implicit conversion to CasADi MX
     * @return Underlying MX object
     */
    operator SymbolicScalar() const { return mx_; }

    /**
     * @brief Get underlying CasADi MX object
     * @return Copy of the stored MX
     */
    SymbolicScalar get() const { return mx_; }

  private:
    SymbolicScalar mx_;
};

} // namespace metis
