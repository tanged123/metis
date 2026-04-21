#pragma once
/**
 * @file Logic.hpp
 * @brief Conditional selection, comparison, and logical operations
 * @see Arithmetic.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Arithmetic.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <type_traits>
#include <utility>

namespace metis {

// --- Trait to deduce the boolean type for a scalar ---
template <typename T> struct BooleanType {
    using type = bool;
};

template <> struct BooleanType<SymbolicScalar> {
    using type = SymbolicScalar;
};

template <typename T> using BooleanType_t = typename BooleanType<T>::type;

// --- where (Scalar) ---
/**
 * @brief Select values based on condition (ternary operator)
 * Returns: cond ? if_true : if_false
 * Supports mixed types.
 *
 * @param cond Condition
 * @param if_true Value if true
 * @param if_false Value if false
 * @return Selected value
 */
// Relaxed to allow mixed types (e.g. MX and double)
template <typename Cond, MetisScalar T1, MetisScalar T2>
auto where(const Cond &cond, const T1 &if_true, const T2 &if_false) {
    if constexpr (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>) {
        return cond ? if_true : if_false;
    } else {
        // Assume non-floating point (e.g. CasADi) handles mixed ops
        return if_else(cond, if_true, if_false);
    }
}

namespace detail {

template <typename Cond, typename DerivedTrue, typename DerivedFalse>
auto select(const Cond &cond, const Eigen::MatrixBase<DerivedTrue> &if_true,
            const Eigen::MatrixBase<DerivedFalse> &if_false) {
    if (if_true.rows() != if_false.rows() || if_true.cols() != if_false.cols()) {
        throw InvalidArgument("select: matrix inputs must have the same shape");
    }

    using ResultScalar = std::decay_t<decltype(metis::where(
        std::declval<Cond>(), std::declval<typename DerivedTrue::Scalar>(),
        std::declval<typename DerivedFalse::Scalar>()))>;
    using ResultMatrix =
        Eigen::Matrix<ResultScalar, DerivedTrue::RowsAtCompileTime, DerivedTrue::ColsAtCompileTime,
                      DerivedTrue::Options, DerivedTrue::MaxRowsAtCompileTime,
                      DerivedTrue::MaxColsAtCompileTime>;

    ResultMatrix res(if_true.rows(), if_true.cols());
    for (Eigen::Index i = 0; i < if_true.rows(); ++i) {
        for (Eigen::Index j = 0; j < if_true.cols(); ++j) {
            res(i, j) = metis::where(cond, if_true(i, j), if_false(i, j));
        }
    }
    return res;
}

inline bool is_symbolic_predicate(const SymbolicScalar &expr) {
    if (expr.n_dep() == 0) {
        return false;
    }

    switch (expr.op()) {
    case casadi::OP_LT:
    case casadi::OP_LE:
    case casadi::OP_EQ:
    case casadi::OP_NE:
    case casadi::OP_AND:
    case casadi::OP_OR:
    case casadi::OP_NOT:
        return true;
    default:
        return false;
    }
}

inline SymbolicScalar as_symbolic_predicate(const SymbolicScalar &expr) {
    return is_symbolic_predicate(expr) ? expr : expr != 0;
}

template <typename Derived>
SymbolicScalar count_symbolic_truthy(const Eigen::MatrixBase<Derived> &a) {
    SymbolicScalar result = 0.0;
    for (Eigen::Index i = 0; i < a.rows(); ++i) {
        for (Eigen::Index j = 0; j < a.cols(); ++j) {
            result = result + as_symbolic_predicate(a(i, j));
        }
    }
    return result;
}

} // namespace detail

// --- where (Vector/Matrix) ---
/**
 * @brief Element-wise select
 *
 * @param cond Condition matrix/array
 * @param if_true Matrix of values if true
 * @param if_false Matrix of values if false
 * @return Result matrix
 */
template <typename DerivedCond, typename DerivedTrue, typename DerivedFalse>
auto where(const Eigen::ArrayBase<DerivedCond> &cond, const Eigen::MatrixBase<DerivedTrue> &if_true,
           const Eigen::MatrixBase<DerivedFalse> &if_false) {
    using Scalar = typename DerivedTrue::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        // Manual element-wise select for generic types/CasADi
        // Assuming if_true and if_false have same dimensions as cond
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> res(if_true.rows(), if_true.cols());
        for (Eigen::Index i = 0; i < if_true.rows(); ++i) {
            for (Eigen::Index j = 0; j < if_true.cols(); ++j) {
                // cond(i,j) might be an expression, evaluate it.
                res(i, j) = metis::where(cond.derived().coeff(i, j), if_true(i, j), if_false(i, j));
            }
        }
        return res;
    } else {
        return cond.select(if_true, if_false);
    }
}

// --- Min ---
/**
 * @brief Computes minimum of two values
 * @param a First value
 * @param b Second value
 * @return Minimum value
 */
// Relaxed for mixed types
template <MetisScalar T1, MetisScalar T2> auto min(const T1 &a, const T2 &b) {
    if constexpr (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>) {
        return std::min(a, b);
    } else {
        // use fmin for mixed (fmin(double, MX) works in CasADi)
        // std::min(double, MX) does NOT work usually?
        // Actually, CasADi overloads fmin.
        return fmin(a, b);
    }
}

/**
 * @brief Computes minimum element-wise for a matrix/vector
 * @tparam Derived Eigen matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Matrix of minimums
 */
template <typename Derived>
auto min(const Eigen::MatrixBase<Derived> &a, const Eigen::MatrixBase<Derived> &b) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> res(a.rows(),
                                                                                          a.cols());
        for (Eigen::Index i = 0; i < a.rows(); ++i) {
            for (Eigen::Index j = 0; j < a.cols(); ++j) {
                res(i, j) = metis::min(a(i, j), b(i, j));
            }
        }
        return res;
    } else {
        return a.cwiseMin(b);
    }
}

// --- Max ---
/**
 * @brief Computes maximum of two values
 * @param a First value
 * @param b Second value
 * @return Maximum value
 */
// Relaxed for mixed types
template <MetisScalar T1, MetisScalar T2> auto max(const T1 &a, const T2 &b) {
    if constexpr (std::is_floating_point_v<T1> && std::is_floating_point_v<T2>) {
        return std::max(a, b);
    } else {
        return fmax(a, b);
    }
}

/**
 * @brief Computes maximum element-wise for a matrix/vector
 * @tparam Derived Eigen matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Matrix of maximums
 */
template <typename Derived>
auto max(const Eigen::MatrixBase<Derived> &a, const Eigen::MatrixBase<Derived> &b) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> res(a.rows(),
                                                                                          a.cols());
        for (Eigen::Index i = 0; i < a.rows(); ++i) {
            for (Eigen::Index j = 0; j < a.cols(); ++j) {
                res(i, j) = metis::max(a(i, j), b(i, j));
            }
        }
        return res;
    } else {
        return a.cwiseMax(b);
    }
}

// --- Clamp ---
/**
 * @brief Clamps value between low and high
 * @param val Input value
 * @param low Lower bound
 * @param high Upper bound
 * @return Clamped value
 */
// Relaxed for mixed types
template <MetisScalar T, MetisScalar TLow, MetisScalar THigh>
auto clamp(const T &val, const TLow &low, const THigh &high) {
    return metis::min(metis::max(val, low), high);
}

/**
 * @brief Clamps values element-wise for a matrix/vector
 * @tparam Derived Eigen matrix type
 * @tparam Scalar Bounds type
 * @param val Input matrix
 * @param low Lower bound
 * @param high Upper bound
 * @return Matrix of clamped values
 */
template <typename Derived, typename Scalar>
auto clamp(const Eigen::MatrixBase<Derived> &val, const Scalar &low, const Scalar &high) {
    using MatrixScalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<MatrixScalar, SymbolicScalar>) {
        return val.unaryExpr([=](const auto &x) { return metis::clamp(x, low, high); });
    } else {
        return val.cwiseMax(low).cwiseMin(high);
    }
}

// --- Less Than (lt) ---
/**
 * @brief Element-wise less than comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a < b
 */
// Returns expressions suitable for 'where' condition
template <typename DerivedA, typename DerivedB>
auto lt(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x < y; });
    } else {
        return (a.array() < b.array());
    }
}

// --- Greater Than (gt) ---
/**
 * @brief Element-wise greater than comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a > b
 */
template <typename DerivedA, typename DerivedB>
auto gt(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x > y; });
    } else {
        return (a.array() > b.array());
    }
}

// --- Less Than or Equal (le) ---
/**
 * @brief Element-wise less than or equal comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a <= b
 */
template <typename DerivedA, typename DerivedB>
auto le(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x <= y; });
    } else {
        return (a.array() <= b.array());
    }
}

// --- Greater Than or Equal (ge) ---
/**
 * @brief Element-wise greater than or equal comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a >= b
 */
template <typename DerivedA, typename DerivedB>
auto ge(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x >= y; });
    } else {
        return (a.array() >= b.array());
    }
}

// --- Equal (eq) ---
/**
 * @brief Element-wise equality comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a == b
 */
template <typename DerivedA, typename DerivedB>
auto eq(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x == y; });
    } else {
        return (a.array() == b.array());
    }
}

// --- Not Equal (neq) ---
/**
 * @brief Element-wise inequality comparison
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask where a != b
 */
template <typename DerivedA, typename DerivedB>
auto neq(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x != y; });
    } else {
        return (a.array() != b.array());
    }
}

// --- sigmoid_blend ---
/**
 * @brief Smoothly blends between val_low and val_high using a sigmoid function
 * blend = val_low + (val_high - val_low) * (1 / (1 + exp(-sharpness * x)))
 *
 * @param x Control variable (0 centers the sigmoid)
 * @param val_low Value when x is negative large
 * @param val_high Value when x is positive large
 * @param sharpness Steepness of the transition
 * @return Blended value
 */
// Relaxed for mixed types
template <MetisScalar T, MetisScalar TLow, MetisScalar THigh, MetisScalar Sharpness = double>
auto sigmoid_blend(const T &x, const TLow &val_low, const THigh &val_high,
                   const Sharpness &sharpness = 1.0) {
    // using metis::exp from Arithmetic.hpp
    auto alpha = 1.0 / (1.0 + metis::exp(-sharpness * x));
    return val_low + alpha * (val_high - val_low);
}

// Vectorized sigmoid_blend could be added here if needed,
// strictly relying on .array() operations in implementation code might be enough
// if we make a vectorized wrapper like in Arithmetic.hpp

/**
 * @brief Smoothly blends element-wise for a matrix using a sigmoid function
 * @tparam Derived Eigen matrix type
 * @tparam Scalar Scalar type
 * @param x Control variable matrix
 * @param val_low Low value
 * @param val_high High value
 * @param sharpness Steepness
 * @return Matrix of blended values
 */
template <typename Derived, typename Scalar>
auto sigmoid_blend(const Eigen::MatrixBase<Derived> &x, const Scalar &val_low,
                   const Scalar &val_high, const Scalar &sharpness = 1.0) {
    auto alpha = (1.0 + (-sharpness * x.array()).exp()).inverse();
    return (val_low + alpha * (val_high - val_low)).matrix();
}

// --- Logical AND ---
/**
 * @brief Logical AND (x && y)
 * @param x1 First operand
 * @param x2 Second operand
 * @return Boolean result (numeric) or symbolic expression
 */
template <MetisScalar T1, MetisScalar T2> auto logical_and(const T1 &x1, const T2 &x2) {
    // Both define operator &&
    return x1 && x2;
}

/**
 * @brief Element-wise logical AND for matrices
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask
 */
template <typename DerivedA, typename DerivedB>
auto logical_and(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x && y; });
    } else {
        // Ensure boolean context for Eigen arrays
        return ((a.array() != 0) && (b.array() != 0));
    }
}

// --- Logical OR ---
/**
 * @brief Logical OR (x || y)
 * @param x1 First operand
 * @param x2 Second operand
 * @return Boolean result (numeric) or symbolic expression
 */
template <MetisScalar T1, MetisScalar T2> auto logical_or(const T1 &x1, const T2 &x2) {
    return x1 || x2;
}

/**
 * @brief Element-wise logical OR for matrices
 * @tparam DerivedA First matrix type
 * @tparam DerivedB Second matrix type
 * @param a First matrix
 * @param b Second matrix
 * @return Boolean expression/mask
 */
template <typename DerivedA, typename DerivedB>
auto logical_or(const Eigen::MatrixBase<DerivedA> &a, const Eigen::MatrixBase<DerivedB> &b) {
    using Scalar = typename DerivedA::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.binaryExpr(b, [](const auto &x, const auto &y) { return x || y; });
    } else {
        return ((a.array() != 0) || (b.array() != 0));
    }
}

// --- Logical NOT ---
/**
 * @brief Logical NOT (!x)
 * @param x Operand
 * @return Boolean result (numeric) or symbolic expression
 */
template <MetisScalar T> auto logical_not(const T &x) { return !x; }

/**
 * @brief Element-wise logical NOT for a matrix
 * @tparam Derived Eigen matrix type
 * @param a Input matrix
 * @return Boolean expression/mask
 */
template <typename Derived> auto logical_not(const Eigen::MatrixBase<Derived> &a) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return a.unaryExpr([](const auto &x) { return !x; });
    } else {
        return (a.array() == 0);
    }
}

// --- All ---
/**
 * @brief Returns true if all elements are true (non-zero)
 * @param a Input matrix/array
 * @return Boolean (numeric) or symbolic expression
 */
template <typename Derived> auto all(const Eigen::MatrixBase<Derived> &a) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return detail::count_symbolic_truthy(a) == static_cast<double>(a.size());
    } else {
        return (a.array() != 0).all();
    }
}

// --- Any ---
/**
 * @brief Returns true if any element is true (non-zero)
 * @param a Input matrix/array
 * @return Boolean (numeric) or symbolic expression
 */
template <typename Derived> auto any(const Eigen::MatrixBase<Derived> &a) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        return detail::count_symbolic_truthy(a) >= 1.0;
    } else {
        return (a.array() != 0).any();
    }
}

// --- Select (Multi-way branching like switch/case) ---
/**
 * @brief Multi-way conditional selection (cleaner alternative to nested where)
 *
 * Evaluates conditions in order and returns the first matching value.
 * Similar to NumPy's select() or a switch statement.
 *
 * Example:
 *   select({x < 0, x < 10, x < 100}, {-1, 0, 1}, 2)
 *   // Returns: -1 if x<0, 0 if 0<=x<10, 1 if 10<=x<100, else 2
 *
 * @tparam CondType Condition type (result of comparison, e.g., bool or Scalar)
 * @tparam Scalar Scalar type for values (numeric or symbolic)
 * @param conditions Vector of conditions to check (in order)
 * @param values Vector of values to return (same size as conditions)
 * @param default_value Value to return if no condition matches
 * @return Result of first matching condition, or default
 */
template <typename CondType, typename Scalar>
Scalar select(const std::vector<CondType> &conditions, const std::vector<Scalar> &values,
              const Scalar &default_value) {
    if (conditions.size() != values.size()) {
        throw InvalidArgument("select: conditions and values must have same size");
    }

    // Start with default
    Scalar result = default_value;

    // Work backwards so earlier conditions override later ones
    for (int i = static_cast<int>(conditions.size()) - 1; i >= 0; --i) {
        result = where(conditions[i], values[i], result);
    }

    return result;
}

// Overload for initializer lists (cleaner syntax)
template <typename CondType, typename Scalar>
Scalar select(std::initializer_list<CondType> conditions, std::initializer_list<Scalar> values,
              const Scalar &default_value) {
    return select(std::vector<CondType>(conditions), std::vector<Scalar>(values), default_value);
}

} // namespace metis
