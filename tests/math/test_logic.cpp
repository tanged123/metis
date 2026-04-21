#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Linalg.hpp> // for to_mx
#include <metis/math/Logic.hpp>

namespace {

bool contains_op(const metis::SymbolicScalar &expr, casadi_int op) {
    if (expr.n_dep() > 0 && expr.op() == op) {
        return true;
    }

    for (casadi_int i = 0; i < expr.n_dep(); ++i) {
        if (contains_op(expr.dep(i), op)) {
            return true;
        }
    }

    return false;
}

} // namespace

template <typename Scalar> void test_logic_ops() {
    Scalar a = 1.0;
    Scalar b = 2.0;

    // Test where with comparison
    auto res_where = metis::where(a < b, a, b);

    // Test min/max/clamp
    auto res_min = metis::min(a, b);
    auto res_max = metis::max(a, b);

    Scalar val = 5.0;
    Scalar low = 0.0;
    Scalar high = 3.0;
    auto res_clamp = metis::clamp(val, low, high); // Should be 3.0

    // Test sigmoid blend
    Scalar val_low = 10.0;
    Scalar val_high = 20.0;
    auto blend_low = metis::sigmoid_blend(static_cast<Scalar>(-10.0), val_low, val_high);
    auto blend_high = metis::sigmoid_blend(static_cast<Scalar>(10.0), val_low, val_high);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_where, 1.0);
        EXPECT_DOUBLE_EQ(res_min, 1.0);
        EXPECT_DOUBLE_EQ(res_max, 2.0);
        EXPECT_DOUBLE_EQ(res_clamp, 3.0);
        EXPECT_NEAR(blend_low, 10.0, 1e-3);
        EXPECT_NEAR(blend_high, 20.0, 1e-3);
    } else {
        EXPECT_DOUBLE_EQ(metis::eval(res_where), 1.0);
        EXPECT_DOUBLE_EQ(metis::eval(res_min), 1.0);
        EXPECT_DOUBLE_EQ(metis::eval(res_max), 2.0);
        EXPECT_DOUBLE_EQ(metis::eval(res_clamp), 3.0);
        EXPECT_NEAR(metis::eval(blend_low), 10.0, 1e-3);
        EXPECT_NEAR(metis::eval(blend_high), 20.0, 1e-3);
    }
}

template <typename Scalar> void test_logic_matrix() {
    using Matrix = metis::MetisMatrix<Scalar>;
    Matrix A(2, 2);
    A << 1.0, 4.0, 2.0, 5.0;
    Matrix B(2, 2);
    B << 3.0, 2.0, 1.0, 6.0;

    // Element-wise min: [1, 2], [1, 5]
    auto M = metis::min(A, B);
    auto cond = metis::lt(A, B);

    // For numeric, cond is Array<bool>. For symbolic, Matrix<MX>.
    // metis::where expects ArrayBase.
    // If Matrix<MX>, .array() makes it ArrayWrapper, which is ArrayBase.
    // If Array<bool>, it is ArrayBase.

    // However, cond.array() works for both if we ensure cond is expression that supports .array().
    // Eigen binaryExpr returns CwiseBinaryOp which supports .array() if it's Matrix expression?
    // Actually, binaryExpr on Matrix returns Matrix.
    // Comparison on Array returns Array.

    // Let's use auto and .array() or pass derived if compatible.
    // Our where loop calls .coeff().

    auto where_mat = metis::where(cond.array(), A, B);

    // Test new comparisons
    // A: [[1, 4], [2, 5]]
    // B: [[3, 2], [1, 6]]

    // A > B: [[F, T], [T, F]] -> where(A>B, 10, -10)
    auto cond_gt = metis::gt(A, B);
    Matrix Ones = Matrix::Ones(2, 2) * 10;
    Matrix NegOnes = Matrix::Ones(2, 2) * -10;
    auto check_gt = metis::where(cond_gt.array(), Ones, NegOnes);

    // A <= B: [[T, F], [F, T]] (Inverse of >)
    auto cond_le = metis::le(A, B);
    auto check_le = metis::where(cond_le.array(), Ones, NegOnes);

    // A == A
    auto cond_eq = metis::eq(A, A);
    auto check_eq = metis::where(cond_eq.array(), Ones, NegOnes);

    // A != B (All true as scalars different everywhere)
    auto cond_neq = metis::neq(A, B);
    auto check_neq = metis::where(cond_neq.array(), Ones, NegOnes);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(M(0, 0), 1.0);
        EXPECT_DOUBLE_EQ(M(0, 1), 2.0);
        EXPECT_DOUBLE_EQ(M(1, 0), 1.0);
        EXPECT_DOUBLE_EQ(M(1, 1), 5.0);

        EXPECT_DOUBLE_EQ(where_mat(0, 1), 2.0);

        // gt: [F, T], [T, F] -> [-10, 10], [10, -10]
        EXPECT_DOUBLE_EQ(check_gt(0, 0), -10.0);
        EXPECT_DOUBLE_EQ(check_gt(0, 1), 10.0);
        EXPECT_DOUBLE_EQ(check_gt(1, 0), 10.0);
        EXPECT_DOUBLE_EQ(check_gt(1, 1), -10.0);

        // le: Inverse of gt
        EXPECT_DOUBLE_EQ(check_le(0, 0), 10.0);
        EXPECT_DOUBLE_EQ(check_le(0, 1), -10.0);

        // eq: All true
        EXPECT_DOUBLE_EQ(check_eq(0, 0), 10.0);

        // neq: All true
        EXPECT_DOUBLE_EQ(check_neq(0, 0), 10.0);
    } else {
        auto M_eval = metis::eval(M);
        EXPECT_DOUBLE_EQ(M_eval(0, 0), 1.0);
        EXPECT_DOUBLE_EQ(M_eval(0, 1), 2.0);
        EXPECT_DOUBLE_EQ(M_eval(1, 0), 1.0);
        EXPECT_DOUBLE_EQ(M_eval(1, 1), 5.0);

        auto W_eval = metis::eval(where_mat);
        EXPECT_DOUBLE_EQ(W_eval(0, 1), 2.0);

        auto G_eval = metis::eval(check_gt);
        EXPECT_DOUBLE_EQ(G_eval(0, 0), -10.0);
        EXPECT_DOUBLE_EQ(G_eval(0, 1), 10.0);

        auto L_eval = metis::eval(check_le);
        EXPECT_DOUBLE_EQ(L_eval(0, 0), 10.0);
        EXPECT_DOUBLE_EQ(L_eval(0, 1), -10.0);

        auto E_eval = metis::eval(check_eq);
        EXPECT_DOUBLE_EQ(E_eval(0, 0), 10.0);

        auto N_eval = metis::eval(check_neq);
        EXPECT_DOUBLE_EQ(N_eval(0, 0), 10.0);
    }
}

template <typename Scalar> void test_extended_logic() {
    using Matrix = metis::MetisMatrix<Scalar>;
    Matrix A(2, 2);
    A << 1.0, 0.0, 1.0, 1.0;
    Matrix B(2, 2);
    B << 1.0, 1.0, 0.0, 1.0;

    // AND: [1, 0; 0, 1]
    // OR:  [1, 1; 1, 1]
    auto res_and = metis::logical_and(A, B);
    auto res_or = metis::logical_or(A, B);

    // NOT A: [0, 1; 0, 0]
    auto res_not = metis::logical_not(A);

    // All/Any
    // A has zeros -> all = false, any = true
    auto res_all = metis::all(A);
    auto res_any = metis::any(A);

    // Clamp
    Matrix C(2, 2);
    C << -5.0, 5.0, 0.0, 10.0;
    auto res_clip = metis::clamp(C, 0.0, 2.0); // -> [0, 2; 0, 2]

    if constexpr (std::is_same_v<Scalar, double>) {
        // Numeric checks
        // AND
        EXPECT_TRUE((bool)res_and(0, 0));
        EXPECT_FALSE((bool)res_and(0, 1));
        EXPECT_FALSE((bool)res_and(1, 0));
        EXPECT_TRUE((bool)res_and(1, 1));

        // OR
        EXPECT_TRUE((bool)res_or(0, 0));
        EXPECT_TRUE((bool)res_or(0, 1));

        // NOT
        EXPECT_FALSE((bool)res_not(0, 0));
        EXPECT_TRUE((bool)res_not(0, 1));

        // All/Any
        EXPECT_FALSE(res_all);
        EXPECT_TRUE(res_any);

        // Clip
        EXPECT_DOUBLE_EQ(res_clip(0, 0), 0.0);
        EXPECT_DOUBLE_EQ(res_clip(0, 1), 2.0);
        EXPECT_DOUBLE_EQ(res_clip(1, 0), 0.0);
        EXPECT_DOUBLE_EQ(res_clip(1, 1), 2.0);

        // Scalar Logic
        EXPECT_TRUE(metis::logical_and(1.0, 1.0));
        EXPECT_FALSE(metis::logical_and(1.0, 0.0));
        EXPECT_TRUE(metis::logical_or(0.0, 1.0));
        EXPECT_FALSE(metis::logical_not(1.0));

    } else {
        // Symbolic checks
        auto and_eval = metis::eval(res_and);
        EXPECT_NEAR(and_eval(0, 0), 1.0, 1e-9);
        EXPECT_NEAR(and_eval(0, 1), 0.0, 1e-9);

        auto or_eval = metis::eval(res_or);
        EXPECT_NEAR(or_eval(0, 0), 1.0, 1e-9);

        auto not_eval = metis::eval(res_not);
        EXPECT_NEAR(not_eval(0, 0), 0.0, 1e-9);
        EXPECT_NEAR(not_eval(0, 1), 1.0, 1e-9);

        auto all_val = metis::eval(res_all);
        EXPECT_NEAR(all_val, 0.0, 1e-9);

        auto any_val = metis::eval(res_any);
        EXPECT_NEAR(any_val, 1.0, 1e-9);

        auto clip_eval = metis::eval(res_clip);
        EXPECT_NEAR(clip_eval(0, 0), 0.0, 1e-9);
        EXPECT_NEAR(clip_eval(0, 1), 2.0, 1e-9);

        // Scalar Logic
        EXPECT_NEAR(
            metis::eval(metis::logical_and(metis::SymbolicScalar(1.0), metis::SymbolicScalar(1.0))),
            1.0, 1e-9);
        EXPECT_NEAR(
            metis::eval(metis::logical_and(metis::SymbolicScalar(1.0), metis::SymbolicScalar(0.0))),
            0.0, 1e-9);
    }
}

TEST(LogicTests, Numeric) {
    test_logic_ops<double>();
    test_logic_matrix<double>();
    test_extended_logic<double>();
}

TEST(LogicTests, Symbolic) {
    test_logic_ops<metis::SymbolicScalar>();
    test_logic_matrix<metis::SymbolicScalar>();
}

TEST(LogicTests, ExtendedLogicSymbolic) { test_extended_logic<metis::SymbolicScalar>(); }

// --- Tests for select() function ---

template <typename Scalar> void test_select() {
    // Test 1: Basic 3-way select
    Scalar x = 15.0;
    auto result1 = metis::select({x < 10.0, x < 20.0, x < 30.0},
                                 {Scalar(1.0), Scalar(2.0), Scalar(3.0)}, Scalar(4.0));

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(result1, 2.0); // 10 <= 15 < 20
    } else {
        EXPECT_DOUBLE_EQ(metis::eval(result1), 2.0);
    }

    // Test 2: First condition matches
    Scalar y = 5.0;
    auto result2 =
        metis::select({y < 10.0, y < 20.0}, {Scalar(100.0), Scalar(200.0)}, Scalar(300.0));

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(result2, 100.0);
    } else {
        EXPECT_DOUBLE_EQ(metis::eval(result2), 100.0);
    }

    // Test 3: Default value (no condition matches)
    Scalar z = 100.0;
    auto result3 = metis::select({z < 10.0, z < 20.0, z < 30.0},
                                 {Scalar(1.0), Scalar(2.0), Scalar(3.0)}, Scalar(99.0));

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(result3, 99.0); // Default
    } else {
        EXPECT_DOUBLE_EQ(metis::eval(result3), 99.0);
    }

    // Test 4: Single condition
    Scalar w = 15.0;
    auto result4 = metis::select({w > 10.0}, {Scalar(42.0)}, Scalar(7.0));

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(result4, 42.0);
    } else {
        EXPECT_DOUBLE_EQ(metis::eval(result4), 42.0);
    }
}

TEST(LogicTests, SelectNumeric) { test_select<double>(); }

TEST(LogicTests, SelectSymbolic) { test_select<metis::SymbolicScalar>(); }

// Test select() vs nested where() equivalence
TEST(LogicTests, SelectEquivalence) {
    double x = 15.0;

    // Using select
    auto result_select = metis::select({x < 10.0, x < 20.0, x < 30.0}, {1.0, 2.0, 3.0}, 4.0);

    // Using nested where
    auto result_where =
        metis::where(x < 10.0, 1.0, metis::where(x < 20.0, 2.0, metis::where(x < 30.0, 3.0, 4.0)));

    EXPECT_DOUBLE_EQ(result_select, result_where);
}

TEST(LogicTests, SelectError) {
    EXPECT_THROW(metis::select({true, false}, {1.0}, 0.0), metis::InvalidArgument);
}

TEST(LogicTests, SelectInitializerList) {
    double x = 5.0;
    auto res = metis::select({x < 0.0, x > 0.0}, {-1.0, 1.0}, 0.0);
    EXPECT_DOUBLE_EQ(res, 1.0);
}

TEST(LogicTests, NumericAllAny) {
    metis::NumericMatrix M(2, 2);
    M << 1, 0, 1, 1;
    // all should be false
    EXPECT_FALSE(metis::all(M));
    // any should be true
    EXPECT_TRUE(metis::any(M));

    metis::NumericMatrix AllOnes = metis::NumericMatrix::Ones(2, 2);
    EXPECT_TRUE(metis::all(AllOnes));

    metis::NumericMatrix AllZeros = metis::NumericMatrix::Zero(2, 2);
    EXPECT_FALSE(metis::any(AllZeros));
}

TEST(LogicTests, SymbolicAllAnyTruthiness) {
    auto [v, v_mx] = metis::sym_vec_pair("v", 2);
    metis::Function f({v_mx}, {metis::all(v), metis::any(v)});

    metis::NumericVector both_nonzero(2);
    both_nonzero << 2.0, -3.0;
    auto both_nonzero_eval = f(both_nonzero);
    EXPECT_DOUBLE_EQ(both_nonzero_eval[0](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(both_nonzero_eval[1](0, 0), 1.0);

    metis::NumericVector one_zero(2);
    one_zero << 2.0, 0.0;
    auto one_zero_eval = f(one_zero);
    EXPECT_DOUBLE_EQ(one_zero_eval[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(one_zero_eval[1](0, 0), 1.0);

    metis::NumericVector all_zero(2);
    all_zero << 0.0, 0.0;
    auto all_zero_eval = f(all_zero);
    EXPECT_DOUBLE_EQ(all_zero_eval[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(all_zero_eval[1](0, 0), 0.0);
}

TEST(LogicTests, SymbolicAllAnyUseConstraintReductionOps) {
    auto v = metis::sym_vec("v", 2);
    auto all_expr = metis::all(v);
    auto any_expr = metis::any(v);

    EXPECT_TRUE(contains_op(all_expr, casadi::OP_ADD));
    EXPECT_FALSE(contains_op(all_expr, casadi::OP_NORMINF));

    EXPECT_TRUE(contains_op(any_expr, casadi::OP_ADD));
    EXPECT_FALSE(contains_op(any_expr, casadi::OP_NORMINF));
}

TEST(LogicTests, MixedTypeLogic) {
    // Test mixed double/Symbolic for min/max/clamp to hit those branches
    double d = 1.0;
    metis::SymbolicScalar s(2.0);

    auto res_min = metis::min(d, s); // Should convert to fmin(double, MX) -> MX
    auto res_max = metis::max(d, s);

    EXPECT_DOUBLE_EQ(metis::eval(res_min), 1.0);
    EXPECT_DOUBLE_EQ(metis::eval(res_max), 2.0);

    // Test clamp with mixed
    auto res_clamp = metis::clamp(metis::SymbolicScalar(5.0), 0.0, 3.0);
    EXPECT_DOUBLE_EQ(metis::eval(res_clamp), 3.0);
}
