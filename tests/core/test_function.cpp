#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Arithmetic.hpp>
#include <metis/math/AutoDiff.hpp>
#include <metis/math/Trig.hpp>
#include <vector>

TEST(FunctionTests, BasicEvaluation) {
    // f(x, y) = x * y + x
    auto x = metis::sym("x");
    auto y = metis::sym("y");
    auto f_sym = x * y + x;

    metis::Function f("f", {x, y}, {f_sym});

    // Test variadic scalar evaluation
    auto res1 = f(2.0, 3.0);
    // 2*3 + 2 = 8
    ASSERT_EQ(res1.size(), 1);
    EXPECT_NEAR(res1[0](0, 0), 8.0, 1e-9);

    // Test vector evaluation
    std::vector<double> args = {3.0, 4.0};
    auto res2 = f(args);
    // 3*4 + 3 = 15
    ASSERT_EQ(res2.size(), 1);
    EXPECT_NEAR(res2[0](0, 0), 15.0, 1e-9);
}

TEST(FunctionTests, VectorIO) {
    // f([x, y]) = [x+y, x-y]
    auto x = metis::sym("x");
    auto y = metis::sym("y");

    // Let's test explicit vector inputs as separate symbolic variables for now
    // to match signature: Function(..., {inputs}, {outputs})

    auto out1 = x + y;
    auto out2 = x - y;

    metis::Function f("f_vec", {x, y}, {out1, out2});

    auto res = f(10.0, 5.0);
    ASSERT_EQ(res.size(), 2);
    // out1 = 15
    EXPECT_NEAR(res[0](0, 0), 15.0, 1e-9);
    // out2 = 5
    EXPECT_NEAR(res[1](0, 0), 5.0, 1e-9);
}

TEST(FunctionTests, MatrixOutput) {
    // f(x) = [x, 2x; 3x, 4x]
    auto x = metis::sym("x");

    metis::SymbolicMatrix M(2, 2);
    M(0, 0) = x;
    M(0, 1) = 2.0 * x;
    M(1, 0) = 3.0 * x;
    M(1, 1) = 4.0 * x;

    // Convert Eigen matrix of MX to single MX for Function output
    metis::Function f("f_mat", {x}, {metis::to_mx(M)});

    auto res = f(2.0);
    ASSERT_EQ(res.size(), 1);
    const auto &R = res[0];
    EXPECT_EQ(R.rows(), 2);
    EXPECT_EQ(R.cols(), 2);

    EXPECT_NEAR(R(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(R(0, 1), 4.0, 1e-9);
    EXPECT_NEAR(R(1, 0), 6.0, 1e-9);
    EXPECT_NEAR(R(1, 1), 8.0, 1e-9);
}

TEST(FunctionTests, JacobianWrapper) {
    // Test the jacobian helper integration with Function
    auto x = metis::sym("x");
    auto fx = metis::pow(x, 2.0);    // x^2
    auto J = metis::jacobian(fx, x); // 2x

    metis::Function f_J("J", {x}, {J});

    auto res = f_J(3.0);
    EXPECT_NEAR(res[0](0, 0), 6.0, 1e-9);
}

// =============================================================================
// Lambda-Style Function Tests (make_function)
// =============================================================================

TEST(FunctionTests, LambdaSingleInSingleOut) {
    // f(x) = x^2
    auto f = metis::make_function<1, 1>("square", [](auto x) { return x * x; });

    auto res = f(3.0);
    ASSERT_EQ(res.size(), 1);
    EXPECT_NEAR(res[0](0, 0), 9.0, 1e-9);

    // Also test with negative number
    auto res2 = f(-4.0);
    EXPECT_NEAR(res2[0](0, 0), 16.0, 1e-9);
}

TEST(FunctionTests, LambdaMultiInSingleOut) {
    // f(x, y) = x + y
    auto f = metis::make_function<2, 1>("sum", [](auto x, auto y) { return x + y; });

    auto res = f(3.0, 7.0);
    ASSERT_EQ(res.size(), 1);
    EXPECT_NEAR(res[0](0, 0), 10.0, 1e-9);
}

TEST(FunctionTests, LambdaMultiInMultiOut) {
    // f(x, y) = (x+y, x-y)
    auto f = metis::make_function<2, 2>(
        "add_sub", [](auto x, auto y) { return std::make_tuple(x + y, x - y); });

    auto res = f(10.0, 3.0);
    ASSERT_EQ(res.size(), 2);
    EXPECT_NEAR(res[0](0, 0), 13.0, 1e-9); // x + y = 13
    EXPECT_NEAR(res[1](0, 0), 7.0, 1e-9);  // x - y = 7
}

TEST(FunctionTests, LambdaNamedInputs) {
    // f(x, y) = x * y using named inputs
    auto f = metis::make_function<2>("product", {"x", "y"}, [](auto x, auto y) { return x * y; });

    auto res = f(4.0, 5.0);
    ASSERT_EQ(res.size(), 1);
    EXPECT_NEAR(res[0](0, 0), 20.0, 1e-9);
}

TEST(FunctionTests, LambdaNamedMultiOut) {
    // g(a, b, c) = (a+b+c, a*b*c) with named inputs
    auto g = metis::make_function<3>("triple", {"a", "b", "c"}, [](auto a, auto b, auto c) {
        return std::make_tuple(a + b + c, a * b * c);
    });

    auto res = g(2.0, 3.0, 4.0);
    ASSERT_EQ(res.size(), 2);
    EXPECT_NEAR(res[0](0, 0), 9.0, 1e-9);  // 2 + 3 + 4 = 9
    EXPECT_NEAR(res[1](0, 0), 24.0, 1e-9); // 2 * 3 * 4 = 24
}

TEST(FunctionTests, LambdaWithMetisMath) {
    // Test using metis:: math functions inside lambda
    auto f = metis::make_function<1, 1>("exp_plus_one", [](auto x) { return metis::exp(x) + 1.0; });

    auto res = f(0.0);
    EXPECT_NEAR(res[0](0, 0), 2.0, 1e-9); // exp(0) + 1 = 2

    auto res2 = f(1.0);
    EXPECT_NEAR(res2[0](0, 0), std::exp(1.0) + 1.0, 1e-9);
}

TEST(FunctionTests, MapBatchEvaluation) {
    auto x = metis::sym("x", 2, 1);
    metis::Function f("affine_batch", {x}, {3.0 * x - 1.0});

    auto mapped = f.map(3, metis::MapParallelization::Parallel, 2);

    metis::NumericMatrix X(2, 3);
    X(0, 0) = 1.0;
    X(1, 0) = 4.0;
    X(0, 1) = 2.0;
    X(1, 1) = 5.0;
    X(0, 2) = 3.0;
    X(1, 2) = 6.0;

    auto res = mapped(X);
    ASSERT_EQ(res.size(), 1);

    const auto &Y = res[0];
    ASSERT_EQ(Y.rows(), 2);
    ASSERT_EQ(Y.cols(), 3);

    EXPECT_NEAR(Y(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(Y(1, 0), 11.0, 1e-9);
    EXPECT_NEAR(Y(0, 1), 5.0, 1e-9);
    EXPECT_NEAR(Y(1, 1), 14.0, 1e-9);
    EXPECT_NEAR(Y(0, 2), 8.0, 1e-9);
    EXPECT_NEAR(Y(1, 2), 17.0, 1e-9);
}

TEST(FunctionTests, MapPreservesSymbolicDerivatives) {
    auto x = metis::sym("x");
    metis::Function square("square_batch", {x}, {x * x});
    auto mapped = square.map(4, metis::MapParallelization::Parallel);

    auto X = metis::sym("X", 1, 4);
    metis::SymbolicScalar Y = metis::to_mx(mapped.eval(X));
    metis::SymbolicScalar J = metis::jacobian(Y, X);
    metis::Function jac_fn("mapped_square_jac", {X}, {J});

    metis::NumericMatrix Xval(1, 4);
    Xval(0, 0) = -2.0;
    Xval(0, 1) = -0.5;
    Xval(0, 2) = 1.5;
    Xval(0, 3) = 3.0;

    auto res = jac_fn(Xval);
    ASSERT_EQ(res.size(), 1);

    const auto &Jval = res[0];
    ASSERT_EQ(Jval.rows(), 4);
    ASSERT_EQ(Jval.cols(), 4);

    EXPECT_NEAR(Jval(0, 0), -4.0, 1e-9);
    EXPECT_NEAR(Jval(1, 1), -1.0, 1e-9);
    EXPECT_NEAR(Jval(2, 2), 3.0, 1e-9);
    EXPECT_NEAR(Jval(3, 3), 6.0, 1e-9);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j)
                continue;
            EXPECT_NEAR(Jval(i, j), 0.0, 1e-9);
        }
    }
}

TEST(FunctionTests, MapRejectsInvalidSizes) {
    auto x = metis::sym("x");
    metis::Function identity("identity_batch", {x}, {x});

    EXPECT_THROW(identity.map(0), metis::InvalidArgument);
    EXPECT_THROW(identity.map(-3, metis::MapParallelization::Parallel), metis::InvalidArgument);
    EXPECT_THROW(identity.map(2, metis::MapParallelization::Parallel, 0), metis::InvalidArgument);
}
