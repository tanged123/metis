#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/Function.hpp> // Function is in core/Function.hpp not MetisFunction.hpp
#include <metis/math/RootFinding.hpp>

using namespace metis;

TEST(RootFindingTest, NumericSimpleQuadratic) {
    // F(x) = x^2 - 4 = 0
    auto x = sym("x");
    auto f_expr = x * x - 4.0;
    Function f("f", {x}, {f_expr});

    Eigen::VectorXd x0(1);
    x0 << 1.0; // Guess closer to 2

    auto res = rootfinder<double>(f, x0);

    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.method, RootSolveMethod::TrustRegionNewton);
    EXPECT_NEAR(res.x(0), 2.0, 1e-6);
    EXPECT_LT(res.residual_norm, 1e-8);

    // Guess closer to -2
    x0 << -1.0;
    res = rootfinder<double>(f, x0);
    EXPECT_EQ(res.method, RootSolveMethod::TrustRegionNewton);
    EXPECT_NEAR(res.x(0), -2.0, 1e-6);
}

TEST(RootFindingTest, NumericSystem2D) {
    // F(x, y) = [x^2 + y^2 - 1, x - y] = 0
    // Intersection of unit circle and line y=x.
    // Solutions: (sqrt(0.5), sqrt(0.5)) and (-sqrt(0.5), -sqrt(0.5))

    auto x = sym("x");
    auto y = sym("y");
    auto r1 = x * x + y * y - 1.0;
    auto r2 = x - y;

    // We need input as a single vector or multiple args?
    // rootfinder expects F taking 1 input (vector x).
    // So we need to vertically concatenate x and y?
    // Or define F taking vector.

    auto xy = sym("xy", 2);
    auto r1_vec = xy(0) * xy(0) + xy(1) * xy(1) - 1.0;
    auto r2_vec = xy(0) - xy(1);

    Function f("f", {xy}, {casadi::MX::vertcat({r1_vec, r2_vec})});

    Eigen::VectorXd guess(2);
    guess << 0.5, 0.5;

    RootFinderOptions opts;
    opts.strategy = RootSolveStrategy::LineSearchNewton;
    auto res = rootfinder<double>(f, guess, opts);

    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.method, RootSolveMethod::LineSearchNewton);
    double expected = std::sqrt(0.5);
    EXPECT_NEAR(res.x(0), expected, 1e-6);
    EXPECT_NEAR(res.x(1), expected, 1e-6);
}

TEST(RootFindingTest, NumericQuasiNewtonBroyden) {
    auto x = sym("x");
    auto f_expr = x * x - 9.0;
    Function f("f_broyden", {x}, {f_expr});

    Eigen::VectorXd x0(1);
    x0 << 1.0;

    RootFinderOptions opts;
    opts.strategy = RootSolveStrategy::QuasiNewtonBroyden;
    auto res = rootfinder<double>(f, x0, opts);

    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.method, RootSolveMethod::QuasiNewtonBroyden);
    EXPECT_NEAR(res.x(0), 3.0, 1e-6);
}

TEST(RootFindingTest, NumericPseudoTransientContinuation) {
    auto x = sym("x");
    auto f_expr = x * x - 1.0;
    Function f("f_pseudo", {x}, {f_expr});

    Eigen::VectorXd x0(1);
    x0 << 0.0; // Singular Jacobian for Newton at the initial guess

    RootFinderOptions opts;
    opts.strategy = RootSolveStrategy::PseudoTransientContinuation;
    opts.max_iter = 60;
    opts.pseudo_transient_dt0 = 0.1;
    auto res = rootfinder<double>(f, x0, opts);

    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.method, RootSolveMethod::PseudoTransientContinuation);
    EXPECT_NEAR(res.x(0), 1.0, 1e-6);
}

TEST(RootFindingTest, AutoFallbackUsesPseudoTransientContinuation) {
    auto x = sym("x");
    auto f_expr = x * x - 1.0;
    Function f("f_auto_fallback", {x}, {f_expr});

    Eigen::VectorXd x0(1);
    x0 << 0.0; // Trust-region, line-search, and Broyden all stall here on their first step

    RootFinderOptions opts;
    opts.max_iter = 60;
    opts.pseudo_transient_dt0 = 0.1;
    auto res = rootfinder<double>(f, x0, opts);

    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.method, RootSolveMethod::PseudoTransientContinuation);
    EXPECT_GT(res.iterations, 1);
    EXPECT_NEAR(res.x(0), 1.0, 1e-6);
}

TEST(RootFindingTest, SymbolicGraph) {
    auto x = sym("x");
    auto f_expr = x * x - 9.0;
    Function f("f", {x}, {f_expr});

    SymbolicMatrix x0(1, 1);
    x0(0, 0) = casadi::MX(2.0); // Symbolic scalar 2.0

    auto res = rootfinder<SymbolicScalar>(f, x0);

    // res.x is SymbolicVector (Eigen<MX>)
    // Evaluate it
    EXPECT_NEAR(eval_scalar(res.x(0)), 3.0, 1e-6);
}

TEST(RootFindingTest, ImplicitFunctionCreation) {
    // G(x, p) = x^2 - p = 0  => x = sqrt(p)
    auto x = sym("x");
    auto p = sym("p");
    auto g_expr = x * x - p;
    Function g("g", {x, p}, {g_expr});

    Eigen::VectorXd x_guess(1);
    x_guess << 1.0;

    auto implicit_fn = create_implicit_function(g, x_guess);

    // Eval at p=16 -> x=4
    Eigen::VectorXd p_val(1);
    p_val << 16.0;

    auto res = implicit_fn.eval(p_val);

    EXPECT_NEAR(res(0), 4.0, 1e-6);
}

TEST(RootFindingTest, ImplicitFunctionDerivative) {
    // Check if derivatives propagate through the implicit function
    // x = sqrt(p) -> dx/dp = 1/(2*sqrt(p))
    // At p=4, x=2, dx/dp = 1/4 = 0.25

    auto x = sym("x");
    auto p = sym("p");
    auto g_expr = x * x - p;
    Function g("g", {x, p}, {g_expr});

    Eigen::VectorXd x_guess(1);
    x_guess << 1.0;

    auto implicit_fn = create_implicit_function(g, x_guess);

    auto p_sym = sym("p_sym");
    auto x_sym = implicit_fn(p_sym)[0];

    auto jac = casadi::MX::jacobian(metis::to_mx(x_sym), p_sym);

    casadi::Function j_fn("j_fn", {p_sym}, {jac});
    auto j_val = j_fn(std::vector<casadi::DM>{casadi::DM(4.0)});

    EXPECT_NEAR(double(j_val[0]), 0.25, 1e-6);
}

TEST(RootFindingTest, ImplicitFunctionMultiParameterDerivative) {
    // G(x, a, b) = x^2 + a*x - b = 0
    // At (a, b) = (3, 4), the positive root is x = 1,
    // dx/da = -x / (2x + a) = -0.2 and dx/db = 1 / (2x + a) = 0.2.
    auto x = sym("x");
    auto a = sym("a");
    auto b = sym("b");
    auto residual = x * x + a * x - b;
    Function g("g_parametric", {x, a, b}, {residual});

    Eigen::VectorXd x_guess(1);
    x_guess << 1.0;

    auto implicit_fn = create_implicit_function(g, x_guess);

    auto res = implicit_fn.eval(3.0, 4.0);
    EXPECT_NEAR(res(0), 1.0, 1e-6);

    auto a_sym = sym("a_sym");
    auto b_sym = sym("b_sym");
    auto x_sym = implicit_fn(a_sym, b_sym)[0];
    auto ab_sym = casadi::MX::vertcat({a_sym, b_sym});
    auto jac = casadi::MX::jacobian(metis::to_mx(x_sym), ab_sym);
    casadi::Function j_fn("j_fn_multi", {a_sym, b_sym}, {jac});
    auto j_val = j_fn(std::vector<casadi::DM>{casadi::DM(3.0), casadi::DM(4.0)});
    std::vector<double> jac_vals = std::vector<double>(j_val[0]);

    ASSERT_EQ(jac_vals.size(), 2u);
    EXPECT_NEAR(jac_vals[0], -0.2, 1e-6);
    EXPECT_NEAR(jac_vals[1], 0.2, 1e-6);
}

TEST(RootFindingTest, ImplicitFunctionCustomSlots) {
    // Same implicit equation as above, but the unknown lives in input slot 1
    // and the residual lives in output slot 1.
    auto a = sym("a");
    auto x = sym("x");
    auto b = sym("b");
    auto aux = a - b;
    auto residual = x * x + a * x - b;
    Function g("g_custom_slots", {a, x, b}, {aux, residual});

    Eigen::VectorXd x_guess(1);
    x_guess << 1.0;

    ImplicitFunctionOptions implicit_opts;
    implicit_opts.implicit_input_index = 1;
    implicit_opts.implicit_output_index = 1;
    auto implicit_fn = create_implicit_function(g, x_guess, {}, implicit_opts);

    auto res = implicit_fn.eval(3.0, 4.0);
    EXPECT_NEAR(res(0), 1.0, 1e-6);

    auto a_sym = sym("a_sym_custom");
    auto b_sym = sym("b_sym_custom");
    auto x_sym = implicit_fn(a_sym, b_sym)[0];
    auto ab_sym = casadi::MX::vertcat({a_sym, b_sym});
    auto jac = casadi::MX::jacobian(metis::to_mx(x_sym), ab_sym);
    casadi::Function j_fn("j_fn_custom", {a_sym, b_sym}, {jac});
    auto j_val = j_fn(std::vector<casadi::DM>{casadi::DM(3.0), casadi::DM(4.0)});
    std::vector<double> jac_vals = std::vector<double>(j_val[0]);

    ASSERT_EQ(jac_vals.size(), 2u);
    EXPECT_NEAR(jac_vals[0], -0.2, 1e-6);
    EXPECT_NEAR(jac_vals[1], 0.2, 1e-6);
}

TEST(RootFindingTest, ImplicitFunctionDifferentInputOutputIndices) {
    // The unknown lives in input slot 1 but the residual lives in output slot 0.
    // G(a, x, b) -> (x^2 + a*x - b, a - b)
    //   implicit_input_index  = 1  (x is the unknown)
    //   implicit_output_index = 0  (first output is the residual)
    auto a = sym("a");
    auto x = sym("x");
    auto b = sym("b");
    auto residual = x * x + a * x - b;
    auto aux = a - b;
    Function g("g_diff_indices", {a, x, b}, {residual, aux});

    Eigen::VectorXd x_guess(1);
    x_guess << 1.0;

    ImplicitFunctionOptions implicit_opts;
    implicit_opts.implicit_input_index = 1;
    implicit_opts.implicit_output_index = 0;
    auto implicit_fn = create_implicit_function(g, x_guess, {}, implicit_opts);

    // At (a, b) = (3, 4), the positive root is x = 1.
    auto res = implicit_fn.eval(3.0, 4.0);
    EXPECT_NEAR(res(0), 1.0, 1e-6);

    // Verify derivatives: dx/da = -x/(2x+a) = -0.2, dx/db = 1/(2x+a) = 0.2
    auto a_sym = sym("a_sym_diff");
    auto b_sym = sym("b_sym_diff");
    auto x_sym = implicit_fn(a_sym, b_sym)[0];
    auto ab_sym = casadi::MX::vertcat({a_sym, b_sym});
    auto jac = casadi::MX::jacobian(metis::to_mx(x_sym), ab_sym);
    casadi::Function j_fn("j_fn_diff_idx", {a_sym, b_sym}, {jac});
    auto j_val = j_fn(std::vector<casadi::DM>{casadi::DM(3.0), casadi::DM(4.0)});
    std::vector<double> jac_vals = std::vector<double>(j_val[0]);

    ASSERT_EQ(jac_vals.size(), 2u);
    EXPECT_NEAR(jac_vals[0], -0.2, 1e-6);
    EXPECT_NEAR(jac_vals[1], 0.2, 1e-6);
}

TEST(RootFindingTest, NewtonSolverClass) {
    auto x = sym("x");
    auto f_expr = x * x - 9.0; // Roots at +/- 3
    Function f("f", {x}, {f_expr});

    // Create persistent solver
    metis::NewtonSolver solver(f);

    // Solve with first guess
    Eigen::VectorXd x0(1);
    x0 << 1.0;
    auto res1 = solver.solve(x0);
    EXPECT_TRUE(res1.converged);
    EXPECT_EQ(res1.method, RootSolveMethod::TrustRegionNewton);
    EXPECT_NEAR(res1.x(0), 3.0, 1e-6);

    // Solve with second guess
    x0 << -1.0;
    auto res2 = solver.solve(x0);
    EXPECT_TRUE(res2.converged);
    EXPECT_EQ(res2.method, RootSolveMethod::TrustRegionNewton);
    EXPECT_NEAR(res2.x(0), -3.0, 1e-6);
}
