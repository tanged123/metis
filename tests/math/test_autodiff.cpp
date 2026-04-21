#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Arithmetic.hpp>
#include <metis/math/AutoDiff.hpp>
#include <metis/math/Linalg.hpp>
#include <metis/math/Trig.hpp>

// Test Jacobian of a simple quadratic function
TEST(AutoDiffTests, JacobianQuadratic) {
    // f(x) = x^2
    // df/dx = 2*x
    auto x = metis::sym("x");
    auto f = metis::pow(x, 2.0);

    auto J_sym = metis::jacobian(f, x);

    // Create function to evaluate Jacobian
    metis::Function J_fun({x}, {J_sym});

    // Evaluate at x = 3
    auto J = J_fun.eval(3.0);

    // Should be 2*3 = 6
    EXPECT_NEAR(J(0, 0), 6.0, 1e-10);
}

// Test Jacobian with multiple inputs
TEST(AutoDiffTests, JacobianMultipleInputs) {
    // f(x, y) = x^2 + 2*x*y + y^2
    // df/dx = 2*x + 2*y
    // df/dy = 2*x + 2*y
    auto x = metis::sym("x");
    auto y = metis::sym("y");

    auto f = metis::pow(x, 2.0) + 2.0 * x * y + metis::pow(y, 2.0);

    // Jacobian: [df/dx, df/dy]
    auto J_sym = metis::jacobian(f, x, y);

    metis::Function J_fun({x, y}, {J_sym});

    // Evaluate at x=1, y=2
    auto J = J_fun.eval(1.0, 2.0);

    // df/dx = 2*1 + 2*2 = 6
    // df/dy = 2*1 + 2*2 = 6
    EXPECT_NEAR(J(0, 0), 6.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 6.0, 1e-10);
}

// Test Jacobian from drag coefficient example
TEST(AutoDiffTests, JacobianDragCoefficient) {
    // Drag model: Drag = 0.5 * rho * v^2 * S * (Cd0 + k * (Cl - Cl0)^2)
    auto v = metis::sym("v");
    auto Cl = metis::sym("Cl");

    // Constants
    double rho = 1.225;
    double S = 10.0;
    double Cd0 = 0.02;
    double k = 0.04;
    double Cl0 = 0.1;

    metis::SymbolicScalar rho_s = rho;
    metis::SymbolicScalar S_s = S;
    metis::SymbolicScalar Cd0_s = Cd0;
    metis::SymbolicScalar k_s = k;
    metis::SymbolicScalar Cl0_s = Cl0;

    auto q = 0.5 * rho_s * metis::pow(v, 2.0);
    auto Cd = Cd0_s + k_s * metis::pow(Cl - Cl0_s, 2.0);
    auto drag = q * S_s * Cd;

    // Compute Jacobian: [dDrag/dv, dDrag/dCl]
    auto J_sym = metis::jacobian(drag, v, Cl);

    metis::Function J_fun({v, Cl}, {J_sym});

    // Test at v=50, Cl=0.5
    double v_val = 50.0;
    double Cl_val = 0.5;
    auto J = J_fun.eval(v_val, Cl_val);

    // Analytic derivatives:
    // Cd = 0.02 + 0.04 * (0.5 - 0.1)^2 = 0.02 + 0.04 * 0.16 = 0.0264
    // q = 0.5 * 1.225 * 50^2 = 1531.25
    // dDrag/dv = rho * v * S * Cd = 1.225 * 50 * 10 * 0.0264 = 16.17
    // dDrag/dCl = q * S * 2 * k * (Cl - Cl0) = 1531.25 * 10 * 2 * 0.04 * 0.4 = 490

    EXPECT_NEAR(J(0, 0), 16.17, 0.01);
    EXPECT_NEAR(J(0, 1), 490.0, 1.0);
}

// Test Jacobian with vector of expressions and variables
TEST(AutoDiffTests, JacobianVectorInputs) {
    // f1(x, y) = x + y
    // f2(x, y) = x * y
    // J = [[df1/dx, df1/dy],
    //      [df2/dx, df2/dy]]
    //   = [[1, 1],
    //      [y, x]]
    auto x = metis::sym("x");
    auto y = metis::sym("y");

    auto f1 = x + y;
    auto f2 = x * y;

    auto J_sym = metis::jacobian({f1, f2}, {x, y});

    metis::Function J_fun({x, y}, {J_sym});

    // Evaluate at x=3, y=4
    auto J = J_fun.eval(3.0, 4.0);

    // Expected: [[1, 1], [4, 3]]
    EXPECT_NEAR(J(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 1.0, 1e-10);
    EXPECT_NEAR(J(1, 0), 4.0, 1e-10);
    EXPECT_NEAR(J(1, 1), 3.0, 1e-10);
}

// Test Jacobian of trigonometric functions
TEST(AutoDiffTests, JacobianTrigonometric) {
    // f(x) = sin(x)
    // df/dx = cos(x)
    auto x = metis::sym("x");
    auto f = metis::sin(x);

    auto J = metis::jacobian(f, x);

    metis::Function J_fun({x}, {J});

    // Evaluate at x=0
    auto J0 = J_fun.eval(0.0);

    // cos(0) = 1
    EXPECT_NEAR(J0(0, 0), 1.0, 1e-10);

    // Evaluate at x=pi/2
    auto J_pi2 = J_fun.eval(M_PI / 2.0);

    // cos(pi/2) ≈ 0
    EXPECT_NEAR(J_pi2(0, 0), 0.0, 1e-10);
}

// Test Jacobian with matrix output
TEST(AutoDiffTests, JacobianMatrixSymbolicArg) {
    // Test using SymbolicArg for matrix inputs
    using SymbolicMatrix = metis::SymbolicMatrix;

    auto x = metis::sym("x");
    auto y = metis::sym("y");

    // Create a 2x1 symbolic matrix
    SymbolicMatrix vec(2, 1);
    vec(0, 0) = x;
    vec(1, 0) = y;

    // f(vec) = sum of squares = x^2 + y^2
    auto f = metis::pow(x, 2.0) + metis::pow(y, 2.0);

    // Jacobian using SymbolicArg wrapper
    auto J_sym = metis::jacobian({f}, {x, y});

    metis::Function J_fun({x, y}, {J_sym});

    // Evaluate at x=1, y=2
    auto J = J_fun.eval(1.0, 2.0);

    // Expected: [2*1, 2*2] = [2, 4]
    EXPECT_NEAR(J(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 4.0, 1e-10);
}

// ============================================================================
// Higher-Order Derivatives Tests (Phase 7)
// ============================================================================

TEST(AutoDiffTests, SymbolicGradient) {
    // f(x, y) = x^2 * y
    // grad(f) = [2xy, x^2] (column vector)
    auto x = metis::sym("x");
    auto y = metis::sym("y");
    auto f = metis::pow(x, 2.0) * y;

    // Compute symbolic gradient
    auto g_sym = metis::sym_gradient(f, {x, y});

    // Verify it's a column vector
    EXPECT_EQ(g_sym.cols(), 1);
    EXPECT_EQ(g_sym.rows(), 2);

    // Evaluate
    metis::Function g_fun({x, y}, {g_sym});
    auto g_val = g_fun.eval(3.0, 2.0); // x=3, y=2

    // grad = [2*3*2, 3^2] = [12, 9]
    EXPECT_NEAR(g_val(0), 12.0, 1e-10);
    EXPECT_NEAR(g_val(1), 9.0, 1e-10);
}

TEST(AutoDiffTests, HessianMatrix) {
    // f(x, y) = x^2 + 3xy + y^3
    // grad(f) = [2x + 3y, 3x + 3y^2]
    // H(f) = [[2, 3],
    //         [3, 6y]]
    auto x = metis::sym("x");
    auto y = metis::sym("y");
    auto f = metis::pow(x, 2.0) + 3.0 * x * y + metis::pow(y, 3.0);

    auto H_sym = metis::hessian(f, {x, y});

    // Verify dimensions
    EXPECT_EQ(H_sym.rows(), 2);
    EXPECT_EQ(H_sym.cols(), 2);

    // Evaluate
    metis::Function H_fun({x, y}, {H_sym});
    auto H_val = H_fun.eval(1.0, 2.0); // y=2

    // H = [[2, 3], [3, 6*2]] = [[2, 3], [3, 12]]
    EXPECT_NEAR(H_val(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(H_val(0, 1), 3.0, 1e-10);
    EXPECT_NEAR(H_val(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(H_val(1, 1), 12.0, 1e-10);
}

TEST(AutoDiffTests, HessianLagrangian) {
    // Minimize f(x) = x^2 subject to g(x) = x - 1 = 0
    // L = x^2 + lambda * (x - 1)
    // H_L = d^2L/dx^2 = 2

    auto x = metis::sym("x");
    auto lam = metis::sym("lam");

    auto obj = metis::pow(x, 2.0);
    auto constr = x - 1.0;

    auto HL_sym = metis::hessian_lagrangian(obj, constr, x, lam);

    metis::Function HL_fun({x, lam}, {HL_sym});
    auto HL_val = HL_fun.eval(0.0, 0.0);

    EXPECT_NEAR(HL_val(0, 0), 2.0, 1e-10);

    // Case 2: Multi-dimensional
    // f = x^2 + y^2, g = x+y
    // L = x^2 + y^2 + lam*(x+y)
    // H_L = diag(2, 2)
    auto y = metis::sym("y");
    auto obj2 = metis::pow(x, 2.0) + metis::pow(y, 2.0);
    auto constr2 = x + y;

    auto HL2_sym = metis::hessian_lagrangian(obj2, constr2, {x, y}, lam);
    metis::Function HL2_fun({x, y, lam}, {HL2_sym});
    auto HL2_val = HL2_fun.eval(0.0, 0.0, 0.0);

    EXPECT_NEAR(HL2_val(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(HL2_val(1, 1), 2.0, 1e-10);
    EXPECT_NEAR(HL2_val(0, 1), 0.0, 1e-10);
}

TEST(AutoDiffTests, SensitivityRegimeSelectsForwardMode) {
    auto rec = metis::select_sensitivity_regime(4, 12);

    EXPECT_EQ(rec.regime, metis::SensitivityRegime::Forward);
    EXPECT_TRUE(rec.uses_forward_mode());
    EXPECT_FALSE(rec.uses_checkpointing());
    EXPECT_EQ(rec.casadi_direction_count(), 4);

    casadi::Dict opts = rec.integrator_options();
    EXPECT_EQ(int(opts.at("nfwd")), 4);
    EXPECT_TRUE(bool(opts.at("fsens_err_con")));
    EXPECT_EQ(opts.count("nadj"), 0u);
}

TEST(AutoDiffTests, SensitivityRegimeSelectsAdjointMode) {
    auto rec = metis::select_sensitivity_regime(150, 2);

    EXPECT_EQ(rec.regime, metis::SensitivityRegime::Adjoint);
    EXPECT_TRUE(rec.uses_reverse_mode());
    EXPECT_FALSE(rec.uses_checkpointing());
    EXPECT_EQ(rec.casadi_direction_count(), 2);

    casadi::Dict opts = rec.integrator_options();
    EXPECT_EQ(int(opts.at("nadj")), 2);
    EXPECT_EQ(opts.count("nfwd"), 0u);
    EXPECT_EQ(opts.count("steps_per_checkpoint"), 0u);
}

TEST(AutoDiffTests, SensitivityRegimeSelectsCheckpointedAdjoint) {
    auto rec = metis::select_sensitivity_regime(120, 2, 400, true);

    EXPECT_EQ(rec.regime, metis::SensitivityRegime::CheckpointedAdjoint);
    EXPECT_TRUE(rec.uses_checkpointing());
    EXPECT_EQ(rec.checkpoint_interpolation, metis::CheckpointInterpolation::Polynomial);
    EXPECT_EQ(rec.steps_per_checkpoint, 20);

    casadi::Dict opts = rec.integrator_options();
    EXPECT_EQ(int(opts.at("nadj")), 2);
    EXPECT_EQ(int(opts.at("steps_per_checkpoint")), 20);
    EXPECT_EQ(std::string(opts.at("interpolation_type")), "polynomial");
}

TEST(AutoDiffTests, SensitivityJacobianBuildsForwardJacobian) {
    auto x = metis::sym("x", 2);
    casadi::MX x0 = x(0);
    casadi::MX x1 = x(1);
    auto y = casadi::MX::vertcat({x0 + 2.0 * x1, x0 * x1, metis::sin(x0) + 0.5 * x1});

    metis::Function f("forward_regime_model", {x}, {y});
    auto rec = metis::select_sensitivity_regime(f);
    ASSERT_EQ(rec.regime, metis::SensitivityRegime::Forward);

    metis::Function J_fun = metis::sensitivity_jacobian(f);

    metis::NumericVector x_val(2);
    x_val << 2.0, 3.0;

    auto J = J_fun.eval(x_val);

    ASSERT_EQ(J.rows(), 3);
    ASSERT_EQ(J.cols(), 2);
    EXPECT_NEAR(J(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(J(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(J(1, 1), 2.0, 1e-10);
    EXPECT_NEAR(J(2, 0), std::cos(2.0), 1e-10);
    EXPECT_NEAR(J(2, 1), 0.5, 1e-10);
}

TEST(AutoDiffTests, SensitivityJacobianBuildsAdjointJacobianForSelectedBlock) {
    auto x = metis::sym("x", 2);
    auto p = metis::sym("p", 4);
    casadi::MX x0 = x(0);
    casadi::MX x1 = x(1);
    casadi::MX p0 = p(0);
    casadi::MX p1 = p(1);
    casadi::MX p2 = p(2);
    casadi::MX p3 = p(3);

    auto y0 = x0 + x1;
    auto y1 = p0 * p0 + 2.0 * p1 + 3.0 * p2 * p3;

    metis::Function f("adjoint_regime_model", {x, p}, {y0, y1});
    auto rec = metis::select_sensitivity_regime(f, 1, 1);
    ASSERT_EQ(rec.regime, metis::SensitivityRegime::Adjoint);

    metis::Function J_fun = metis::sensitivity_jacobian(f, 1, 1);

    metis::NumericVector x_val(2);
    x_val << 0.25, -0.5;
    metis::NumericVector p_val(4);
    p_val << 2.0, -1.0, 3.0, 4.0;

    auto J = J_fun.eval(x_val, p_val);

    ASSERT_EQ(J.rows(), 1);
    ASSERT_EQ(J.cols(), 4);
    EXPECT_NEAR(J(0, 0), 4.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(J(0, 2), 12.0, 1e-10);
    EXPECT_NEAR(J(0, 3), 9.0, 1e-10);
}

TEST(AutoDiffTests, HessianVectorProductMatchesDenseHessian) {
    auto x = metis::sym("x", 2);
    auto v = metis::sym("v", 2);
    casadi::MX x0 = x(0);
    casadi::MX x1 = x(1);

    auto f = x0 * x0 * x1 + metis::sin(x1);

    metis::Function hvp_fun("expr_hvp", {x, v}, {metis::hessian_vector_product(f, x, v)});
    metis::Function H_fun("expr_hess", {x}, {metis::hessian(f, x)});

    metis::NumericVector x_val(2);
    x_val << 1.5, -0.3;
    metis::NumericVector v_val(2);
    v_val << 0.2, -0.7;

    auto hvp = hvp_fun.eval(x_val, v_val);
    metis::NumericMatrix H = H_fun.eval(x_val);
    metis::NumericVector expected = H * v_val;

    ASSERT_EQ(hvp.rows(), 2);
    ASSERT_EQ(hvp.cols(), 1);
    EXPECT_NEAR(hvp(0, 0), expected(0), 1e-10);
    EXPECT_NEAR(hvp(1, 0), expected(1), 1e-10);
}

TEST(AutoDiffTests, FunctionHessianVectorProductBuildsScalarBlockAction) {
    auto x = metis::sym("x");
    auto p = metis::sym("p", 3);
    casadi::MX p0 = p(0);
    casadi::MX p1 = p(1);
    casadi::MX p2 = p(2);

    auto y0 = casadi::MX::vertcat({x + p0, p1 - p2});
    auto y1 = p0 * p0 * p1 + metis::sin(p2) + x * p1;

    metis::Function f("function_hvp_model", {x, p}, {y0, y1});
    metis::Function hvp_fun = metis::hessian_vector_product(f, 1, 1);
    metis::Function H_fun("function_hvp_ref", {x, p}, {metis::hessian(y1, p)});

    const double x_val = 0.5;
    metis::NumericVector p_val(3);
    p_val << 2.0, -1.0, 0.3;
    metis::NumericVector v_val(3);
    v_val << 0.1, -0.2, 0.4;

    auto hvp = hvp_fun.eval(x_val, p_val, v_val);
    metis::NumericMatrix H = H_fun.eval(x_val, p_val);
    metis::NumericVector expected = H * v_val;

    ASSERT_EQ(hvp.rows(), 3);
    ASSERT_EQ(hvp.cols(), 1);
    EXPECT_NEAR(hvp(0, 0), expected(0), 1e-10);
    EXPECT_NEAR(hvp(1, 0), expected(1), 1e-10);
    EXPECT_NEAR(hvp(2, 0), expected(2), 1e-10);
}

TEST(AutoDiffTests, LagrangianHessianVectorProductMatchesDenseHessian) {
    auto x = metis::sym("x", 2);
    auto lam = metis::sym("lam", 2);
    auto v = metis::sym("v", 2);
    casadi::MX x0 = x(0);
    casadi::MX x1 = x(1);

    auto objective = x0 * x0 + x0 * x1 + metis::sin(x1);
    auto constraints = casadi::MX::vertcat({x0 * x1, x0 + x1 * x1});

    metis::Function hvp_fun(
        "lagrangian_hvp_expr", {x, lam, v},
        {metis::lagrangian_hessian_vector_product(objective, constraints, x, lam, v)});
    metis::Function H_fun("lagrangian_hess_expr", {x, lam},
                          {metis::hessian_lagrangian(objective, constraints, x, lam)});

    metis::NumericVector x_val(2);
    x_val << 0.7, -1.2;
    metis::NumericVector lam_val(2);
    lam_val << 1.5, -0.4;
    metis::NumericVector v_val(2);
    v_val << -0.2, 0.6;

    auto hvp = hvp_fun.eval(x_val, lam_val, v_val);
    metis::NumericMatrix H = H_fun.eval(x_val, lam_val);
    metis::NumericVector expected = H * v_val;

    ASSERT_EQ(hvp.rows(), 2);
    ASSERT_EQ(hvp.cols(), 1);
    EXPECT_NEAR(hvp(0, 0), expected(0), 1e-10);
    EXPECT_NEAR(hvp(1, 0), expected(1), 1e-10);
}

TEST(AutoDiffTests, FunctionLagrangianHessianVectorProductBuildsSecondOrderAdjointAction) {
    auto x = metis::sym("x", 2);
    auto p = metis::sym("p", 2);
    casadi::MX x0 = x(0);
    casadi::MX x1 = x(1);
    casadi::MX p0 = p(0);
    casadi::MX p1 = p(1);

    auto measurement = x0 + 2.0 * p0;
    auto objective = metis::pow(x0 - p0, 2.0) + x1 * p1 + metis::sin(x0 * x1);
    auto constraints = casadi::MX::vertcat({x0 + p0 * x1, x0 * x0 + x1 - p1});
    auto lam = metis::sym("lam", 2);

    metis::Function f("lagrangian_hvp_model", {x, p}, {measurement, objective, constraints});
    metis::Function hvp_fun = metis::lagrangian_hessian_vector_product(f, 1, 2, 0);
    metis::Function H_fun("lagrangian_hvp_ref", {x, p, lam},
                          {metis::hessian_lagrangian(objective, constraints, x, lam)});

    metis::NumericVector x_val(2);
    x_val << 0.4, -0.7;
    metis::NumericVector p_val(2);
    p_val << 1.2, -0.8;
    metis::NumericVector lam_val(2);
    lam_val << 0.5, -1.1;
    metis::NumericVector v_val(2);
    v_val << 0.3, -0.2;

    auto hvp = hvp_fun.eval(x_val, p_val, lam_val, v_val);
    metis::NumericMatrix H = H_fun.eval(x_val, p_val, lam_val);
    metis::NumericVector expected = H * v_val;

    ASSERT_EQ(hvp.rows(), 2);
    ASSERT_EQ(hvp.cols(), 1);
    EXPECT_NEAR(hvp(0, 0), expected(0), 1e-10);
    EXPECT_NEAR(hvp(1, 0), expected(1), 1e-10);
}

TEST(AutoDiffTests, HessianVectorProductValidationErrors) {
    auto x = metis::sym("x", 2);
    auto v_bad = metis::sym("v_bad", 3);
    casadi::MX x0 = x(0);
    casadi::MX x1 = x(1);
    auto f = x0 * x0 + x0 * x1;

    EXPECT_THROW(metis::hessian_vector_product(f, x, v_bad), metis::InvalidArgument);

    auto vec_out = casadi::MX::vertcat({x0, x1});
    metis::Function block_fn("nonscalar_output", {x}, {vec_out});
    EXPECT_THROW(metis::hessian_vector_product(block_fn), metis::InvalidArgument);

    auto lam_bad = metis::sym("lam_bad", 3);
    auto constraints = casadi::MX::vertcat({x0 + x1, x0 - x1});
    auto v = metis::sym("v", 2);
    EXPECT_THROW(metis::lagrangian_hessian_vector_product(f, constraints, x, lam_bad, v),
                 metis::InvalidArgument);
}
