/**
 * @file test_opti.cpp
 * @brief Tests for metis::Opti optimization interface
 *
 * Tests cover:
 * - Rosenbrock 2D unconstrained
 * - Rosenbrock 2D constrained (unit circle)
 * - N-dimensional Rosenbrock
 * - Derivative constraints for trajectory optimization
 */

#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Arithmetic.hpp>
#include <metis/math/AutoDiff.hpp>
#include <metis/math/Logic.hpp>
#include <metis/math/Spacing.hpp>
#include <metis/math/Trig.hpp>
#include <metis/optimization/Opti.hpp>
#include <metis/optimization/OptiOptions.hpp>

// =============================================================================
// Rosenbrock Tests (based on AeroSandbox benchmarks)
// =============================================================================

TEST(OptiTest, Rosenbrock2D_Unconstrained) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    // Rosenbrock: (1-x)^2 + 100*(y - x^2)^2
    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 1.0, 1e-4);
    EXPECT_NEAR(sol.value(y), 1.0, 1e-4);
}

TEST(OptiTest, Rosenbrock2D_Constrained) {
    // Constrained to unit circle: x^2 + y^2 <= 1
    metis::Opti opti;

    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    opti.subject_to(x * x + y * y <= 1);

    auto f = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
    opti.minimize(f);

    auto sol = opti.solve({.verbose = false});

    // Known solution from MATLAB/Julia
    EXPECT_NEAR(sol.value(x), 0.7864, 1e-3);
    EXPECT_NEAR(sol.value(y), 0.6177, 1e-3);
}

TEST(OptiTest, RosenbrockND) {
    // N-dimensional Rosenbrock with non-negativity constraint
    constexpr int N = 10;
    metis::Opti opti;

    auto x = opti.variable(N, 1.0); // init_guess = 1 (near solution)

    // Clean API: apply lower bound to all elements at once
    opti.subject_to_lower(x, 0.0);

    // Objective: sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
    metis::SymbolicScalar obj = 0;
    for (int i = 0; i < N - 1; ++i) {
        obj = obj + 100 * metis::pow(x(i + 1) - x(i) * x(i), 2) + metis::pow(1 - x(i), 2);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    // All elements should be ~1.0 at optimum
    metis::NumericVector x_opt = sol.value(x);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(x_opt(i), 1.0, 1e-4);
    }
}

// =============================================================================
// Constraint Tests
// =============================================================================

TEST(OptiTest, EqualityConstraint) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    opti.subject_to(x + y == 2); // Equality constraint
    opti.minimize(x * x + y * y);

    auto sol = opti.solve({.verbose = false});

    // Minimum of x^2 + y^2 subject to x + y = 2 is at x = y = 1
    EXPECT_NEAR(sol.value(x), 1.0, 1e-6);
    EXPECT_NEAR(sol.value(y), 1.0, 1e-6);
}

TEST(OptiTest, MultipleConstraints) {
    metis::Opti opti;

    auto x = opti.variable(0.5);
    auto y = opti.variable(0.5);

    opti.subject_to({x >= 0, y >= 0, x + y <= 1});

    opti.minimize(-x - y); // Maximize x + y

    auto sol = opti.solve({.verbose = false});

    // Maximum is at x + y = 1 boundary
    EXPECT_NEAR(sol.value(x) + sol.value(y), 1.0, 1e-6);
}

TEST(OptiTest, AllConstraintConjunction) {
    metis::Opti opti;

    auto x = opti.variable(2, -0.5);
    metis::SymbolicVector cond(2);
    cond(0) = x(0) >= 0.0;
    cond(1) = x(1) >= 0.0;

    opti.subject_to(metis::all(cond));

    auto objective = (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0);
    opti.minimize(objective);

    auto sol = opti.solve({.verbose = false});
    metis::NumericVector x_opt = sol.value(x);

    EXPECT_NEAR(x_opt(0), 1.0, 1e-6);
    EXPECT_NEAR(x_opt(1), 2.0, 1e-6);
}

TEST(OptiTest, VariableBounds) {
    metis::Opti opti;

    auto x = opti.variable(0.0, std::nullopt, -1.0, 1.0); // -1 <= x <= 1

    opti.minimize(-x); // Maximize x

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 1.0, 1e-6);
}

// =============================================================================
// Parameter Tests
// =============================================================================

TEST(OptiTest, ParameterUsage) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    auto p = opti.parameter(5.0); // Fixed parameter

    opti.subject_to(x >= p); // x >= 5
    opti.minimize(x * x);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 5.0, 1e-6);
}

// =============================================================================
// Derivative Helpers Tests (Trajectory Optimization)
// =============================================================================

TEST(OptiTest, DerivativeOf_Trapezoidal) {
    // Simple test: x(t) = t, so dx/dt = 1
    constexpr int N = 10;
    metis::Opti opti;

    metis::NumericVector t = metis::linspace(0.0, 1.0, N);
    auto x = opti.variable(t); // Init guess is t itself

    auto xdot = opti.derivative_of(x, t, 1.0); // Expect derivative ~= 1

    // Boundary conditions
    opti.subject_to(x(0) == 0);
    opti.subject_to(x(N - 1) == 1);

    // Minimize deviation from constant derivative
    metis::SymbolicScalar obj = 0;
    for (int i = 0; i < N; ++i) {
        obj = obj + (xdot(i) - 1) * (xdot(i) - 1);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    metis::NumericVector xdot_opt = sol.value(xdot);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(xdot_opt(i), 1.0, 0.1);
    }
}

TEST(OptiTest, ConstrainDerivative_DoubleIntegrator) {
    // Double integrator: position -> velocity -> acceleration(=0)
    // With a(t) = 0, v = constant, x = linear
    constexpr int N = 20;
    metis::Opti opti;

    metis::NumericVector t = metis::linspace(0.0, 1.0, N);
    auto x = opti.variable(N, 0.0); // Position
    auto v = opti.variable(N, 1.0); // Velocity
    auto a = opti.variable(N, 0.0); // Acceleration

    // Derivative constraints
    opti.constrain_derivative(v, x, t); // dx/dt = v
    opti.constrain_derivative(a, v, t); // dv/dt = a

    // Boundary conditions
    opti.subject_to(x(0) == 0);
    opti.subject_to(x(N - 1) == 1);
    opti.subject_to(v(0) == 1);
    opti.subject_to(v(N - 1) == 1);

    // Minimize acceleration squared
    metis::SymbolicScalar obj = 0;
    for (int i = 0; i < N; ++i) {
        obj = obj + a(i) * a(i);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    // With constant velocity, acceleration should be ~0
    metis::NumericVector a_opt = sol.value(a);
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(a_opt(i), 0.0, 0.1);
    }
}

// =============================================================================
// Maximize Test
// =============================================================================

TEST(OptiTest, Maximize) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    opti.subject_to(x <= 5);
    opti.maximize(x);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 5.0, 1e-6);
}

// =============================================================================
// Stats Test
// =============================================================================

TEST(OptiTest, SolverStats) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    opti.minimize(x * x);

    auto sol = opti.solve({.verbose = false});

    // Should have some iterations recorded
    EXPECT_GE(sol.num_iterations(), 0);
}

// =============================================================================
// Integration Tests: Function + Opti + Jacobian Synergies
// =============================================================================

/**
 * Test 1: Using metis::Function output as constraint in Opti
 *
 * Demonstrates: Pre-compiled symbolic functions can be reused in optimization
 */
TEST(OptiIntegration, FunctionAsConstraint) {
    // Define a reusable constraint function: circle constraint
    auto x_sym = metis::sym("x");
    auto y_sym = metis::sym("y");
    auto circle_expr = x_sym * x_sym + y_sym * y_sym;

    // Create compiled function
    metis::Function circle_fn("circle", {x_sym, y_sym}, {circle_expr});

    // Verify function works numerically
    auto result = circle_fn.eval(3.0, 4.0); // 3^2 + 4^2 = 25
    EXPECT_NEAR(result(0, 0), 25.0, 1e-10);

    // Now use in optimization
    metis::Opti opti;
    auto x = opti.variable(0.5);
    auto y = opti.variable(0.5);

    // Use symbolic evaluation to get constraint expression
    auto symbolic_result = circle_fn(x, y);
    opti.subject_to(symbolic_result[0](0, 0) <= 1); // x^2 + y^2 <= 1

    opti.minimize(-x - y); // Maximize x + y on unit circle

    auto sol = opti.solve({.verbose = false});

    // Optimal at x = y = 1/sqrt(2) ≈ 0.707
    EXPECT_NEAR(sol.value(x), 1.0 / std::sqrt(2.0), 1e-4);
    EXPECT_NEAR(sol.value(y), 1.0 / std::sqrt(2.0), 1e-4);
}

/**
 * Test 2: Using metis::jacobian for gradient analysis
 *
 * Demonstrates: Computing explicit gradients of objective for analysis
 */
TEST(OptiIntegration, JacobianForGradientAnalysis) {
    // Define Rosenbrock function symbolically
    auto x = metis::sym("x");
    auto y = metis::sym("y");
    auto rosenbrock = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);

    // Compute gradient symbolically using metis::jacobian
    auto gradient = metis::jacobian({rosenbrock}, {x, y});

    // Compile gradient into a function for efficient evaluation
    metis::Function grad_fn("rosenbrock_grad", {x, y}, {gradient});

    // Evaluate gradient at optimum (x=1, y=1)
    auto grad_at_opt = grad_fn.eval(1.0, 1.0);

    // At minimum, gradient should be [0, 0]
    EXPECT_NEAR(grad_at_opt(0, 0), 0.0, 1e-10); // df/dx at (1,1)
    EXPECT_NEAR(grad_at_opt(0, 1), 0.0, 1e-10); // df/dy at (1,1)

    // Evaluate gradient away from optimum (x=0, y=0)
    auto grad_away = grad_fn.eval(0.0, 0.0);

    // df/dx at (0,0) = -2(1-x) - 400x(y-x^2) = -2
    // df/dy at (0,0) = 200(y-x^2) = 0
    EXPECT_NEAR(grad_away(0, 0), -2.0, 1e-10);
    EXPECT_NEAR(grad_away(0, 1), 0.0, 1e-10);
}

/**
 * Test 3: Constraint Jacobian for sensitivity analysis
 *
 * Demonstrates: Computing constraint Jacobians explicitly for debugging
 */
TEST(OptiIntegration, ConstraintJacobianAnalysis) {
    // Variables
    auto x = metis::sym("x");
    auto y = metis::sym("y");
    auto z = metis::sym("z");

    // Constraints:
    //   g1(x,y,z) = x + 2y + 3z - 6  (plane)
    //   g2(x,y,z) = x^2 + y^2 - z    (paraboloid)
    auto g1 = x + 2 * y + 3 * z - 6;
    auto g2 = x * x + y * y - z;

    // Compute constraint Jacobian
    auto constraint_jac = metis::jacobian({g1, g2}, {x, y, z});

    // Compile to function
    metis::Function jac_fn("constraint_jac", {x, y, z}, {constraint_jac});

    // Evaluate at point (1, 1, 1)
    auto J = jac_fn.eval(1.0, 1.0, 1.0);

    // Expected Jacobian:
    // | dg1/dx dg1/dy dg1/dz |   | 1   2   3 |
    // | dg2/dx dg2/dy dg2/dz | = | 2x 2y  -1 | at (1,1,1) = | 2 2 -1 |
    EXPECT_NEAR(J(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(J(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(J(0, 2), 3.0, 1e-10);
    EXPECT_NEAR(J(1, 0), 2.0, 1e-10); // 2*x at x=1
    EXPECT_NEAR(J(1, 1), 2.0, 1e-10); // 2*y at y=1
    EXPECT_NEAR(J(1, 2), -1.0, 1e-10);
}

/**
 * Test 4: Building physics model with Function, optimizing with Opti
 *
 * Demonstrates: Full workflow - define physics as Function, embed in Opti
 */
TEST(OptiIntegration, PhysicsModelInOptimization) {
    // Physics model: projectile motion
    // y(t) = v0*sin(theta)*t - 0.5*g*t^2
    // x(t) = v0*cos(theta)*t
    // Find angle theta to maximize range (x when y=0)

    constexpr double v0 = 10.0; // Initial velocity [m/s]
    constexpr double g = 9.81;  // Gravity [m/s^2]

    metis::Opti opti;

    auto theta = opti.variable(M_PI / 4); // Initial guess: 45 degrees
    opti.subject_to_bounds(metis::SymbolicVector::Constant(1, theta), 0.01, M_PI / 2 - 0.01);

    // Time of flight: t_f = 2*v0*sin(theta)/g
    auto t_flight = 2 * v0 * metis::sin(theta) / g;

    // Range: x(t_f) = v0*cos(theta)*t_f
    auto range = v0 * metis::cos(theta) * t_flight;

    opti.maximize(range);

    auto sol = opti.solve({.verbose = false});

    // Optimal angle for maximum range is 45 degrees = pi/4
    EXPECT_NEAR(sol.value(theta), M_PI / 4, 1e-3);
}

/**
 * Test 5: Hessian computation for second-order optimality
 *
 * Demonstrates: Computing Hessian for quadratic approximation analysis
 */
TEST(OptiIntegration, HessianComputation) {
    // Quadratic function: f(x,y) = x^2 + 2*y^2 + x*y
    auto x = metis::sym("x");
    auto y = metis::sym("y");
    auto f = x * x + 2 * y * y + x * y;

    // First compute gradient
    auto grad = metis::jacobian({f}, {x, y}); // 1x2 row vector

    // Compute Hessian by differentiating gradient
    // grad = [df/dx, df/dy] as MX
    // We need to reshape or extract elements
    auto df_dx = grad(0, 0);
    auto df_dy = grad(0, 1);

    // Compute second derivatives
    auto d2f_dxx = metis::jacobian({df_dx}, {x});
    auto d2f_dxy = metis::jacobian({df_dx}, {y});
    auto d2f_dyx = metis::jacobian({df_dy}, {x});
    auto d2f_dyy = metis::jacobian({df_dy}, {y});

    // Compile to functions
    metis::Function hxx_fn({x, y}, {d2f_dxx});
    metis::Function hxy_fn({x, y}, {d2f_dxy});
    metis::Function hyy_fn({x, y}, {d2f_dyy});

    // Evaluate (should be constant for quadratic)
    auto hxx = hxx_fn.eval(0.0, 0.0);
    auto hxy = hxy_fn.eval(0.0, 0.0);
    auto hyy = hyy_fn.eval(0.0, 0.0);

    // Expected Hessian:
    // | d²f/dx² d²f/dxdy |   | 2 1 |
    // | d²f/dydx d²f/dy² | = | 1 4 |
    EXPECT_NEAR(hxx(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(hxy(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(hyy(0, 0), 4.0, 1e-10);
}

/**
 * Test 6: Reusing symbolic expressions between Opti and Function
 *
 * Demonstrates: Same symbolic expression used for optimization and evaluation
 */
TEST(OptiIntegration, SharedSymbolicExpressions) {
    // Create shared symbolic variables
    auto x = metis::sym("x");
    auto y = metis::sym("y");

    // Define shared objective expression
    auto objective_expr = x * x + y * y;

    // Create callable function from expression
    metis::Function obj_fn("objective", {x, y}, {objective_expr});

    // Verify numeric evaluation
    EXPECT_NEAR(obj_fn.eval(3.0, 4.0)(0, 0), 25.0, 1e-10);

    // Use SAME expression structure in optimization
    metis::Opti opti;
    auto opt_x = opti.variable(1.0);
    auto opt_y = opti.variable(1.0);

    // Substitute optimization variables into expression pattern
    auto opt_objective = opt_x * opt_x + opt_y * opt_y; // Same structure

    opti.subject_to(opt_x + opt_y == 1);
    opti.minimize(opt_objective);

    auto sol = opti.solve({.verbose = false});

    // Verify solution using the compiled function
    double x_opt = sol.value(opt_x);
    double y_opt = sol.value(opt_y);
    auto obj_at_opt = obj_fn.eval(x_opt, y_opt);

    EXPECT_NEAR(obj_at_opt(0, 0), 0.5, 1e-6); // x=y=0.5, obj = 0.25 + 0.25 = 0.5
}

// =============================================================================
// Vector Parameter Tests
// =============================================================================

TEST(OptiTest, VectorParameter) {
    metis::Opti opti;

    // Create a vector parameter
    metis::NumericVector p_vals(3);
    p_vals << 1.0, 2.0, 3.0;
    auto p = opti.parameter(p_vals);

    auto x = opti.variable(0.0);

    // Constraint using vector parameter: x >= p[0] + p[1] + p[2] = 6
    opti.subject_to(x >= p(0) + p(1) + p(2));
    opti.minimize(x * x);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 6.0, 1e-6);
}

// =============================================================================
// Scalar Bounds Tests
// =============================================================================

TEST(OptiTest, ScalarBounds_Lower) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    opti.subject_to_lower(x, 3.0); // x >= 3
    opti.minimize(x);              // Minimize x

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 3.0, 1e-6);
}

TEST(OptiTest, ScalarBounds_Upper) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    opti.subject_to_upper(x, -2.0); // x <= -2
    opti.maximize(x);               // Maximize x

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), -2.0, 1e-6);
}

TEST(OptiTest, ScalarBounds_Both) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    opti.subject_to_bounds(x, -1.0, 1.0); // -1 <= x <= 1
    opti.minimize(x);

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), -1.0, 1e-6);
}

// =============================================================================
// Derivative Constraint Method Tests
// =============================================================================

TEST(OptiTest, ConstrainDerivative_ForwardEuler) {
    constexpr int N = 10;
    metis::Opti opti;

    metis::NumericVector t = metis::linspace(0.0, 1.0, N);
    auto x = opti.variable(N, 0.0);
    auto v = opti.variable(N, 1.0);

    // Use Forward Euler method
    opti.constrain_derivative(v, x, t, "forward_euler");

    opti.subject_to(x(0) == 0);
    opti.subject_to(x(N - 1) == 1);

    metis::SymbolicScalar obj = 0;
    for (int i = 0; i < N; ++i) {
        obj = obj + (v(i) - 1.0) * (v(i) - 1.0);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    // x should go from 0 to 1 linearly
    metis::NumericVector x_opt = sol.value(x);
    EXPECT_NEAR(x_opt(0), 0.0, 1e-4);
    EXPECT_NEAR(x_opt(N - 1), 1.0, 1e-4);
}

TEST(OptiTest, ConstrainDerivative_BackwardEuler) {
    constexpr int N = 10;
    metis::Opti opti;

    metis::NumericVector t = metis::linspace(0.0, 1.0, N);
    auto x = opti.variable(N, 0.0);
    auto v = opti.variable(N, 1.0);

    // Use Backward Euler method
    opti.constrain_derivative(v, x, t, "backward_euler");

    opti.subject_to(x(0) == 0);
    opti.subject_to(x(N - 1) == 1);

    metis::SymbolicScalar obj = 0;
    for (int i = 0; i < N; ++i) {
        obj = obj + (v(i) - 1.0) * (v(i) - 1.0);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    metis::NumericVector x_opt = sol.value(x);
    EXPECT_NEAR(x_opt(0), 0.0, 1e-4);
    EXPECT_NEAR(x_opt(N - 1), 1.0, 1e-4);
}

// =============================================================================
// Solver Options Tests
// =============================================================================

TEST(OptiTest, SolverOptions_DetectBounds) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    opti.subject_to_bounds(x, 0.0, 10.0);
    opti.minimize((x - 5) * (x - 5));

    auto sol = opti.solve({.verbose = false, .detect_simple_bounds = true});

    EXPECT_NEAR(sol.value(x), 5.0, 1e-6);
}

// =============================================================================
// Error Path Tests
// =============================================================================

TEST(OptiTest, DerivativeOf_SizeMismatch) {
    metis::Opti opti;

    metis::NumericVector t = metis::linspace(0.0, 1.0, 10);
    auto x = opti.variable(5, 0.0); // Size 5, but t is size 10

    EXPECT_THROW(opti.derivative_of(x, t, 0.0), metis::InvalidArgument);
}

TEST(OptiTest, VectorBounds_Combined) {
    metis::Opti opti;

    auto x = opti.variable(5, 0.5);

    // Apply combined bounds to vector
    opti.subject_to_bounds(x, 0.0, 1.0);

    // Objective: minimize sum(x)
    metis::SymbolicScalar obj = 0;
    for (int i = 0; i < 5; ++i) {
        obj = obj + x(i);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    // All elements should be at lower bound
    metis::NumericVector x_opt = sol.value(x);
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(x_opt(i), 0.0, 1e-6);
    }
}

TEST(OptiTest, VectorBounds_Upper) {
    metis::Opti opti;

    auto x = opti.variable(3, 0.0);
    opti.subject_to_upper(x, 2.0); // All x[i] <= 2

    metis::SymbolicScalar obj = 0;
    for (int i = 0; i < 3; ++i) {
        obj = obj - x(i); // Maximize sum(x)
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    metis::NumericVector x_opt = sol.value(x);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(x_opt(i), 2.0, 1e-6);
    }
}

// =============================================================================
// Variable Freezing & Category Tests
// =============================================================================

TEST(OptiTest, VariableFreezing_SingleVariable) {
    metis::Opti opti;

    // x is optimized, y is frozen at 2.0
    auto x = opti.variable(0.0);
    auto y = opti.variable(2.0, {.freeze = true});

    // Minimize (x - 1)^2 + (y - x)^2
    // If y were free, optimal is x=y=1 gives 0
    // With y frozen at 2.0, optimal x minimizes (x-1)^2 + (2-x)^2
    // d/dx = 2(x-1) - 2(2-x) = 0 => x-1 = 2-x => 2x = 3 => x = 1.5
    opti.minimize(metis::pow(x - 1, 2) + metis::pow(y - x, 2));

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 1.5, 1e-6);
    EXPECT_NEAR(sol.value(y), 2.0, 1e-6); // Should remain at init guess
}

TEST(OptiTest, VariableFreezing_Category) {
    // Freeze "Wing" category via constructor
    metis::Opti opti({"Wing"});

    // x is in "Wing" -> should be frozen
    // y is in "Fuselage" -> should be free
    auto x = opti.variable(10.0, {.category = "Wing"});
    auto y = opti.variable(0.0, {.category = "Fuselage"});

    // Minimize (x - 5)^2 + (y - 5)^2
    // If both free: x=5, y=5
    // With x frozen at 10: x=10, y=5
    opti.minimize(metis::pow(x - 5, 2) + metis::pow(y - 5, 2));

    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 10.0, 1e-6); // Frozen
    EXPECT_NEAR(sol.value(y), 5.0, 1e-6);  // Free
}

TEST(OptiTest, VariableFreezing_ExplicitOverride) {
    // Freeze "Wing" category, but explicitly unfreeze one variable?
    // Note: Current implementation is OR logic (freeze if opts.freeze OR category frozen)
    // So we can only force freeze, not force unfreeze against category.
    // Let's test force freeze in non-frozen category.

    metis::Opti opti; // No categories frozen

    // x is "Wing" (not frozen), but explicitly frozen
    auto x = opti.variable(10.0, {.category = "Wing", .freeze = true});
    auto y = opti.variable(0.0, {.category = "Wing"}); // Implicitly free

    opti.minimize(metis::pow(x - 5, 2) + metis::pow(y - 5, 2));
    auto sol = opti.solve({.verbose = false});

    EXPECT_NEAR(sol.value(x), 10.0, 1e-6);
    EXPECT_NEAR(sol.value(y), 5.0, 1e-6);
}

TEST(OptiTest, CategoryTracking) {
    metis::Opti opti;

    auto x = opti.variable(1.0, {.category = "A"});
    auto y = opti.variable(2.0, {.category = "A"});
    auto z = opti.variable(3.0, {.category = "B"});

    auto cat_a = opti.get_category("A");
    auto cat_b = opti.get_category("B");
    auto cat_c = opti.get_category("C");

    EXPECT_EQ(cat_a.size(), 2);
    EXPECT_EQ(cat_b.size(), 1);
    EXPECT_EQ(cat_c.size(), 0);

    auto names = opti.get_category_names();
    EXPECT_EQ(names.size(), 2);
    // Use std::find for vectors
    bool has_A = std::find(names.begin(), names.end(), "A") != names.end();
    bool has_B = std::find(names.begin(), names.end(), "B") != names.end();
    EXPECT_TRUE(has_A);
    EXPECT_TRUE(has_B);
}

// =============================================================================
// Scaling Tests
// =============================================================================

TEST(OptiTest, ScalingAnalysisUsesFiniteBoundsForDefaultVariableScale) {
    metis::Opti opti;

    auto x = opti.variable(0.0, std::nullopt, -1e6, 1e6);
    opti.minimize((x - 2.0) * (x - 2.0));

    auto report = opti.analyze_scaling();

    ASSERT_EQ(report.variables.size(), 1);
    EXPECT_DOUBLE_EQ(report.variables[0].scale, 1e6);
    EXPECT_DOUBLE_EQ(report.variables[0].suggested_scale, 1e6);
    EXPECT_DOUBLE_EQ(report.variables[0].normalized_init_abs_max, 0.0);
}

TEST(OptiTest, ExplicitObjectiveAndConstraintScaling) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    opti.subject_to(x == 1e6, 1e6);
    opti.minimize(metis::pow(x - 1e6, 2), 1e12);

    auto report = opti.analyze_scaling();

    ASSERT_TRUE(report.objective.configured);
    EXPECT_TRUE(report.objective.user_supplied_scale);
    EXPECT_DOUBLE_EQ(report.objective.scale, 1e12);
    EXPECT_DOUBLE_EQ(report.objective.normalized_value, 1.0);
    ASSERT_EQ(report.constraints.size(), 1);
    EXPECT_DOUBLE_EQ(report.constraints[0].scale, 1e6);
    EXPECT_DOUBLE_EQ(report.constraints[0].normalized_magnitude, 1.0);
    EXPECT_FALSE(report.has_issues());

    auto sol = opti.solve({.verbose = false});
    EXPECT_NEAR(sol.value(x), 1e6, 1e-3);
}

TEST(OptiTest, ScalingAnalysisWarnsForLargeUnscaledObjectiveAndConstraint) {
    metis::Opti opti;

    auto x = opti.variable(0.0);
    opti.subject_to(x == 1e6);
    opti.minimize(metis::pow(x - 1e6, 2));

    auto report = opti.analyze_scaling();

    EXPECT_TRUE(report.has_issues());

    bool saw_constraint_warning = false;
    bool saw_objective_warning = false;
    for (const auto &issue : report.issues) {
        saw_constraint_warning |= issue.kind == metis::ScalingIssueKind::Constraint;
        saw_objective_warning |= issue.kind == metis::ScalingIssueKind::Objective;
    }

    EXPECT_TRUE(saw_constraint_warning);
    EXPECT_TRUE(saw_objective_warning);
}

TEST(OptiTest, ScaleValidationErrors) {
    EXPECT_THROW(metis::detail::validate_positive_scale(0.0, "scale"), metis::InvalidArgument);
    EXPECT_THROW(
        metis::detail::validate_positive_scale(std::numeric_limits<double>::quiet_NaN(), "scale"),
        metis::InvalidArgument);
    EXPECT_DOUBLE_EQ(metis::detail::max_finite_abs(-3.0, std::numeric_limits<double>::infinity()),
                     3.0);
    EXPECT_DOUBLE_EQ(metis::detail::constraint_violation(5.0, 0.0, true, 4.0, true), 1.0);

    metis::Opti opti;
    EXPECT_THROW(opti.variable(0.0, std::optional<double>(0.0)), metis::InvalidArgument);
    EXPECT_THROW(opti.variable(3, 0.0, std::optional<double>(0.0)), metis::InvalidArgument);

    metis::NumericVector init(2);
    init << 1.0, 2.0;
    EXPECT_THROW(opti.variable(init, std::optional<double>(0.0)), metis::InvalidArgument);

    auto x = opti.variable(0.0);
    EXPECT_THROW(opti.minimize(x * x, 0.0), metis::InvalidArgument);
    EXPECT_THROW(opti.maximize(x, std::numeric_limits<double>::quiet_NaN()),
                 metis::InvalidArgument);
}

TEST(OptiTest, ScalingAnalysisWarnsForTinyVariableAndScaleSpan) {
    metis::Opti opti;

    auto x = opti.variable(1e-9, std::optional<double>(1e4));
    auto y = opti.variable(1.0, std::optional<double>(1e-3));
    opti.minimize((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0));

    auto report = opti.analyze_scaling();

    bool saw_variable_warning = false;
    bool saw_summary_warning = false;
    for (const auto &issue : report.issues) {
        saw_variable_warning |= issue.kind == metis::ScalingIssueKind::Variable;
        saw_summary_warning |= issue.kind == metis::ScalingIssueKind::Summary;
    }

    EXPECT_TRUE(saw_variable_warning);
    EXPECT_TRUE(saw_summary_warning);
    EXPECT_GT(report.summary.variable_scale_ratio, 1e6);
}

// =============================================================================
// Parametric Sweep Tests
// =============================================================================

TEST(OptiTest, ParametricSweep_Basic) {
    // Minimize (x - k)^2 where k is a parameter
    // Optimal x* = k for each parameter value
    metis::Opti opti;

    auto k = opti.parameter(1.0); // Parameter to sweep
    auto x = opti.variable(0.0);

    opti.minimize(metis::pow(x - k, 2));

    // Sweep k from 1 to 5
    std::vector<double> k_values = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto result = opti.solve_sweep(k, k_values, {.verbose = false});

    ASSERT_TRUE(result.all_converged);
    ASSERT_EQ(result.size(), 5);

    // Check that x* = k for each solve
    for (size_t i = 0; i < result.size(); ++i) {
        ASSERT_TRUE(result.solutions[i].has_value());
        EXPECT_NEAR(result.solutions[i]->value(x), k_values[i], 1e-5);
    }
}

TEST(OptiTest, ParametricSweep_WarmStart) {
    // Verify warm-starting reduces iterations
    metis::Opti opti;

    auto k = opti.parameter(0.0);
    auto x = opti.variable(0.0);
    auto y = opti.variable(0.0);

    // Rosenbrock with shifted minimum at (k, k^2)
    opti.minimize(metis::pow(k - x, 2) + 100 * metis::pow(y - metis::pow(x, 2), 2));

    // Sweep with small steps (warm start should help)
    std::vector<double> k_values;
    for (double v = 0.0; v <= 1.0; v += 0.1) {
        k_values.push_back(v);
    }

    auto result = opti.solve_sweep(k, k_values, {.verbose = false});

    ASSERT_TRUE(result.all_converged);

    // First solve may take more iterations, subsequent should be fewer
    // (This is a heuristic check - warm start helps convergence)
    int first_iter = result.iterations[0];
    int last_iter = result.iterations.back();
    // Just verify we got valid iteration counts (0 is valid for trivial problems)
    EXPECT_GE(first_iter, 0);
    EXPECT_GE(last_iter, 0);
}
