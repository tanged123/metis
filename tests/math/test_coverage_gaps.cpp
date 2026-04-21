#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <metis/core/Function.hpp>
#include <metis/core/MetisError.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/FiniteDifference.hpp>
#include <metis/math/Integrate.hpp>
#include <metis/math/Linalg.hpp>
#include <metis/math/Logic.hpp>
#include <metis/math/OrthogonalPolynomials.hpp>
#include <metis/math/PolynomialChaos.hpp>
#include <metis/math/Quadrature.hpp>
#include <metis/math/RootFinding.hpp>
#include <metis/math/Spacing.hpp>

namespace {

std::pair<casadi::Function, casadi::Function>
make_identity_root_functions(const std::string &name) {
    casadi::MX x = casadi::MX::sym(name + "_x");
    casadi::MX residual = x;

    return {
        casadi::Function(name + "_residual", {x}, {residual}),
        casadi::Function(name + "_jacobian", {x}, {casadi::MX::jacobian(residual, x)}),
    };
}

metis::detail::NumericState make_scalar_state(double x_value, double residual_value,
                                              double jacobian_value) {
    metis::detail::NumericState state;
    state.x = Eigen::VectorXd::Constant(1, x_value);
    state.residual = Eigen::VectorXd::Constant(1, residual_value);
    state.jacobian = Eigen::MatrixXd::Constant(1, 1, jacobian_value);
    state.residual_norm = std::abs(residual_value);
    state.merit = 0.5 * residual_value * residual_value;
    return state;
}

} // namespace

// ======================================================================
// FiniteDifference.hpp Coverage
// ======================================================================

TEST(FiniteDifferenceCoverage, ErrorChecks) {
    metis::MetisVector<double> x(3);
    x << 0, 1, 2;

    // Invalid degree
    EXPECT_THROW(metis::finite_difference_coefficients(x, 0.0, -1), metis::InvalidArgument);

    // Too few points
    // Degree 3 requires 4 points
    EXPECT_THROW(metis::finite_difference_coefficients(x, 0.0, 3), metis::InvalidArgument);
}

// ======================================================================
// Integrate.hpp Coverage
// ======================================================================

TEST(IntegrateCoverage, SymbolicLambdaError) {
    auto x = metis::sym("x");
    // Use quad(func, a, b) signature with Symbolic arguments to trigger the template runtime error
    metis::SymbolicScalar a(0.0), b(1.0);
    EXPECT_THROW(metis::quad([](metis::SymbolicScalar s) { return s; }, a, b),
                 metis::IntegrationError);
}

// ======================================================================
// Spacing.hpp Coverage
// ======================================================================

TEST(SpacingCoverage, InvalidN) {
    // n < 1 should throw for all
    EXPECT_THROW(metis::linspace(0.0, 1.0, 0), metis::InvalidArgument);
    EXPECT_THROW(metis::cosine_spacing(0.0, 1.0, 0), metis::InvalidArgument);
    EXPECT_THROW(metis::sinspace(0.0, 1.0, 0), metis::InvalidArgument);
    EXPECT_THROW(metis::logspace(0.0, 1.0, 0), metis::InvalidArgument);
    EXPECT_THROW(metis::geomspace(1.0, 10.0, 0), metis::InvalidArgument);
}

// ======================================================================
// Linalg.hpp Coverage
// ======================================================================

TEST(LinalgCoverage, CrossError) {
    metis::MetisVector<double> a(2);
    a << 1, 2;
    metis::MetisVector<double> b(3);
    b << 1, 2, 3;

    EXPECT_THROW(metis::cross(a, b), metis::InvalidArgument);
}

TEST(RootFindingCoverage, HelperFunctionsAndValidationErrors) {
    EXPECT_EQ(metis::detail::method_name(metis::RootSolveMethod::None), "none");
    EXPECT_EQ(metis::detail::method_name(metis::RootSolveMethod::TrustRegionNewton),
              "trust-region Newton");
    EXPECT_EQ(metis::detail::method_name(metis::RootSolveMethod::LineSearchNewton),
              "line-search Newton");
    EXPECT_EQ(metis::detail::method_name(metis::RootSolveMethod::QuasiNewtonBroyden),
              "quasi-Newton Broyden");
    EXPECT_EQ(metis::detail::method_name(metis::RootSolveMethod::PseudoTransientContinuation),
              "pseudo-transient continuation");
    EXPECT_THROW(metis::detail::method_name(static_cast<metis::RootSolveMethod>(-1)),
                 metis::InvalidArgument);

    EXPECT_EQ(metis::detail::strategy_to_method(metis::RootSolveStrategy::Auto),
              metis::RootSolveMethod::None);
    EXPECT_EQ(metis::detail::strategy_to_method(metis::RootSolveStrategy::TrustRegionNewton),
              metis::RootSolveMethod::TrustRegionNewton);
    EXPECT_EQ(metis::detail::strategy_to_method(metis::RootSolveStrategy::LineSearchNewton),
              metis::RootSolveMethod::LineSearchNewton);
    EXPECT_EQ(metis::detail::strategy_to_method(metis::RootSolveStrategy::QuasiNewtonBroyden),
              metis::RootSolveMethod::QuasiNewtonBroyden);
    EXPECT_EQ(
        metis::detail::strategy_to_method(metis::RootSolveStrategy::PseudoTransientContinuation),
        metis::RootSolveMethod::PseudoTransientContinuation);
    EXPECT_THROW(metis::detail::strategy_to_method(static_cast<metis::RootSolveStrategy>(-1)),
                 metis::InvalidArgument);

    const std::string name0 = metis::detail::unique_name("rf");
    const std::string name1 = metis::detail::unique_name("rf");
    EXPECT_NE(name0, name1);
    EXPECT_EQ(name0.rfind("rf_", 0), 0u);

    metis::RootFinderOptions opts;
    opts.verbose = true;
    opts.linear_solver_options["pivot"] = 1;
    const casadi::Dict dict = metis::detail::opts_to_dict(opts);
    EXPECT_NE(dict.find("linear_solver_options"), dict.end());
    EXPECT_NE(dict.find("verbose"), dict.end());
    EXPECT_NE(dict.find("print_in"), dict.end());
    EXPECT_NE(dict.find("print_out"), dict.end());

    Eigen::VectorXd x(2);
    x << 1.0, -2.0;
    const casadi::DM x_dm = metis::detail::vector_to_dm(x);
    EXPECT_TRUE(metis::detail::dm_to_vector(x_dm).isApprox(x, 1e-12));

    casadi::DM M(2, 2);
    M(0, 0) = 1.0;
    M(0, 1) = 2.0;
    M(1, 0) = 3.0;
    M(1, 1) = 4.0;
    metis::NumericMatrix M_expected(2, 2);
    M_expected << 1.0, 2.0, 3.0, 4.0;
    EXPECT_TRUE(metis::detail::dm_to_matrix(M).isApprox(M_expected, 1e-12));

    testing::internal::CaptureStdout();
    metis::detail::maybe_log(opts, "coverage");
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("coverage"), std::string::npos);

    metis::RootFinderOptions bad_opts;
    bad_opts.abstol = 0.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.abstolStep = 0.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.max_iter = 0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.trust_region_initial_damping = 0.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.trust_region_damping_increase = 1.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.trust_region_damping_decrease = 1.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.line_search_contraction = 1.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.line_search_sufficient_decrease = 1.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.max_backtracking_steps = 0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.broyden_jacobian_refresh = -1;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.pseudo_transient_dt0 = 0.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.pseudo_transient_dt_growth = 1.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    bad_opts = metis::RootFinderOptions();
    bad_opts.pseudo_transient_dt0 = 2.0;
    bad_opts.pseudo_transient_dt_max = 1.0;
    EXPECT_THROW(metis::detail::validate_root_options(bad_opts, "ctx"), metis::InvalidArgument);

    EXPECT_TRUE(metis::detail::solve_linear_system(M_expected, x)
                    .isApprox(M_expected.colPivHouseholderQr().solve(x), 1e-12));
}

TEST(RootFindingCoverage, ProblemShapeValidationAndBranchErrors) {
    casadi::MX x = casadi::MX::sym("x");
    casadi::MX y = casadi::MX::sym("y");
    casadi::MX x_row = casadi::MX::sym("x_row", 1, 2);
    casadi::MX x_col = casadi::MX::sym("x_col", 2, 1);

    const casadi::Function bad_arity("bad_arity", {x, y}, {x});
    EXPECT_THROW(metis::detail::validate_root_problem(bad_arity, "ctx"), metis::InvalidArgument);

    const casadi::Function bad_input_shape("bad_input_shape", {x_row}, {casadi::MX::zeros(2, 1)});
    EXPECT_THROW(metis::detail::validate_root_problem(bad_input_shape, "ctx"),
                 metis::InvalidArgument);

    const casadi::Function bad_output_shape("bad_output_shape", {x_col}, {casadi::MX::zeros(1, 2)});
    EXPECT_THROW(metis::detail::validate_root_problem(bad_output_shape, "ctx"),
                 metis::InvalidArgument);

    const casadi::Function bad_dimensions("bad_dimensions", {x_col}, {casadi::MX::zeros(1, 1)});
    EXPECT_THROW(metis::detail::validate_root_problem(bad_dimensions, "ctx"),
                 metis::InvalidArgument);

    EXPECT_EQ(metis::detail::implicit_function_name(bad_dimensions), "bad_dimensions_implicit");

    metis::ImplicitFunctionOptions implicit_opts;
    Eigen::VectorXd x_guess = Eigen::VectorXd::Zero(2);

    implicit_opts.implicit_input_index = 2;
    EXPECT_THROW(metis::detail::validate_implicit_problem(bad_dimensions, x_guess, implicit_opts),
                 metis::InvalidArgument);

    implicit_opts = metis::ImplicitFunctionOptions();
    implicit_opts.implicit_output_index = 1;
    EXPECT_THROW(metis::detail::validate_implicit_problem(bad_dimensions, x_guess, implicit_opts),
                 metis::InvalidArgument);

    const casadi::Function bad_impl_input("bad_impl_input", {x_row, y}, {casadi::MX::zeros(2, 1)});
    EXPECT_THROW(metis::detail::validate_implicit_problem(bad_impl_input, x_guess,
                                                          metis::ImplicitFunctionOptions()),
                 metis::InvalidArgument);

    const casadi::Function bad_impl_output("bad_impl_output", {x_col, y},
                                           {casadi::MX::zeros(1, 2)});
    EXPECT_THROW(metis::detail::validate_implicit_problem(bad_impl_output, x_guess,
                                                          metis::ImplicitFunctionOptions()),
                 metis::InvalidArgument);

    EXPECT_THROW(metis::detail::validate_implicit_problem(bad_dimensions, x_guess,
                                                          metis::ImplicitFunctionOptions()),
                 metis::InvalidArgument);

    Eigen::VectorXd short_guess = Eigen::VectorXd::Zero(1);
    const casadi::Function good_impl("good_impl", {x_col, y}, {x_col});
    EXPECT_THROW(metis::detail::validate_implicit_problem(good_impl, short_guess,
                                                          metis::ImplicitFunctionOptions()),
                 metis::InvalidArgument);

    const casadi::Function bad_residual_eval("bad_residual_eval", {x}, {x, x});
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 1.0);
    EXPECT_THROW(metis::detail::evaluate_residual_only(bad_residual_eval, x0), metis::MetisError);

    const casadi::Function good_residual_eval("good_residual_eval", {x}, {x});
    const casadi::Function bad_jacobian_eval("bad_jacobian_eval", {x}, {x, x});
    EXPECT_THROW(
        metis::detail::evaluate_state(good_residual_eval, bad_jacobian_eval, x0, "rootfinder"),
        metis::MetisError);

    const casadi::Function nan_jacobian_eval(
        "nan_jacobian_eval", {x}, {casadi::MX(std::numeric_limits<double>::quiet_NaN())});
    EXPECT_THROW(
        metis::detail::evaluate_state(good_residual_eval, nan_jacobian_eval, x0, "rootfinder"),
        metis::MetisError);

    auto root_fns = make_identity_root_functions("root_stage");
    metis::RootFinderOptions opts;
    opts.abstol = 1e-12;
    opts.abstolStep = 1e-12;

    const auto converged_state = metis::detail::evaluate_state(
        root_fns.first, root_fns.second, Eigen::VectorXd::Zero(1), "root_stage");
    const auto tr_converged = metis::detail::solve_trust_region(root_fns.first, root_fns.second,
                                                                converged_state, opts, 5);
    EXPECT_TRUE(tr_converged.converged);
    EXPECT_EQ(tr_converged.message, "initial iterate satisfies tolerance");

    const auto nonconverged_state = metis::detail::evaluate_state(
        root_fns.first, root_fns.second, Eigen::VectorXd::Ones(1), "root_stage");
    const auto ls_none = metis::detail::solve_line_search(root_fns.first, root_fns.second,
                                                          nonconverged_state, opts, 0);
    EXPECT_FALSE(ls_none.converged);
    EXPECT_EQ(ls_none.message, "no line-search iterations remaining");

    const auto br_none =
        metis::detail::solve_broyden(root_fns.first, root_fns.second, nonconverged_state, opts, 0);
    EXPECT_FALSE(br_none.converged);
    EXPECT_EQ(br_none.message, "no Broyden iterations remaining");

    const auto pt_none = metis::detail::solve_pseudo_transient(root_fns.first, root_fns.second,
                                                               nonconverged_state, opts, 0);
    EXPECT_FALSE(pt_none.converged);
    EXPECT_EQ(pt_none.message, "no pseudo-transient iterations remaining");

    const auto nan_state = make_scalar_state(0.0, 1.0, std::numeric_limits<double>::quiet_NaN());
    const auto tr_nan =
        metis::detail::solve_trust_region(root_fns.first, root_fns.second, nan_state, opts, 1);
    EXPECT_NE(tr_nan.message.find("non-finite"), std::string::npos);

    const auto ls_nan =
        metis::detail::solve_line_search(root_fns.first, root_fns.second, nan_state, opts, 1);
    EXPECT_NE(ls_nan.message.find("non-finite"), std::string::npos);

    const auto br_nan =
        metis::detail::solve_broyden(root_fns.first, root_fns.second, nan_state, opts, 1);
    EXPECT_NE(br_nan.message.find("non-finite"), std::string::npos);

    const auto pt_tiny = metis::detail::solve_pseudo_transient(
        root_fns.first, root_fns.second, make_scalar_state(0.0, 1.0, 1e20), opts, 1);
    EXPECT_NE(pt_tiny.message.find("tiny step"), std::string::npos);

    auto x_sym = metis::sym("x_root");
    metis::Function f_identity("f_identity", {x_sym}, {x_sym});
    metis::NewtonSolver solver(f_identity);

    Eigen::VectorXd bad_guess = Eigen::VectorXd::Zero(2);
    EXPECT_THROW(solver.solve(bad_guess), metis::InvalidArgument);

    Eigen::VectorXd zero_guess = Eigen::VectorXd::Zero(1);
    const auto solved = solver.solve(zero_guess);
    EXPECT_TRUE(solved.converged);
    EXPECT_NE(solved.message.find("Initial guess"), std::string::npos);

    metis::Function f_no_root("f_no_root", {x_sym}, {x_sym * x_sym + 1.0});
    metis::RootFinderOptions fail_opts;
    fail_opts.max_iter = 2;
    fail_opts.strategy = metis::RootSolveStrategy::TrustRegionNewton;
    const auto failed = metis::rootfinder<double>(f_no_root, zero_guess, fail_opts);
    EXPECT_FALSE(failed.converged);
    EXPECT_EQ(failed.method, metis::RootSolveMethod::TrustRegionNewton);
    EXPECT_NE(failed.message.find("Failed to converge"), std::string::npos);

    auto p_sym = metis::sym("p_root");
    auto x_row_sym = metis::sym("x_row_root", 1, 2);
    auto x_col_sym = metis::sym("x_col_root", 2, 1);
    metis::Function impl_bad_input("impl_bad_input", {x_row_sym, p_sym}, {casadi::MX::zeros(2, 1)});
    EXPECT_THROW(metis::create_implicit_function(impl_bad_input, x_guess), metis::InvalidArgument);

    metis::Function impl_bad_output("impl_bad_output", {x_col_sym, p_sym},
                                    {casadi::MX::zeros(1, 2)});
    EXPECT_THROW(metis::create_implicit_function(impl_bad_output, x_guess), metis::InvalidArgument);

    metis::Function impl_bad_dim("impl_bad_dim", {x_col_sym, p_sym}, {casadi::MX::zeros(1, 1)});
    EXPECT_THROW(metis::create_implicit_function(impl_bad_dim, x_guess), metis::InvalidArgument);

    metis::Function impl_good("impl_good", {x_col_sym, p_sym}, {x_col_sym});
    EXPECT_THROW(metis::create_implicit_function(impl_good, short_guess), metis::InvalidArgument);
}

TEST(QuadratureCoverage, HelperFunctionsAndValidationErrors) {
    EXPECT_THROW(metis::detail::validate_order(0, "quad"), metis::InvalidArgument);
    EXPECT_THROW(metis::detail::validate_level(0, "quad"), metis::InvalidArgument);

    EXPECT_DOUBLE_EQ(metis::detail::binomial(5, 2), 10.0);
    EXPECT_DOUBLE_EQ(metis::detail::binomial(4, -1), 0.0);
    EXPECT_DOUBLE_EQ(metis::detail::binomial(4, 5), 0.0);
    EXPECT_EQ(metis::detail::clenshaw_curtis_order_from_level(1), 1);
    EXPECT_EQ(metis::detail::clenshaw_curtis_order_from_level(3), 5);
    EXPECT_EQ(metis::detail::gauss_order_from_level(4), 4);
    EXPECT_TRUE(metis::detail::is_bounded_support(metis::legendre_dimension()));
    EXPECT_TRUE(metis::detail::is_bounded_support(metis::jacobi_dimension(1.0, 2.0)));
    EXPECT_FALSE(metis::detail::is_bounded_support(metis::hermite_dimension()));

    EXPECT_DOUBLE_EQ(metis::detail::standard_normal_moment(3), 0.0);
    EXPECT_NEAR(metis::detail::standard_normal_moment(4), 3.0, 1e-12);
    EXPECT_NEAR(metis::detail::shifted_beta_moment(1, 1.0, 2.0), 0.2, 1e-12);

    EXPECT_THROW(metis::detail::probability_moment(metis::legendre_dimension(), -1),
                 metis::InvalidArgument);
    EXPECT_NEAR(metis::detail::probability_moment(metis::hermite_dimension(), 4), 3.0, 1e-12);
    EXPECT_DOUBLE_EQ(metis::detail::probability_moment(metis::legendre_dimension(), 3), 0.0);
    EXPECT_GT(metis::detail::probability_moment(metis::jacobi_dimension(1.0, 2.0), 2), 0.0);
    EXPECT_NEAR(metis::detail::probability_moment(metis::laguerre_dimension(), 3), 6.0, 1e-12);

    metis::PolynomialChaosDimension bad_dimension = metis::legendre_dimension();
    bad_dimension.family = static_cast<metis::PolynomialChaosFamily>(999);
    EXPECT_THROW(metis::detail::probability_moment(bad_dimension, 0), metis::InvalidArgument);

    metis::NumericVector singleton_node(1);
    singleton_node << 0.0;
    EXPECT_THROW(metis::detail::clenshaw_curtis_probability_weights(metis::hermite_dimension(),
                                                                    singleton_node),
                 metis::InvalidArgument);
    const auto singleton_weights = metis::detail::clenshaw_curtis_probability_weights(
        metis::jacobi_dimension(1.0, 2.0), singleton_node);
    EXPECT_DOUBLE_EQ(singleton_weights(0), 1.0);

    const metis::NumericVector jacobi_nodes = metis::cgl_nodes(3);
    const metis::NumericVector jacobi_weights = metis::detail::clenshaw_curtis_probability_weights(
        metis::jacobi_dimension(1.0, 2.0), jacobi_nodes);
    EXPECT_NEAR(jacobi_weights.sum(), 1.0, 1e-12);
}

TEST(QuadratureCoverage, RuleConstructionAndSparseGridValidation) {
    const auto hermite_rule = metis::detail::gauss_rule(metis::hermite_dimension(), 3, 2);
    const auto jacobi_rule = metis::detail::gauss_rule(metis::jacobi_dimension(1.0, 2.0), 3, 2);
    const auto laguerre_rule = metis::detail::gauss_rule(metis::laguerre_dimension(), 3, 2);
    EXPECT_EQ(hermite_rule.nodes.size(), 3);
    EXPECT_EQ(jacobi_rule.nodes.size(), 3);
    EXPECT_EQ(laguerre_rule.nodes.size(), 3);

    metis::PolynomialChaosDimension bad_dimension = metis::legendre_dimension();
    bad_dimension.family = static_cast<metis::PolynomialChaosFamily>(999);
    EXPECT_THROW(metis::detail::gauss_rule(bad_dimension, 2, 1), metis::InvalidArgument);

    EXPECT_THROW(metis::detail::gauss_kronrod_rule(metis::hermite_dimension(), 7, 1),
                 metis::InvalidArgument);
    EXPECT_THROW(metis::detail::gauss_kronrod_rule(metis::legendre_dimension(), 9, 1),
                 metis::InvalidArgument);
    EXPECT_THROW(metis::detail::clenshaw_curtis_rule(metis::hermite_dimension(), 3, 1),
                 metis::InvalidArgument);

    const auto cc_rule = metis::detail::clenshaw_curtis_rule(metis::legendre_dimension(), 1, 1);
    ASSERT_EQ(cc_rule.nodes.size(), 1);
    EXPECT_DOUBLE_EQ(cc_rule.nodes(0), 0.0);
    EXPECT_DOUBLE_EQ(cc_rule.weights(0), 1.0);

    const auto compositions = metis::detail::positive_compositions(3, 5);
    ASSERT_FALSE(compositions.empty());
    for (const auto &composition : compositions) {
        EXPECT_EQ(std::accumulate(composition.begin(), composition.end(), 0), 5);
    }

    metis::NumericVector point(2);
    point << 0.25, -0.25;
    EXPECT_THROW(metis::detail::sample_key(point, 0.0), metis::InvalidArgument);
    EXPECT_FALSE(metis::detail::sample_key(point, 1e-3).empty());

    EXPECT_THROW(
        metis::stochastic_quadrature_rule(metis::legendre_dimension(), 3,
                                          static_cast<metis::StochasticQuadratureRule>(999)),
        metis::InvalidArgument);
    EXPECT_THROW(
        metis::stochastic_quadrature_level(metis::legendre_dimension(), 3,
                                           metis::StochasticQuadratureRule::GaussKronrod15),
        metis::InvalidArgument);

    EXPECT_THROW(metis::tensor_product_quadrature({}), metis::InvalidArgument);

    metis::UnivariateQuadratureRule bad_rule;
    bad_rule.nodes.resize(0);
    bad_rule.weights.resize(0);
    EXPECT_THROW(metis::tensor_product_quadrature({bad_rule}), metis::InvalidArgument);

    bad_rule.nodes.resize(1);
    bad_rule.nodes << 0.0;
    bad_rule.weights.resize(2);
    bad_rule.weights << 1.0, 0.0;
    EXPECT_THROW(metis::tensor_product_quadrature({bad_rule}), metis::InvalidArgument);

    metis::SmolyakQuadratureOptions smolyak_opts;
    smolyak_opts.merge_tolerance = 0.0;
    EXPECT_THROW(metis::smolyak_sparse_grid({metis::legendre_dimension()}, 1, smolyak_opts),
                 metis::InvalidArgument);

    smolyak_opts = metis::SmolyakQuadratureOptions();
    smolyak_opts.zero_weight_tolerance = -1.0;
    EXPECT_THROW(metis::smolyak_sparse_grid({metis::legendre_dimension()}, 1, smolyak_opts),
                 metis::InvalidArgument);

    smolyak_opts = metis::SmolyakQuadratureOptions();
    smolyak_opts.zero_weight_tolerance = 2.0;
    EXPECT_THROW(metis::smolyak_sparse_grid({metis::legendre_dimension()}, 1, smolyak_opts),
                 metis::RuntimeError);

    metis::PolynomialChaosBasis multivariate_basis(
        {metis::legendre_dimension(), metis::hermite_dimension()}, 1);
    const auto rule = metis::stochastic_quadrature_rule(metis::legendre_dimension(), 3);
    metis::NumericVector values = metis::NumericVector::Ones(rule.nodes.size());
    EXPECT_THROW(metis::pce_projection_coefficients(multivariate_basis, rule, values),
                 metis::InvalidArgument);

    metis::NumericMatrix matrix_values(rule.nodes.size(), 1);
    matrix_values.setOnes();
    EXPECT_THROW(metis::pce_projection_coefficients(multivariate_basis, rule, matrix_values),
                 metis::InvalidArgument);
}

TEST(LinalgCoverage, ValidationAndSymbolicSmallMatrixBranches) {
    EXPECT_THROW(metis::detail::validate_linear_solve_dims(0, 1, 0, "solve"),
                 metis::InvalidArgument);
    EXPECT_THROW(metis::detail::validate_linear_solve_dims(2, 2, 1, "solve"),
                 metis::InvalidArgument);
    EXPECT_THROW(metis::detail::validate_square_required(2, 3, "solve", "LLT"),
                 metis::InvalidArgument);

    metis::LinearSolvePolicy iterative_policy = metis::LinearSolvePolicy::iterative();
    iterative_policy.tolerance = 0.0;
    EXPECT_THROW(metis::detail::validate_iterative_policy(iterative_policy, "solve"),
                 metis::InvalidArgument);
    iterative_policy = metis::LinearSolvePolicy::iterative();
    iterative_policy.max_iterations = 0;
    EXPECT_THROW(metis::detail::validate_iterative_policy(iterative_policy, "solve"),
                 metis::InvalidArgument);
    iterative_policy = metis::LinearSolvePolicy::iterative();
    iterative_policy.gmres_restart = 0;
    EXPECT_THROW(metis::detail::validate_iterative_policy(iterative_policy, "solve"),
                 metis::InvalidArgument);

    metis::NumericMatrix dense(2, 2);
    dense << 0.0, 0.0, 0.0, 2.0;
    const metis::SparseMatrix sparse = metis::to_sparse(dense);
    metis::NumericVector rhs(2);
    rhs << 1.0, 2.0;

    const auto none_prec = metis::detail::make_preconditioner(
        sparse, metis::LinearSolvePolicy::iterative(metis::IterativeKrylovSolver::BiCGSTAB,
                                                    metis::IterativePreconditioner::None));
    EXPECT_TRUE(none_prec(rhs).isApprox(rhs, 1e-12));

    const auto diag_prec = metis::detail::make_preconditioner(
        sparse, metis::LinearSolvePolicy::iterative(metis::IterativeKrylovSolver::BiCGSTAB,
                                                    metis::IterativePreconditioner::Diagonal));
    const metis::NumericVector scaled_rhs = diag_prec(rhs);
    EXPECT_DOUBLE_EQ(scaled_rhs(0), 1.0);
    EXPECT_DOUBLE_EQ(scaled_rhs(1), 1.0);

    metis::LinearSolvePolicy bad_prec_policy = metis::LinearSolvePolicy::iterative();
    bad_prec_policy.iterative_preconditioner = static_cast<metis::IterativePreconditioner>(999);
    EXPECT_THROW(metis::detail::make_preconditioner(sparse, bad_prec_policy),
                 metis::InvalidArgument);

    metis::NumericMatrix nonsquare(2, 3);
    nonsquare << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    EXPECT_THROW(metis::solve(nonsquare, rhs,
                              metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::FullPivLU)),
                 metis::InvalidArgument);
    EXPECT_THROW(metis::solve(nonsquare, rhs,
                              metis::LinearSolvePolicy::sparse_direct(
                                  metis::SparseDirectLinearSolver::SimplicialLLT)),
                 metis::InvalidArgument);

    metis::NumericMatrix indefinite(2, 2);
    indefinite << 0.0, 1.0, 1.0, 0.0;
    EXPECT_THROW(metis::solve(indefinite, rhs,
                              metis::LinearSolvePolicy::dense(metis::DenseLinearSolver::LLT)),
                 metis::InvalidArgument);

    metis::NumericVector zero_vec = metis::NumericVector::Zero(2);
    EXPECT_THROW(metis::detail::normalize_vector(zero_vec), metis::InvalidArgument);

    metis::NumericMatrix numeric_scalar_matrix(1, 1);
    numeric_scalar_matrix << 7.0;
    const auto numeric_eig = metis::eig(numeric_scalar_matrix);
    EXPECT_DOUBLE_EQ(numeric_eig.eigenvalues(0), 7.0);
    EXPECT_DOUBLE_EQ(numeric_eig.eigenvectors(0, 0), 1.0);

    metis::SymbolicMatrix symbolic_1x1(1, 1);
    symbolic_1x1(0, 0) = 5.0;
    const auto sym_eig_1x1 = metis::eig_symmetric(symbolic_1x1);
    EXPECT_DOUBLE_EQ(metis::eval(sym_eig_1x1.eigenvalues)(0), 5.0);
    EXPECT_DOUBLE_EQ(metis::eval(sym_eig_1x1.eigenvectors)(0, 0), 1.0);

    metis::SymbolicMatrix symbolic_2x2(2, 2);
    symbolic_2x2 << 3.0, 1.0, 1.0, 3.0;
    const auto sym_eig_2x2 = metis::eig_symmetric(symbolic_2x2);
    const metis::NumericVector sym_values_2x2 = metis::eval(sym_eig_2x2.eigenvalues);
    EXPECT_NEAR(sym_values_2x2(0), 2.0, 1e-12);
    EXPECT_NEAR(sym_values_2x2(1), 4.0, 1e-12);

    metis::SymbolicMatrix symbolic_4x4(4, 4);
    symbolic_4x4.setZero();
    for (int i = 0; i < 4; ++i) {
        symbolic_4x4(i, i) = 1.0;
    }
    EXPECT_THROW(metis::eig_symmetric(symbolic_4x4), metis::InvalidArgument);
}

TEST(IntegrateCoverage, DetailValidationAndMassMatrixBranches) {
    EXPECT_STREQ(metis::detail::method_name(metis::SecondOrderIntegratorMethod::StormerVerlet),
                 "Stormer-Verlet");
    EXPECT_STREQ(metis::detail::method_name(metis::SecondOrderIntegratorMethod::RungeKuttaNystrom4),
                 "RKN4");
    EXPECT_THROW(metis::detail::method_name(static_cast<metis::SecondOrderIntegratorMethod>(999)),
                 metis::InvalidArgument);

    EXPECT_STREQ(metis::detail::method_name(metis::MassMatrixIntegratorMethod::RosenbrockEuler),
                 "Rosenbrock-Euler");
    EXPECT_STREQ(metis::detail::method_name(metis::MassMatrixIntegratorMethod::Bdf1), "BDF1");
    EXPECT_THROW(metis::detail::method_name(static_cast<metis::MassMatrixIntegratorMethod>(999)),
                 metis::InvalidArgument);

    EXPECT_THROW(metis::detail::validate_eval_count("ivp", 1), metis::IntegrationError);

    metis::SecondOrderIvpOptions second_order_opts;
    second_order_opts.substeps = 0;
    EXPECT_THROW(metis::detail::validate_second_order_options(second_order_opts, "ivp"),
                 metis::IntegrationError);

    metis::MassMatrixIvpOptions mass_opts;
    mass_opts.substeps = 0;
    EXPECT_THROW(metis::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 metis::IntegrationError);
    mass_opts = metis::MassMatrixIvpOptions();
    mass_opts.abstol = 0.0;
    EXPECT_THROW(metis::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 metis::IntegrationError);
    mass_opts = metis::MassMatrixIvpOptions();
    mass_opts.reltol = 0.0;
    EXPECT_THROW(metis::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 metis::IntegrationError);
    mass_opts = metis::MassMatrixIvpOptions();
    mass_opts.finite_difference_epsilon = 0.0;
    EXPECT_THROW(metis::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 metis::IntegrationError);
    mass_opts = metis::MassMatrixIvpOptions();
    mass_opts.max_newton_iterations = 0;
    EXPECT_THROW(metis::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 metis::IntegrationError);
    mass_opts = metis::MassMatrixIvpOptions();
    mass_opts.newton_tolerance = 0.0;
    EXPECT_THROW(metis::detail::validate_mass_matrix_options(mass_opts, "ivp"),
                 metis::IntegrationError);

    metis::NumericVector empty_q0(0);
    metis::NumericVector q0(1);
    q0 << 1.0;
    metis::NumericVector v0(1);
    v0 << 0.0;
    metis::NumericVector bad_v0(2);
    bad_v0 << 0.0, 0.0;
    EXPECT_THROW(metis::detail::validate_second_order_initial_state(empty_q0, v0),
                 metis::IntegrationError);
    EXPECT_THROW(metis::detail::validate_second_order_initial_state(q0, bad_v0),
                 metis::IntegrationError);

    EXPECT_DOUBLE_EQ(metis::detail::inf_norm(q0), 1.0);
    EXPECT_TRUE(metis::detail::is_constant_zero(metis::SymbolicScalar(0.0)));
    auto symbolic_t = metis::sym("t_cov");
    EXPECT_FALSE(metis::detail::is_constant_zero(symbolic_t));

    metis::NumericVector x(2);
    x << 1.0, -2.0;
    const metis::NumericMatrix J = metis::detail::finite_difference_jacobian(
        [](const metis::NumericVector &state) { return (2.0 * state).eval(); }, x, 1e-6);
    EXPECT_TRUE(J.isApprox(2.0 * metis::NumericMatrix::Identity(2, 2), 1e-5));

    EXPECT_THROW(metis::detail::evaluate_mass_matrix(
                     [](double, const metis::NumericVector &state) {
                         return metis::NumericMatrix::Identity(state.size() + 1, state.size() + 1);
                     },
                     0.0, x, "mass"),
                 metis::IntegrationError);

    metis::MassMatrixIvpOptions opts;
    EXPECT_THROW(metis::detail::rosenbrock_euler_step(
                     [](double, const metis::NumericVector &) {
                         metis::NumericVector rhs(1);
                         rhs << 1.0;
                         return rhs;
                     },
                     [](double, const metis::NumericVector &state) {
                         return metis::NumericMatrix::Identity(state.size(), state.size());
                     },
                     x, 0.0, 0.1, opts),
                 metis::IntegrationError);

    EXPECT_THROW(metis::detail::bdf1_step(
                     [](double, const metis::NumericVector &) {
                         metis::NumericVector rhs(1);
                         rhs << 1.0;
                         return rhs;
                     },
                     [](double, const metis::NumericVector &state) {
                         return metis::NumericMatrix::Identity(state.size(), state.size());
                     },
                     x, 0.0, 0.1, opts),
                 metis::IntegrationError);

    EXPECT_THROW(metis::solve_second_order_ivp(
                     [](double, const metis::NumericVector &state) { return (-state).eval(); },
                     {0.0, 1.0}, empty_q0, v0, 10),
                 metis::IntegrationError);
    EXPECT_THROW(metis::solve_second_order_ivp(
                     [](double, const metis::NumericVector &state) { return (-state).eval(); },
                     {0.0, 1.0}, q0, bad_v0, 10),
                 metis::IntegrationError);

    metis::NumericVector empty_y0(0);
    EXPECT_THROW(metis::solve_ivp_mass_matrix(
                     [](double, const metis::NumericVector &state) { return state; },
                     [](double, const metis::NumericVector &state) {
                         return metis::NumericMatrix::Identity(state.size(), state.size());
                     },
                     {0.0, 1.0}, empty_y0, 10),
                 metis::IntegrationError);

    metis::NumericVector y0(2);
    y0 << 1.0, 0.0;
    EXPECT_THROW(metis::solve_ivp_mass_matrix(
                     [](double, const metis::NumericVector &) {
                         metis::NumericVector rhs(1);
                         rhs << 1.0;
                         return rhs;
                     },
                     [](double, const metis::NumericVector &state) {
                         return metis::NumericMatrix::Identity(state.size(), state.size());
                     },
                     {0.0, 1.0}, y0, 2),
                 metis::IntegrationError);

    EXPECT_THROW(metis::solve_ivp_mass_matrix(
                     [](double, const metis::NumericVector &state) { return state; },
                     [](double, const metis::NumericVector &state) {
                         return metis::NumericMatrix::Identity(state.size() + 1, state.size() + 1);
                     },
                     {0.0, 1.0}, y0, 2),
                 metis::IntegrationError);
}

TEST(IntegrateCoverage, SymbolicMassMatrixExprErrorAndOdeOnlyBranches) {
    auto t = metis::sym("t_cov_mass");
    auto y = metis::sym("y_cov_mass");
    metis::NumericVector y0(1);
    y0 << 1.0;

    metis::MassMatrixIvpOptions opts;
    opts.symbolic_integrator_options["max_num_steps"] = 1000;
    const auto ode_only = metis::solve_ivp_mass_matrix_expr(-y, casadi::MX::ones(1, 1), t, y,
                                                            {0.0, 1.0}, y0, 8, opts);
    EXPECT_TRUE(ode_only.success);
    EXPECT_NEAR(ode_only.y(0, ode_only.y.cols() - 1), std::exp(-1.0), 1e-4);

    EXPECT_THROW(
        metis::solve_ivp_mass_matrix_expr(-y, casadi::MX::zeros(1, 1), t, y, {0.0, 1.0}, y0, 8),
        metis::IntegrationError);

    EXPECT_THROW(
        metis::solve_ivp_mass_matrix_expr(-y, casadi::MX::zeros(2, 2), t, y, {0.0, 1.0}, y0, 8),
        metis::IntegrationError);

    EXPECT_THROW(metis::solve_ivp_mass_matrix_expr(casadi::MX::zeros(1, 2), casadi::MX::ones(1, 1),
                                                   t, y, {0.0, 1.0}, y0, 8),
                 metis::IntegrationError);
}

TEST(PolynomialChaosCoverage, ValidationErrorsAndMatrixRegressionPaths) {
    EXPECT_THROW(metis::detail::validate_degree(-1, "pce"), metis::InvalidArgument);
    EXPECT_THROW(metis::detail::validate_dimension(metis::jacobi_dimension(-1.0, 0.0), "pce"),
                 metis::InvalidArgument);
    EXPECT_THROW(metis::detail::validate_dimension(metis::laguerre_dimension(-1.0), "pce"),
                 metis::InvalidArgument);

    EXPECT_DOUBLE_EQ(metis::detail::raw_jacobi_polynomial(0, 0.25, 1.0, 2.0), 1.0);
    EXPECT_DOUBLE_EQ(metis::detail::raw_jacobi_polynomial(1, 0.25, 1.0, 2.0), 0.125);
    EXPECT_DOUBLE_EQ(metis::detail::raw_laguerre_polynomial(0, 0.2, 0.5), 1.0);
    EXPECT_DOUBLE_EQ(metis::detail::raw_laguerre_polynomial(1, 0.2, 0.5), 1.3);
    EXPECT_DOUBLE_EQ(metis::pce_squared_norm(metis::legendre_dimension(), 2), 1.0);

    metis::NumericMatrix underdetermined(2, 3);
    underdetermined << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    EXPECT_THROW(metis::detail::regression_operator(underdetermined, 0.0, "pce"),
                 metis::InvalidArgument);
    EXPECT_THROW(
        metis::detail::regression_operator(metis::NumericMatrix::Identity(3, 3), -1.0, "pce"),
        metis::InvalidArgument);

    EXPECT_THROW(metis::PolynomialChaosBasis({}, 1), metis::InvalidArgument);
    EXPECT_THROW(metis::PolynomialChaosBasis({metis::legendre_dimension()}, -1),
                 metis::InvalidArgument);

    metis::PolynomialChaosBasis basis({metis::legendre_dimension()}, 2);

    metis::NumericVector bad_point(2);
    bad_point << 0.0, 1.0;
    EXPECT_THROW(basis.evaluate(bad_point), metis::InvalidArgument);

    metis::NumericMatrix empty_samples(0, 1);
    EXPECT_THROW(basis.evaluate(empty_samples), metis::InvalidArgument);

    metis::NumericMatrix bad_samples(1, 2);
    bad_samples << 0.0, 1.0;
    EXPECT_THROW(basis.evaluate(bad_samples), metis::InvalidArgument);

    metis::NumericMatrix samples(4, 1);
    samples << -1.0, -0.5, 0.5, 1.0;

    metis::NumericVector vector_values(4);
    vector_values << 1.0, 0.5, 0.5, 1.0;
    metis::NumericMatrix matrix_values(4, 2);
    matrix_values.col(0) = vector_values;
    matrix_values.col(1) = 2.0 * vector_values;
    const metis::NumericMatrix regressed =
        metis::pce_regression_coefficients(basis, samples, matrix_values, 1e-3);
    EXPECT_EQ(regressed.rows(), basis.size());
    EXPECT_EQ(regressed.cols(), 2);

    metis::NumericVector bad_vector_values(3);
    bad_vector_values << 1.0, 2.0, 3.0;
    EXPECT_THROW(metis::pce_regression_coefficients(basis, samples, bad_vector_values, 1e-3),
                 metis::InvalidArgument);

    metis::NumericMatrix bad_matrix_values(3, 2);
    bad_matrix_values.setOnes();
    EXPECT_THROW(metis::pce_regression_coefficients(basis, samples, bad_matrix_values, 1e-3),
                 metis::InvalidArgument);

    metis::NumericVector bad_weights(3);
    bad_weights.setOnes();
    EXPECT_THROW(metis::pce_projection_coefficients(basis, samples, bad_weights, vector_values),
                 metis::InvalidArgument);

    metis::NumericMatrix projection_matrix_values(3, 1);
    projection_matrix_values.setOnes();
    metis::NumericVector good_weights = metis::NumericVector::Ones(samples.rows());
    EXPECT_THROW(
        metis::pce_projection_coefficients(basis, samples, good_weights, projection_matrix_values),
        metis::InvalidArgument);

    metis::NumericVector empty_coeffs(0);
    EXPECT_THROW(metis::pce_mean(empty_coeffs), metis::InvalidArgument);

    metis::NumericVector wrong_size_coeffs(2);
    wrong_size_coeffs.setZero();
    EXPECT_THROW(metis::pce_variance(basis, wrong_size_coeffs), metis::InvalidArgument);
}
