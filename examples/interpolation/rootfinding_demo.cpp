#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <string>

/**
 * Root Finding Demo
 *
 * Demonstrates:
 * 1. Automatic globalization fallback from a singular initial Jacobian.
 * 2. Reusing a persistent solver with an explicit nonlinear strategy.
 * 3. Creating an implicit function z(p) from G(z, p)=0 and differentiating it.
 */
namespace {

std::string method_name(metis::RootSolveMethod method) {
    switch (method) {
    case metis::RootSolveMethod::None:
        return "none";
    case metis::RootSolveMethod::TrustRegionNewton:
        return "trust-region Newton";
    case metis::RootSolveMethod::LineSearchNewton:
        return "line-search Newton";
    case metis::RootSolveMethod::QuasiNewtonBroyden:
        return "quasi-Newton Broyden";
    case metis::RootSolveMethod::PseudoTransientContinuation:
        return "pseudo-transient continuation";
    }
    return "unknown";
}

void print_result(const std::string &label, const metis::RootResult<double> &result) {
    std::cout << label << "\n";
    std::cout << "  converged: " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  method:    " << method_name(result.method) << "\n";
    std::cout << "  iterations:" << result.iterations << "\n";
    std::cout << "  residual:  " << result.residual_norm << "\n";
    std::cout << "  step norm: " << result.step_norm << "\n";
    std::cout << "  message:   " << result.message << "\n";
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "=== Root Finding Demo ===\n";

    // ---------------------------------------------------------
    // 1. Automatic globalization stack
    // x^2 - 1 = 0 started from x0 = 0. The initial Jacobian is singular,
    // so Auto must fall through to pseudo-transient continuation.
    // ---------------------------------------------------------
    std::cout << "\n--- 1. Automatic Globalization Fallback ---\n";
    auto xs = metis::sym("xs");
    metis::Function singular_start("SingularStart", {xs}, {xs * xs - 1.0});

    Eigen::VectorXd singular_guess(1);
    singular_guess << 0.0;

    metis::RootFinderOptions auto_opts;
    auto_opts.max_iter = 60;
    auto_opts.pseudo_transient_dt0 = 0.1;

    auto auto_res = metis::rootfinder<double>(singular_start, singular_guess, auto_opts);
    print_result("Auto solve for x^2 - 1 = 0 from x0 = 0", auto_res);
    if (auto_res.converged) {
        std::cout << "  solution:  " << auto_res.x(0) << "\n";
    }

    // ---------------------------------------------------------
    // 2. Solving a 2D system with an explicit strategy
    // x^2 + y^2 - 1 = 0 (Circle)
    // x^2 - y = 0       (Parabola)
    // ---------------------------------------------------------
    std::cout << "\n--- 2. Persistent Solver With Explicit Strategy ---\n";
    std::cout << "System:\n  x^2 + y^2 - 1 = 0\n  x^2 - y = 0\n";

    auto [q, q_mx] = metis::sym_vec_pair("q", 2);

    auto x = q(0);
    auto y = q(1);

    auto F1 = x * x + y * y - 1.0;
    auto F2 = x * x - y;

    // Residual vector [F1, F2]
    auto Res = metis::SymbolicVector(2);
    Res << F1, F2;

    metis::Function F("System2D", {metis::SymbolicArg(q_mx)}, {metis::to_mx(Res)});

    metis::RootFinderOptions line_search_opts;
    line_search_opts.strategy = metis::RootSolveStrategy::LineSearchNewton;
    metis::NewtonSolver solver(F, line_search_opts);

    Eigen::VectorXd q0(2);
    q0 << 0.5, 0.5;

    std::cout << "Solving from initial guess: [" << q0.transpose() << "]\n";
    auto res = solver.solve(q0);
    print_result("Positive branch solve", res);

    if (res.converged) {
        double x_sol = res.x(0);
        double y_sol = res.x(1);
        std::cout << "  solution:  [" << x_sol << ", " << y_sol << "]\n";
        std::cout << "Residuals: " << std::abs(x_sol * x_sol + y_sol * y_sol - 1.0) << ", "
                  << std::abs(x_sol * x_sol - y_sol) << "\n";
    }

    // ---------------------------------------------------------
    // 3. Implicit Function Creation
    // Solve G(z, p) = z^3 + p*z + 1 = 0 for z(p)
    // ---------------------------------------------------------
    std::cout << "\n--- 3. Implicit Function z(p) from G(z, p)=0 ---\n";
    std::cout << "G(z, p) = z^3 + p*z + 1 = 0\n";

    auto z = metis::sym("z");
    auto p = metis::sym("p");
    auto G_expr = z * z * z + p * z + 1.0;

    metis::Function G("G", {z, p}, {G_expr});

    Eigen::VectorXd z_guess(1);
    z_guess << -1.0;

    auto z_of_p = metis::create_implicit_function(G, z_guess);

    auto res_0 = z_of_p(0.0)[0];
    std::cout << "z(0): " << res_0(0, 0) << " (Expected -1.0)\n";

    auto res_1 = z_of_p(1.0)[0];
    std::cout << "z(1): " << res_1(0, 0) << " (Expected ~ -0.6823)\n";

    std::cout << "Computing dz/dp symbolically...\n";

    auto J_sym = metis::jacobian(z_of_p(p)[0](0), p);
    metis::Function J_fn("dzdp", {p}, {J_sym});

    auto dzdp_0 = J_fn(0.0)[0];
    std::cout << "dz/dp at p=0: " << dzdp_0(0, 0) << " (Expected 0.33333)\n";

    std::cout << "\n=== Demo Complete ===\n";
    return 0;
}
