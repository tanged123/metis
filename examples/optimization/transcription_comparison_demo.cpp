/**
 * @file transcription_comparison_demo.cpp
 * @brief Unified comparison of trajectory transcription methods
 *
 * Solves the same brachistochrone problem with all transcriptions and
 * prints a side-by-side comparison.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace metis;

namespace {

struct RunResult {
    std::string method;
    std::string grid_label;
    bool success = false;
    double t_opt = -1.0;
    double abs_error = -1.0;
    int n_vars = 0;
    int n_constraints = 0;
    int n_iters = -1;
    double solve_ms = -1.0;
    std::string failure_reason;
};

SymbolicVector brachistochrone_ode(const SymbolicVector &state, const SymbolicVector &control,
                                   const SymbolicScalar & /*t*/) {
    constexpr double g = 9.80665;
    SymbolicScalar v = state(2);
    SymbolicScalar theta = control(0);

    SymbolicVector dxdt(3);
    dxdt(0) = v * metis::sin(theta);
    dxdt(1) = -v * metis::cos(theta);
    dxdt(2) = g * metis::cos(theta);
    return dxdt;
}

void apply_brachistochrone_constraints(Opti &opti, const SymbolicMatrix &x, const SymbolicMatrix &u,
                                       auto &transcription) {
    transcription.set_initial_state(NumericVector{{0.0, 10.0, 0.001}});
    transcription.set_final_state(0, 10.0);
    transcription.set_final_state(1, 5.0);

    opti.subject_to_bounds(u.col(0), 0.01, M_PI - 0.01);
    opti.subject_to_lower(x.col(2), 0.0);
}

void apply_brachistochrone_initial_guess(Opti &opti, const SymbolicMatrix &x,
                                         const SymbolicMatrix &u, const NumericVector &tau_state) {
    for (int k = 0; k < x.rows(); ++k) {
        const double s = tau_state(k);
        const double x_guess = 10.0 * s;
        const double y_guess = 10.0 - 5.0 * s;
        const double v_guess = std::sqrt(std::max(0.0, 2.0 * 9.80665 * (10.0 - y_guess)));

        opti.set_initial(x(k, 0), x_guess);
        opti.set_initial(x(k, 1), y_guess);
        opti.set_initial(x(k, 2), std::max(1e-3, v_guess));
    }

    for (int k = 0; k < u.rows(); ++k) {
        const double s =
            (u.rows() > 1) ? static_cast<double>(k) / static_cast<double>(u.rows() - 1) : 0.5;
        const double theta_guess = 1.2 - 0.4 * s;
        opti.set_initial(u(k, 0), theta_guess);
    }
}

RunResult solve_collocation(int n_nodes, double ref_time) {
    RunResult out;
    out.method = "DirectCollocation";
    out.grid_label = "N=" + std::to_string(n_nodes);

    Opti opti;
    DirectCollocation dc(opti);
    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    CollocationOptions opts;
    opts.scheme = CollocationScheme::HermiteSimpson;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = dc.setup(3, 1, 0.0, T, opts);
    apply_brachistochrone_initial_guess(opti, x, u, tau);
    dc.set_dynamics(brachistochrone_ode);
    dc.add_dynamics_constraints();
    apply_brachistochrone_constraints(opti, x, u, dc);
    opti.minimize(T);

    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto sol = opti.solve({.max_iter = 500, .verbose = false});
        const auto t1 = std::chrono::steady_clock::now();

        out.success = true;
        out.t_opt = sol.value(T);
        out.abs_error = std::abs(out.t_opt - ref_time);
        out.n_vars = static_cast<int>(opti.casadi_opti().x().numel());
        out.n_constraints = static_cast<int>(opti.casadi_opti().g().numel());
        out.n_iters = static_cast<int>(sol.stats().at("iter_count"));
        out.solve_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    } catch (const std::exception &e) {
        out.failure_reason = e.what();
    }
    return out;
}

RunResult solve_multishoot(int n_intervals, double ref_time) {
    RunResult out;
    out.method = "MultipleShooting";
    out.grid_label = "N=" + std::to_string(n_intervals);

    Opti opti;
    MultipleShooting ms(opti);
    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    MultiShootingOptions opts;
    opts.n_intervals = n_intervals;
    opts.integrator = "cvodes";
    opts.tol = 1e-6;

    auto [x, u, tau] = ms.setup(3, 1, 0.0, T, opts);
    apply_brachistochrone_initial_guess(opti, x, u, tau);
    ms.set_dynamics(brachistochrone_ode);
    ms.add_dynamics_constraints();
    apply_brachistochrone_constraints(opti, x, u, ms);
    opti.minimize(T);

    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto sol = opti.solve({.max_iter = 500, .verbose = false});
        const auto t1 = std::chrono::steady_clock::now();

        out.success = true;
        out.t_opt = sol.value(T);
        out.abs_error = std::abs(out.t_opt - ref_time);
        out.n_vars = static_cast<int>(opti.casadi_opti().x().numel());
        out.n_constraints = static_cast<int>(opti.casadi_opti().g().numel());
        out.n_iters = static_cast<int>(sol.stats().at("iter_count"));
        out.solve_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    } catch (const std::exception &e) {
        out.failure_reason = e.what();
    }
    return out;
}

RunResult solve_pseudospectral(int n_nodes, double ref_time) {
    RunResult out;
    out.method = "Pseudospectral(LGL)";
    out.grid_label = "N=" + std::to_string(n_nodes);

    Opti opti;
    Pseudospectral ps(opti);
    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    PseudospectralOptions opts;
    opts.scheme = PseudospectralScheme::LGL;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = ps.setup(3, 1, 0.0, T, opts);
    apply_brachistochrone_initial_guess(opti, x, u, tau);
    ps.set_dynamics(brachistochrone_ode);
    ps.add_dynamics_constraints();
    apply_brachistochrone_constraints(opti, x, u, ps);
    opti.minimize(T);

    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto sol = opti.solve({.max_iter = 500, .verbose = false});
        const auto t1 = std::chrono::steady_clock::now();

        out.success = true;
        out.t_opt = sol.value(T);
        out.abs_error = std::abs(out.t_opt - ref_time);
        out.n_vars = static_cast<int>(opti.casadi_opti().x().numel());
        out.n_constraints = static_cast<int>(opti.casadi_opti().g().numel());
        out.n_iters = static_cast<int>(sol.stats().at("iter_count"));
        out.solve_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    } catch (const std::exception &e) {
        out.failure_reason = e.what();
    }
    return out;
}

RunResult solve_birkhoff(int n_nodes, double ref_time) {
    RunResult out;
    out.method = "BirkhoffPS(LGL)";
    out.grid_label = "N=" + std::to_string(n_nodes);

    Opti opti;
    BirkhoffPseudospectral bk(opti);
    auto T = opti.variable(2.0, std::nullopt, 0.1, 10.0);

    BirkhoffPseudospectralOptions opts;
    opts.scheme = BirkhoffScheme::LGL;
    opts.n_nodes = n_nodes;

    auto [x, u, tau] = bk.setup(3, 1, 0.0, T, opts);
    apply_brachistochrone_initial_guess(opti, x, u, tau);
    bk.set_dynamics(brachistochrone_ode);
    bk.add_dynamics_constraints();
    apply_brachistochrone_constraints(opti, x, u, bk);
    opti.minimize(T);

    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto sol = opti.solve({.max_iter = 500, .verbose = false});
        const auto t1 = std::chrono::steady_clock::now();

        out.success = true;
        out.t_opt = sol.value(T);
        out.abs_error = std::abs(out.t_opt - ref_time);
        out.n_vars = static_cast<int>(opti.casadi_opti().x().numel());
        out.n_constraints = static_cast<int>(opti.casadi_opti().g().numel());
        out.n_iters = static_cast<int>(sol.stats().at("iter_count"));
        out.solve_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    } catch (const std::exception &e) {
        out.failure_reason = e.what();
    }
    return out;
}

void print_summary(const std::vector<RunResult> &results) {
    std::cout << std::left << std::setw(24) << "Method" << std::setw(8) << "Grid" << std::setw(12)
              << "T*[s]" << std::setw(14) << "|T*-Ref|[s]" << std::setw(8) << "Iters"
              << std::setw(10) << "Vars" << std::setw(12) << "Constr" << std::setw(12)
              << "Solve[ms]" << "Status\n";
    std::cout << std::string(104, '-') << "\n";

    for (const auto &r : results) {
        std::cout << std::left << std::setw(24) << r.method << std::setw(8) << r.grid_label;
        if (r.success) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(5) << r.t_opt
                      << std::setw(14) << std::scientific << std::setprecision(3) << r.abs_error
                      << std::defaultfloat << std::setw(8) << r.n_iters << std::setw(10) << r.n_vars
                      << std::setw(12) << r.n_constraints << std::setw(12) << std::fixed
                      << std::setprecision(2) << r.solve_ms << std::defaultfloat << "OK\n";
        } else {
            std::cout << std::setw(12) << "-" << std::setw(14) << "-" << std::setw(8) << "-"
                      << std::setw(10) << "-" << std::setw(12) << "-" << std::setw(12) << "-"
                      << "FAILED\n";
        }
    }
}

std::string reduction_str(double prev_error, double cur_error) {
    if (prev_error <= 0.0 || cur_error <= 0.0) {
        return "-";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << (prev_error / cur_error) << "x";
    return ss.str();
}

void print_convergence_table(const std::string &title, const std::vector<RunResult> &results) {
    std::cout << "\n[" << title << "]\n";
    std::cout << std::left << std::setw(8) << "Grid" << std::setw(12) << "T*[s]" << std::setw(14)
              << "|T*-Ref|[s]" << std::setw(12) << "Solve[ms]" << std::setw(12) << "ErrDrop"
              << "Status\n";
    std::cout << std::string(68, '-') << "\n";

    double prev_error = -1.0;
    for (const auto &r : results) {
        std::cout << std::left << std::setw(8) << r.grid_label;
        if (r.success) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(5) << r.t_opt
                      << std::setw(14) << std::scientific << std::setprecision(3) << r.abs_error
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.solve_ms
                      << std::setw(12) << reduction_str(prev_error, r.abs_error) << "OK\n";
            prev_error = r.abs_error;
        } else {
            std::cout << std::setw(12) << "-" << std::setw(14) << "-" << std::setw(12) << "-"
                      << std::setw(12) << "-" << "FAILED\n";
        }
    }
}

} // namespace

int main() {
    constexpr double external_ref = 1.80185;

    std::cout << "===============================================================\n";
    std::cout << "  Brachistochrone Transcription Comparison\n";
    std::cout << "  Methods: Direct Collocation, Multiple Shooting, Pseudospectral, BirkhoffPS\n";
    std::cout << "  External Reference: T* = " << external_ref << " s\n";
    std::cout << "===============================================================\n\n";

    std::vector<double> ref_samples;
    const RunResult ref_dc = solve_collocation(61, external_ref);
    const RunResult ref_ms = solve_multishoot(40, external_ref);
    const RunResult ref_ps = solve_pseudospectral(61, external_ref);
    const RunResult ref_bk = solve_birkhoff(61, external_ref);

    if (ref_dc.success)
        ref_samples.push_back(ref_dc.t_opt);
    if (ref_ms.success)
        ref_samples.push_back(ref_ms.t_opt);
    if (ref_ps.success)
        ref_samples.push_back(ref_ps.t_opt);
    if (ref_bk.success)
        ref_samples.push_back(ref_bk.t_opt);

    double internal_ref = external_ref;
    if (!ref_samples.empty()) {
        double sum = 0.0;
        for (double v : ref_samples)
            sum += v;
        internal_ref = sum / static_cast<double>(ref_samples.size());
    }

    std::cout << "Internal reference (high-resolution mean): " << std::fixed << std::setprecision(6)
              << internal_ref << " s\n\n";

    std::vector<RunResult> results;
    results.push_back(solve_collocation(31, internal_ref));
    results.push_back(solve_multishoot(20, internal_ref));
    results.push_back(solve_pseudospectral(31, internal_ref));
    results.push_back(solve_birkhoff(31, internal_ref));

    std::cout << "[Single-Grid Performance Snapshot]\n";
    print_summary(results);

    for (const auto &r : results) {
        if (!r.success) {
            std::cout << "\n" << r.method << " failed: " << r.failure_reason << "\n";
        }
    }

    std::cout << "\n[Convergence Study]\n";
    std::cout
        << "Collocation/Pseudospectral/Birkhoff use nodes; MultipleShooting uses intervals.\n";

    const std::vector<int> collocation_nodes = {5, 7, 9, 11, 15};
    const std::vector<int> multishoot_intervals = {4, 6, 8, 12, 16};
    const std::vector<int> pseudospectral_nodes = {5, 7, 9, 11, 15};
    const std::vector<int> birkhoff_nodes = {5, 7, 9, 11, 15};

    std::vector<RunResult> collocation_sweep;
    std::vector<RunResult> multishoot_sweep;
    std::vector<RunResult> pseudospectral_sweep;
    std::vector<RunResult> birkhoff_sweep;

    for (int n : collocation_nodes) {
        collocation_sweep.push_back(solve_collocation(n, internal_ref));
    }
    for (int n : multishoot_intervals) {
        multishoot_sweep.push_back(solve_multishoot(n, internal_ref));
    }
    for (int n : pseudospectral_nodes) {
        pseudospectral_sweep.push_back(solve_pseudospectral(n, internal_ref));
    }
    for (int n : birkhoff_nodes) {
        birkhoff_sweep.push_back(solve_birkhoff(n, internal_ref));
    }

    print_convergence_table("DirectCollocation (Hermite-Simpson)", collocation_sweep);
    print_convergence_table("MultipleShooting (CVODES)", multishoot_sweep);
    print_convergence_table("Pseudospectral (LGL)", pseudospectral_sweep);
    print_convergence_table("BirkhoffPseudospectral (LGL)", birkhoff_sweep);

    return 0;
}
