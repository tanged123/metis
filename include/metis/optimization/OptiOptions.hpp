/**
 * @file OptiOptions.hpp
 * @brief Solver selection and configuration for Opti
 */

#pragma once

#include <casadi/casadi.hpp>
#include <string>

namespace metis {

/**
 * @brief Available NLP solvers
 *
 * Solvers are provided via CasADi's nlpsol interface.
 * IPOPT is always available. Others require separate installation.
 *
 * @see OptiOptions for solver configuration
 */
enum class Solver {
    Ipopt,  ///< Interior Point OPTimizer (default, always available)
    Snopt,  ///< Sparse Nonlinear OPTimizer (requires license)
    QpOases ///< QP solver for QP subproblems
};

/** @brief Check if a solver is available in the current CasADi build
 *  @param solver Solver to check
 *  @return true if solver plugin is loaded and usable */
inline bool solver_available(Solver solver);

/**
 * @brief Get the CasADi solver name string
 *
 * @param solver Solver enum value
 * @return Solver name as expected by CasADi nlpsol
 */
inline const char *solver_name(Solver solver) {
    switch (solver) {
    case Solver::Snopt:
        return "snopt";
    case Solver::QpOases:
        return "qpoases";
    case Solver::Ipopt:
    default:
        return "ipopt";
    }
}

/**
 * @brief Check if a solver is available in the current CasADi build
 *
 * Uses CasADi's nlpsol plugin system to check availability.
 *
 * @param solver Solver to check
 * @return true if solver plugin is loaded and usable
 */
inline bool solver_available(Solver solver) { return casadi::has_nlpsol(solver_name(solver)); }

/**
 * @brief SNOPT-specific solver options
 *
 * These map to SNOPT's solver parameters.
 * See SNOPT documentation for detailed descriptions.
 */
struct SNOPTOptions {
    int major_iterations_limit = 1000;         ///< Max major (outer) iterations
    int minor_iterations_limit = 500;          ///< Max minor (QP) iterations per major
    double major_optimality_tolerance = 1e-6;  ///< Optimality tolerance
    double major_feasibility_tolerance = 1e-6; ///< Constraint feasibility tolerance
    int print_level = 0;                       ///< 0=silent, 1=summary, 2+=detailed

    /** @brief Set maximum major iterations
     *  @param v iteration limit
     *  @return reference to this for chaining */
    SNOPTOptions &set_major_iterations_limit(int v) {
        major_iterations_limit = v;
        return *this;
    }
    /** @brief Set maximum minor iterations per major
     *  @param v iteration limit
     *  @return reference to this for chaining */
    SNOPTOptions &set_minor_iterations_limit(int v) {
        minor_iterations_limit = v;
        return *this;
    }
    /** @brief Set major optimality tolerance
     *  @param v tolerance value
     *  @return reference to this for chaining */
    SNOPTOptions &set_major_optimality_tolerance(double v) {
        major_optimality_tolerance = v;
        return *this;
    }
    /** @brief Set major feasibility tolerance
     *  @param v tolerance value
     *  @return reference to this for chaining */
    SNOPTOptions &set_major_feasibility_tolerance(double v) {
        major_feasibility_tolerance = v;
        return *this;
    }
    /** @brief Set print verbosity level
     *  @param v print level (0=silent, 1=summary, 2+=detailed)
     *  @return reference to this for chaining */
    SNOPTOptions &set_print_level(int v) {
        print_level = v;
        return *this;
    }
};

/**
 * @brief Options for solving optimization problems
 *
 * Configures solver behavior. Most users can use defaults.
 *
 * Usage (designated initializers - must follow declaration order):
 *   opti.solve({.max_iter = 500, .verbose = false});
 *
 * Usage (builder pattern - any order):
 *   opti.solve(OptiOptions{}.set_verbose(false).set_max_iter(500));
 *
 * Usage (alternative solver):
 *   opti.solve({.solver = metis::Solver::Snopt});
 *
 * @see Opti::solve for usage
 */
struct OptiOptions {
    // Solver selection
    Solver solver = Solver::Ipopt; ///< NLP solver to use
    SNOPTOptions snopt_opts;       ///< SNOPT-specific options (if solver=SNOPT)

    // General solver options (used by all solvers where applicable)
    int max_iter = 1000;                  ///< Maximum iterations
    double max_cpu_time = 1e20;           ///< Maximum solve time [seconds]
    double tol = 1e-8;                    ///< Convergence tolerance
    bool verbose = true;                  ///< Print solver progress
    bool jit = false;                     ///< JIT compile expressions (experimental)
    bool detect_simple_bounds = false;    ///< Detect simple variable bounds
    std::string mu_strategy = "adaptive"; ///< Barrier parameter strategy (IPOPT)

    /** @brief Set maximum iterations
     *  @param v iteration limit
     *  @return reference to this for chaining */
    OptiOptions &set_max_iter(int v) {
        max_iter = v;
        return *this;
    }
    /** @brief Set maximum CPU time
     *  @param v time limit in seconds
     *  @return reference to this for chaining */
    OptiOptions &set_max_cpu_time(double v) {
        max_cpu_time = v;
        return *this;
    }
    /** @brief Set convergence tolerance
     *  @param v tolerance value
     *  @return reference to this for chaining */
    OptiOptions &set_tol(double v) {
        tol = v;
        return *this;
    }
    /** @brief Set verbosity
     *  @param v true to print solver progress
     *  @return reference to this for chaining */
    OptiOptions &set_verbose(bool v) {
        verbose = v;
        return *this;
    }
    /** @brief Enable or disable JIT compilation
     *  @param v true to enable JIT (experimental)
     *  @return reference to this for chaining */
    OptiOptions &set_jit(bool v) {
        jit = v;
        return *this;
    }
    /** @brief Enable simple bound detection
     *  @param v true to enable
     *  @return reference to this for chaining */
    OptiOptions &set_detect_simple_bounds(bool v) {
        detect_simple_bounds = v;
        return *this;
    }
    /** @brief Set IPOPT barrier parameter strategy
     *  @param v strategy name (e.g. "adaptive", "monotone")
     *  @return reference to this for chaining */
    OptiOptions &set_mu_strategy(const std::string &v) {
        mu_strategy = v;
        return *this;
    }
    /** @brief Set the NLP solver
     *  @param v solver enum value
     *  @return reference to this for chaining */
    OptiOptions &set_solver(Solver v) {
        solver = v;
        return *this;
    }
    /** @brief Set SNOPT-specific options
     *  @param v SNOPT options struct
     *  @return reference to this for chaining */
    OptiOptions &set_snopt_opts(const SNOPTOptions &v) {
        snopt_opts = v;
        return *this;
    }
};

} // namespace metis
