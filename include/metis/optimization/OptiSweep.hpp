/// @file OptiSweep.hpp
/// @brief Parametric sweep results for optimization problems

#pragma once

#include "OptiSol.hpp"
#include "metis/core/MetisError.hpp"
#include <optional>
#include <vector>

namespace metis {

/// @brief Result of a parametric sweep
///
/// Stores solutions obtained by solving an optimization problem
/// across a range of parameter values. Each entry in @c solutions
/// corresponds 1:1 with @c param_values; failed solves are represented
/// as @c std::nullopt.
///
/// @code
/// auto result = opti.solve_sweep(param, {0.1, 0.2, 0.3, 0.4, 0.5});
/// for (int i = 0; i < result.size(); ++i) {
///     if (result.converged[i]) {
///         std::cout << "param=" << result.param_values[i]
///                   << " obj=" << result.objective(i, obj_expr) << "\n";
///     }
/// }
/// @endcode
///
/// @see Opti::solve_sweep for creating a SweepResult
/// @see OptiSol for individual solution access
struct SweepResult {
    /// Parameter values that were swept
    std::vector<double> param_values;

    /// Solutions at each parameter value (nullopt for failed solves)
    std::vector<std::optional<OptiSol>> solutions;

    /// Number of iterations for each solve
    std::vector<int> iterations;

    /// Whether all solves converged successfully
    bool all_converged = true;

    /// Per-point convergence status (true = converged)
    std::vector<bool> converged;

    /// Error messages for failed points (empty string if converged)
    std::vector<std::string> errors;

    /// @brief Number of sweep points
    /// @return number of parameter values in the sweep
    size_t size() const { return param_values.size(); }

    /// @brief Get objective value at sweep index
    /// @param i sweep index
    /// @param objective_expr objective expression passed to minimize/maximize
    /// @return optimized objective value at sweep point i
    /// @throws InvalidArgument if index is out of range or point did not converge
    double objective(size_t i, const SymbolicScalar &objective_expr) const {
        if (i >= solutions.size()) {
            throw InvalidArgument("SweepResult::objective: index out of range");
        }
        if (!solutions[i].has_value()) {
            throw InvalidArgument("SweepResult::objective: solve did not converge at index " +
                                  std::to_string(i));
        }
        return solutions[i]->value(objective_expr);
    }

    /// @brief Extract scalar variable values across converged sweep points
    /// @param var symbolic scalar to evaluate
    /// @return vector of values, one per converged sweep point
    NumericVector values(const SymbolicScalar &var) const {
        // Count converged points
        int n_converged = 0;
        for (const auto &sol : solutions) {
            if (sol.has_value()) {
                ++n_converged;
            }
        }
        NumericVector result(n_converged);
        int j = 0;
        for (const auto &sol : solutions) {
            if (sol.has_value()) {
                result(j++) = sol->value(var);
            }
        }
        return result;
    }

    /// @brief Extract vector variable values across converged sweep points
    /// @param var symbolic vector to evaluate
    /// @return matrix with columns corresponding to converged sweep points
    NumericMatrix values(const SymbolicVector &var) const {
        // Collect converged solutions
        std::vector<size_t> converged_indices;
        for (size_t i = 0; i < solutions.size(); ++i) {
            if (solutions[i].has_value()) {
                converged_indices.push_back(i);
            }
        }
        if (converged_indices.empty()) {
            return NumericMatrix(0, 0);
        }
        auto first = solutions[converged_indices[0]]->value(var);
        NumericMatrix result(first.size(), static_cast<int>(converged_indices.size()));
        result.col(0) = first;
        for (size_t j = 1; j < converged_indices.size(); ++j) {
            result.col(static_cast<int>(j)) = solutions[converged_indices[j]]->value(var);
        }
        return result;
    }
};

} // namespace metis
