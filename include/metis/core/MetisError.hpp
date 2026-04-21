#pragma once
/**
 * @file MetisError.hpp
 * @brief Custom exception hierarchy for Metis framework
 *
 * Provides consistent error handling with contextual messages.
 * All exceptions derive from std::runtime_error for backward compatibility.
 */

#include <stdexcept>
#include <string>

namespace metis {

/**
 * @brief Base exception for all Metis errors
 * @see InvalidArgument, RuntimeError, InterpolationError, IntegrationError
 */
class MetisError : public std::runtime_error {
  public:
    /**
     * @brief Construct with a descriptive message
     * @param what Error description (automatically prefixed with "[metis]")
     */
    explicit MetisError(const std::string &what) : std::runtime_error("[metis] " + what) {}
};

/**
 * @brief Input validation failed (e.g., mismatched sizes, invalid parameters)
 */
class InvalidArgument : public MetisError {
  public:
    /// @brief Construct with a descriptive message
    /// @param what Error description
    explicit InvalidArgument(const std::string &what) : MetisError(what) {}
};

/**
 * @brief Operation failed at runtime (e.g., CasADi eval with free variables)
 */
class RuntimeError : public MetisError {
  public:
    /// @brief Construct with a descriptive message
    /// @param what Error description
    explicit RuntimeError(const std::string &what) : MetisError(what) {}
};

/**
 * @brief Interpolation-specific errors
 */
class InterpolationError : public MetisError {
  public:
    /// @brief Construct with a descriptive message
    /// @param what Error description (automatically prefixed with "Interpolation:")
    explicit InterpolationError(const std::string &what) : MetisError("Interpolation: " + what) {}
};

/**
 * @brief Integration/ODE solver errors
 */
class IntegrationError : public MetisError {
  public:
    /// @brief Construct with a descriptive message
    /// @param what Error description (automatically prefixed with "Integration:")
    explicit IntegrationError(const std::string &what) : MetisError("Integration: " + what) {}
};

} // namespace metis
