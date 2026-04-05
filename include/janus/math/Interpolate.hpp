#pragma once
/**
 * @file Interpolate.hpp
 * @brief N-dimensional interpolation (linear, Hermite, B-spline, nearest)
 * @see ScatteredInterpolator.hpp
 */

#include "janus/core/JanusConcepts.hpp"
#include "janus/core/JanusError.hpp"
#include "janus/core/JanusTypes.hpp"
#include "janus/math/Linalg.hpp"
#include <algorithm>
#include <casadi/casadi.hpp>
#include <cmath>
#include <optional>
#include <vector>

namespace janus {

// ============================================================================
// Interpolation Method Enum
// ============================================================================

/**
 * @brief Supported interpolation methods
 *
 * Controls the interpolation algorithm used for both 1D and N-D interpolation.
 */
enum class InterpolationMethod {
    Linear,  ///< Piecewise linear (C0 continuous) - fast, simple
    Hermite, ///< Cubic Hermite/Catmull-Rom (C1 continuous) - smooth gradients
    BSpline, ///< Cubic B-spline (C2 continuous) - smoothest, good for optimization
    Nearest  ///< Nearest neighbor - fast, non-differentiable
};

// ============================================================================
// Extrapolation Mode Enum
// ============================================================================

/**
 * @brief Controls behavior when query points fall outside grid bounds
 */
enum class ExtrapolationMode {
    Clamp, ///< Clamp queries to grid bounds (default, safe)
    Linear ///< Linear extrapolation from boundary slope (1D only)
};

// ============================================================================
// Extrapolation Configuration
// ============================================================================

/**
 * @brief Configuration for extrapolation behavior
 *
 * Use the factory methods for convenient construction:
 *   ExtrapolationConfig::clamp()           - Safe clamping (default)
 *   ExtrapolationConfig::linear()          - Linear extrapolation, unbounded
 *   ExtrapolationConfig::linear(-10, 100)  - Linear with output bounds
 */
struct ExtrapolationConfig {
    ExtrapolationMode mode = ExtrapolationMode::Clamp;

    /// Safety bounds applied to output values (nullopt = unbounded)
    std::optional<double> lower_bound = std::nullopt;
    std::optional<double> upper_bound = std::nullopt;

    /// Factory: create clamp config (default, safe)
    static ExtrapolationConfig clamp() { return {}; }

    /// Factory: create linear extrapolation config with optional bounds
    static ExtrapolationConfig linear(std::optional<double> lower = std::nullopt,
                                      std::optional<double> upper = std::nullopt) {
        return {ExtrapolationMode::Linear, lower, upper};
    }
};

// ============================================================================
// Implementation Details
// ============================================================================

namespace detail {

/**
 * @brief Convert InterpolationMethod enum to CasADi string
 */
inline std::string method_to_casadi_string(InterpolationMethod method) {
    switch (method) {
    case InterpolationMethod::Linear:
        return "linear";
    case InterpolationMethod::BSpline:
        return "bspline";
    case InterpolationMethod::Hermite:
    case InterpolationMethod::Nearest:
        // These don't have native CasADi support, handled separately
        return "linear"; // Fallback for internal use
    default:
        return "linear";
    }
}

inline const char *hermite_symbolic_error_message() {
    return "Interpolator: Hermite/Catmull-Rom is numeric-only because interval selection "
           "requires runtime comparisons. Use BSpline for symbolic or optimization queries.";
}

/**
 * @brief Flatten N-D values array in Fortran order for CasADi
 *
 * CasADi expects values flattened in column-major (Fortran) order.
 * For a 2D array with shape (m, n), this means iterating over
 * the first dimension fastest.
 */
template <typename Derived>
inline JanusVector<typename Derived::Scalar>
flatten_fortran_order(const Eigen::MatrixBase<Derived> &values) {
    using Scalar = typename Derived::Scalar;
    JanusVector<Scalar> flattened(values.size());

    Eigen::Index idx = 0;
    for (Eigen::Index j = 0; j < values.cols(); ++j) {
        for (Eigen::Index i = 0; i < values.rows(); ++i) {
            flattened(idx++) = values(i, j);
        }
    }

    return flattened;
}

inline std::string symbolic_values_method_error_message(InterpolationMethod method) {
    switch (method) {
    case InterpolationMethod::Hermite:
        return "interpn: symbolic table values are not supported for Hermite/Catmull-Rom "
               "because interval selection remains numeric-only. Use BSpline for symbolic or "
               "optimization queries.";
    case InterpolationMethod::Nearest:
        return "interpn: symbolic table values are not supported for Nearest because the "
               "method is non-differentiable. Use Linear or BSpline.";
    default:
        return "interpn: symbolic table values are only supported for Linear and BSpline.";
    }
}

struct InterpnGridData {
    std::vector<std::vector<double>> grid;
    int expected_size = 1;
};

inline InterpnGridData build_interpn_grid(const std::vector<NumericVector> &points,
                                          const char *context) {
    if (points.empty()) {
        throw InterpolationError(std::string(context) + ": points cannot be empty");
    }

    const int n_dims = static_cast<int>(points.size());
    InterpnGridData data;
    data.grid.resize(n_dims);

    for (int d = 0; d < n_dims; ++d) {
        if (points[d].size() < 2) {
            throw InterpolationError(std::string(context) + ": points[" + std::to_string(d) +
                                     "] must have at least 2 points");
        }

        data.grid[d].resize(points[d].size());
        Eigen::VectorXd::Map(data.grid[d].data(), points[d].size()) = points[d];

        for (size_t i = 0; i + 1 < data.grid[d].size(); ++i) {
            if (data.grid[d][i + 1] <= data.grid[d][i]) {
                throw InterpolationError(std::string(context) + ": points[" + std::to_string(d) +
                                         "] must be strictly increasing");
            }
        }

        data.expected_size *= static_cast<int>(points[d].size());
    }

    return data;
}

inline void validate_bspline_grid(const std::vector<std::vector<double>> &grid,
                                  const char *context) {
    for (size_t d = 0; d < grid.size(); ++d) {
        if (grid[d].size() < 4) {
            throw InterpolationError(std::string(context) +
                                     ": BSpline requires at least 4 grid points in dimension " +
                                     std::to_string(d));
        }
    }
}

template <JanusScalar Scalar>
inline JanusMatrix<Scalar> normalize_query_matrix(const JanusMatrix<Scalar> &xi, int n_dims,
                                                  const char *context) {
    if (xi.cols() != n_dims && xi.rows() != n_dims) {
        throw InterpolationError(std::string(context) +
                                 ": xi must have shape (n_points, n_dims) or (n_dims, n_points)");
    }

    const bool need_transpose = (xi.rows() == n_dims && xi.cols() != n_dims);
    if (need_transpose) {
        return xi.transpose().eval();
    }
    return xi.eval();
}

template <JanusScalar Scalar>
inline SymbolicScalar clamp_query_point(const JanusVector<Scalar> &point,
                                        const std::vector<std::vector<double>> &grid) {
    SymbolicScalar clamped(point.size(), 1);
    for (int d = 0; d < point.size(); ++d) {
        SymbolicScalar value =
            std::is_same_v<Scalar, SymbolicScalar> ? point(d) : SymbolicScalar(point(d));
        value = SymbolicScalar::fmax(value, grid[d].front());
        value = SymbolicScalar::fmin(value, grid[d].back());
        clamped(d) = value;
    }
    return clamped;
}

template <JanusScalar Scalar>
inline bool point_out_of_bounds_numeric(const JanusVector<Scalar> &point,
                                        const std::vector<std::vector<double>> &grid) {
    static_assert(std::is_floating_point_v<Scalar>);
    for (int d = 0; d < point.size(); ++d) {
        if (point(d) < grid[d].front() || point(d) > grid[d].back()) {
            return true;
        }
    }
    return false;
}

inline SymbolicScalar point_out_of_bounds_symbolic(const SymbolicVector &point,
                                                   const std::vector<std::vector<double>> &grid) {
    SymbolicScalar out_of_bounds = SymbolicScalar(0);
    for (int d = 0; d < point.size(); ++d) {
        out_of_bounds = SymbolicScalar::logic_or(out_of_bounds, point(d) < grid[d].front());
        out_of_bounds = SymbolicScalar::logic_or(out_of_bounds, point(d) > grid[d].back());
    }
    return out_of_bounds;
}

inline casadi::Function make_parametric_interpolant(const std::vector<std::vector<double>> &grid,
                                                    InterpolationMethod method) {
    if (method == InterpolationMethod::Hermite || method == InterpolationMethod::Nearest) {
        throw InterpolationError(symbolic_values_method_error_message(method));
    }
    if (method == InterpolationMethod::BSpline) {
        validate_bspline_grid(grid, "interpn");
    }

    casadi::Dict opts;
    opts["inline"] = true;
    return casadi::interpolant("interp_parametric", method_to_casadi_string(method), grid, 1, opts);
}

// ============================================================================
// Catmull-Rom (C1 Hermite) Interpolation Helpers
// ============================================================================

/**
 * @brief Compute Catmull-Rom slope at a point using neighboring values
 *
 * For interior points: m = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
 * For endpoints: use one-sided differences
 *
 * @param x Grid x-coordinates
 * @param y Grid y-values
 * @param i Index of point
 * @return Slope at point i
 */
inline double catmull_rom_slope(const std::vector<double> &x, const std::vector<double> &y,
                                size_t i) {
    size_t n = x.size();
    if (n < 2)
        return 0.0;

    if (i == 0) {
        // Left endpoint: forward difference
        return (y[1] - y[0]) / (x[1] - x[0]);
    } else if (i == n - 1) {
        // Right endpoint: backward difference
        return (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    } else {
        // Interior: central difference using neighbors
        return (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    }
}

/**
 * @brief Evaluate 1D Catmull-Rom cubic Hermite spline (numeric only)
 *
 * Uses cubic Hermite basis functions:
 * h00(t) = 2t³ - 3t² + 1
 * h10(t) = t³ - 2t² + t
 * h01(t) = -2t³ + 3t²
 * h11(t) = t³ - t²
 *
 * p(t) = h00(t)*p0 + h10(t)*m0*h + h01(t)*p1 + h11(t)*m1*h
 * where h = x1 - x0, t = (x - x0) / h
 *
 * @param x Grid x-coordinates (sorted)
 * @param y Grid y-values
 * @param query Query point (must be numeric)
 * @return Interpolated value
 */
inline double hermite_interp_1d_numeric(const std::vector<double> &x, const std::vector<double> &y,
                                        double query) {
    size_t n = x.size();
    if (n < 2) {
        return y[0];
    }

    // Clamp query to grid bounds
    query = std::max(query, x.front());
    query = std::min(query, x.back());

    // Find interval containing query using binary search
    auto it = std::upper_bound(x.begin(), x.end(), query);
    size_t idx = 0;
    if (it == x.begin()) {
        idx = 0;
    } else if (it == x.end()) {
        idx = n - 2;
    } else {
        idx = static_cast<size_t>(std::distance(x.begin(), it)) - 1;
    }
    if (idx >= n - 1)
        idx = n - 2; // Safety clamp

    // Get interval endpoints
    double x0 = x[idx];
    double x1 = x[idx + 1];
    double y0 = y[idx];
    double y1 = y[idx + 1];
    double h = x1 - x0;

    // Compute slopes at endpoints (Catmull-Rom)
    double m0 = catmull_rom_slope(x, y, idx);
    double m1 = catmull_rom_slope(x, y, idx + 1);

    // Normalized coordinate
    double t = (query - x0) / h;
    t = std::max(0.0, std::min(1.0, t)); // Clamp to [0, 1]

    double t2 = t * t;
    double t3 = t2 * t;

    // Hermite basis functions
    double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    double h10 = t3 - 2.0 * t2 + t;
    double h01 = -2.0 * t3 + 3.0 * t2;
    double h11 = t3 - t2;

    // Interpolated value
    return h00 * y0 + h10 * (m0 * h) + h01 * y1 + h11 * (m1 * h);
}

/**
 * @brief Compute multi-dimensional index from flat index (Fortran order)
 */
inline std::vector<int> flat_to_multi_index(int flat_idx, const std::vector<int> &dims) {
    std::vector<int> multi(dims.size());
    int remaining = flat_idx;
    for (size_t d = 0; d < dims.size(); ++d) {
        multi[d] = remaining % dims[d];
        remaining /= dims[d];
    }
    return multi;
}

/**
 * @brief Compute flat index from multi-dimensional index (Fortran order)
 */
inline int multi_to_flat_index(const std::vector<int> &multi, const std::vector<int> &dims) {
    int flat = 0;
    int stride = 1;
    for (size_t d = 0; d < dims.size(); ++d) {
        flat += multi[d] * stride;
        stride *= dims[d];
    }
    return flat;
}

/**
 * @brief N-dimensional Catmull-Rom interpolation via tensor product (numeric only)
 *
 * Interpolates in each dimension sequentially using 1D Hermite splines.
 * This function is numeric-only because interval finding requires comparisons.
 */
inline double hermite_interpn_numeric(const std::vector<std::vector<double>> &grid,
                                      const std::vector<double> &values,
                                      const std::vector<double> &query) {
    int n_dims = static_cast<int>(grid.size());

    // Build dimension sizes
    std::vector<int> dims(n_dims);
    for (int d = 0; d < n_dims; ++d) {
        dims[d] = static_cast<int>(grid[d].size());
    }

    // For N-D interpolation, we use recursive tensor product:
    // 1. Interpolate along first dimension to reduce to (n-1)-D
    // 2. Repeat until we have a scalar

    // Start with the full values array
    std::vector<double> current_values = values;
    std::vector<int> current_dims = dims;

    for (int d = 0; d < n_dims; ++d) {
        const std::vector<double> &x = grid[d];
        int n_x = current_dims[0]; // Size in first (current) dimension

        // Compute size of remaining dimensions
        int remaining_size = 1;
        for (size_t dd = 1; dd < current_dims.size(); ++dd) {
            remaining_size *= current_dims[dd];
        }

        // Interpolate along first dimension for each slice
        std::vector<double> new_values(remaining_size);

        for (int slice = 0; slice < remaining_size; ++slice) {
            // Extract 1D slice along first dimension
            std::vector<double> y_slice(n_x);
            for (int i = 0; i < n_x; ++i) {
                int flat_idx = i + slice * n_x;
                y_slice[i] = current_values[flat_idx];
            }

            // Interpolate this slice
            new_values[slice] = hermite_interp_1d_numeric(x, y_slice, query[d]);
        }

        // Update for next dimension
        current_values = std::move(new_values);
        current_dims.erase(current_dims.begin());
    }

    return current_values[0];
}

} // namespace detail

// ============================================================================
// Unified Interpolator Class
// ============================================================================

/**
 * @brief Unified N-dimensional interpolator
 *
 * A single class that handles interpolation in any number of dimensions.
 * The 1D case is simply N=1. Supports both numeric (double) and symbolic
 * (casadi::MX) evaluation via the same interface.
 *
 * For numeric queries, all methods (Linear, Hermite, BSpline, Nearest) are supported.
 * For symbolic queries, only Linear and BSpline are supported. Hermite requires
 * runtime interval selection, so symbolic Hermite queries throw InterpolationError
 * with guidance to use BSpline instead.
 *
 * @example 1D usage:
 * ```cpp
 * NumericVector x(4), y(4);
 * x << 0, 1, 2, 3;
 * y << 0, 1, 4, 9;  // y = x^2
 *
 * janus::Interpolator interp(x, y, janus::InterpolationMethod::Linear);
 *
 * double result = interp(1.5);  // Query at single point
 * NumericVector batch = interp(query_vec);  // Batch query
 * ```
 *
 * @example N-D usage:
 * ```cpp
 * std::vector<NumericVector> grid = {x_pts, y_pts};
 * NumericVector values(4);  // Fortran order
 * values << 0, 1, 1, 2;     // z(x,y) = x + y
 *
 * janus::Interpolator interp(grid, values);
 *
 * NumericVector query(2);
 * query << 0.5, 0.5;
 * double result = interp(query);  // Query at (0.5, 0.5)
 * ```
 *
 * @note `Interpolator` caches numeric table data. For symbolic table values
 *       that must remain optimization variables, use the free `interpn()`
 *       overloads instead.
 */
class Interpolator {
  private:
    std::vector<std::vector<double>> m_grid;
    std::vector<double> m_values;
    casadi::Function m_casadi_fn;
    InterpolationMethod m_method = InterpolationMethod::Linear;
    int m_dims = 0;
    bool m_valid = false;
    bool m_use_custom_hermite = false;

    // Extrapolation configuration
    ExtrapolationConfig m_extrap_config;

    // Precomputed boundary partial derivatives for linear extrapolation
    // For each dimension d: slope of f w.r.t. x_d at left/right boundary
    std::vector<double> m_slopes_left;  // ∂f/∂x_d at x_d = x_d_min (per dimension)
    std::vector<double> m_slopes_right; // ∂f/∂x_d at x_d = x_d_max (per dimension)

  public:
    /**
     * @brief Default constructor (invalid state)
     */
    Interpolator() = default;

    /**
     * @brief Construct N-dimensional interpolator
     *
     * @param points Vector of 1D coordinate arrays for each dimension
     * @param values Flattened values in Fortran order (column-major)
     * @param method Interpolation method (default: Linear)
     * @throw InterpolationError if inputs are invalid
     */
    Interpolator(const std::vector<NumericVector> &points, const NumericVector &values,
                 InterpolationMethod method = InterpolationMethod::Linear) {
        if (points.empty()) {
            throw InterpolationError("Interpolator: points cannot be empty");
        }

        m_dims = static_cast<int>(points.size());
        m_method = method;

        // Convert to std::vector for internal storage
        m_grid.resize(m_dims);
        int expected_size = 1;
        for (int d = 0; d < m_dims; ++d) {
            if (points[d].size() < 2) {
                throw InterpolationError("Interpolator: need at least 2 grid points in dimension " +
                                         std::to_string(d));
            }
            m_grid[d].resize(points[d].size());
            Eigen::VectorXd::Map(m_grid[d].data(), points[d].size()) = points[d];

            if (!std::is_sorted(m_grid[d].begin(), m_grid[d].end())) {
                throw InterpolationError("Interpolator: points[" + std::to_string(d) +
                                         "] must be sorted");
            }
            expected_size *= static_cast<int>(points[d].size());
        }

        // Validate values size
        if (values.size() != expected_size) {
            throw InterpolationError("Interpolator: values size mismatch. Expected " +
                                     std::to_string(expected_size) + ", got " +
                                     std::to_string(values.size()));
        }

        m_values.resize(values.size());
        Eigen::VectorXd::Map(m_values.data(), values.size()) = values;

        // Method-specific setup
        setup_method(method);
    }

    /**
     * @brief Construct 1D interpolator (convenience overload)
     *
     * This is syntactic sugar for the N-D constructor with N=1.
     *
     * @param x Grid points (must be sorted)
     * @param y Function values at grid points
     * @param method Interpolation method (default: Linear)
     * @throw InterpolationError if inputs are invalid
     */
    Interpolator(const NumericVector &x, const NumericVector &y,
                 InterpolationMethod method = InterpolationMethod::Linear) {
        if (x.size() != y.size()) {
            throw InterpolationError("Interpolator: x and y must have same size");
        }
        if (x.size() < 2) {
            throw InterpolationError("Interpolator: need at least 2 grid points");
        }

        m_dims = 1;
        m_method = method;

        // Store grid and values
        m_grid.resize(1);
        m_grid[0].resize(x.size());
        Eigen::VectorXd::Map(m_grid[0].data(), x.size()) = x;

        if (!std::is_sorted(m_grid[0].begin(), m_grid[0].end())) {
            throw InterpolationError("Interpolator: x grid must be sorted");
        }

        m_values.resize(y.size());
        Eigen::VectorXd::Map(m_values.data(), y.size()) = y;

        // Method-specific setup
        setup_method(method);
    }

    /**
     * @brief Construct N-dimensional interpolator with extrapolation config
     *
     * @param points Vector of 1D coordinate arrays for each dimension
     * @param values Flattened values in Fortran order (column-major)
     * @param method Interpolation method
     * @param extrap Extrapolation configuration
     * @throw InterpolationError if inputs are invalid
     */
    Interpolator(const std::vector<NumericVector> &points, const NumericVector &values,
                 InterpolationMethod method, ExtrapolationConfig extrap)
        : Interpolator(points, values, method) {
        m_extrap_config = extrap;
        compute_boundary_slopes();
    }

    /**
     * @brief Construct 1D interpolator with extrapolation config
     *
     * @param x Grid points (must be sorted)
     * @param y Function values at grid points
     * @param method Interpolation method
     * @param extrap Extrapolation configuration
     * @throw InterpolationError if inputs are invalid
     *
     * @example
     * ```cpp
     * // Linear extrapolation with output bounds [0, 100]
     * Interpolator interp(x, y, InterpolationMethod::BSpline,
     *                     ExtrapolationConfig::linear(0.0, 100.0));
     * ```
     */
    Interpolator(const NumericVector &x, const NumericVector &y, InterpolationMethod method,
                 ExtrapolationConfig extrap)
        : Interpolator(x, y, method) {
        m_extrap_config = extrap;
        compute_boundary_slopes();
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /**
     * @brief Get number of dimensions
     */
    int dims() const { return m_dims; }

    /**
     * @brief Get the interpolation method
     */
    InterpolationMethod method() const { return m_method; }

    /**
     * @brief Check if interpolator is valid (initialized)
     */
    bool valid() const { return m_valid; }

    // ========================================================================
    // Scalar Query (1D only)
    // ========================================================================

    /**
     * @brief Evaluate interpolant at a scalar point (1D only)
     * @tparam Scalar Scalar type (double or SymbolicScalar)
     * @param query Query point
     * @return Interpolated value
     * @throw InterpolationError if not 1D or not initialized
     */
    template <JanusScalar Scalar> Scalar operator()(const Scalar &query) const {
        if (!m_valid)
            throw InterpolationError("Interpolator: not initialized");
        if (m_dims != 1)
            throw InterpolationError("Interpolator: scalar query only valid for 1D");

        if constexpr (std::is_floating_point_v<Scalar>) {
            return eval_numeric_scalar(query);
        } else {
            if (m_use_custom_hermite) {
                throw InterpolationError(detail::hermite_symbolic_error_message());
            }
            return eval_symbolic_scalar(query);
        }
    }

    // ========================================================================
    // N-D Point Query
    // ========================================================================

    /**
     * @brief Evaluate N-D interpolant at a single point
     *
     * For N≥2 interpolators, pass a vector of coordinates.
     * For 1D, use scalar overload or batch query.
     *
     * @tparam Scalar Scalar type (double or SymbolicScalar)
     * @param query Query point (size must match dims())
     * @return Interpolated scalar value
     */
    template <JanusScalar Scalar> Scalar operator()(const JanusVector<Scalar> &query) const {
        if (!m_valid)
            throw InterpolationError("Interpolator: not initialized");

        if (query.size() != m_dims) {
            throw InterpolationError("Interpolator: query size must match dims. Expected " +
                                     std::to_string(m_dims) + ", got " +
                                     std::to_string(query.size()));
        }

        if constexpr (std::is_floating_point_v<Scalar>) {
            return eval_numeric_point(query);
        } else {
            if (m_use_custom_hermite) {
                throw InterpolationError(detail::hermite_symbolic_error_message());
            }
            return eval_symbolic_point(query);
        }
    }

    // ========================================================================
    // Batch Query (Multiple Points)
    // ========================================================================

    /**
     * @brief Evaluate interpolant at multiple points
     *
     * For 1D interpolators, pass a vector of query points.
     * For N-D interpolators, pass (n_points, n_dims) matrix.
     *
     * @tparam Derived Eigen expression type
     * @param queries Query points (vector for 1D, matrix for N-D)
     * @return Vector of interpolated values
     */
    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived> &queries) const
        -> JanusVector<typename Derived::Scalar> {
        using Scalar = typename Derived::Scalar;

        if (!m_valid)
            throw InterpolationError("Interpolator: not initialized");

        int n_points;
        if (m_dims == 1) {
            // 1D: flatten input and treat each element as a query point
            n_points = static_cast<int>(queries.size());
        } else {
            // N-D: determine shape
            bool is_transposed = (queries.rows() == m_dims && queries.cols() != m_dims);
            n_points =
                is_transposed ? static_cast<int>(queries.cols()) : static_cast<int>(queries.rows());
        }

        JanusVector<Scalar> result(n_points);

        if constexpr (std::is_floating_point_v<Scalar>) {
            if (m_dims == 1) {
                // 1D: iterate over all elements using coefficient access
                for (int i = 0; i < n_points; ++i) {
                    // Use linear coefficient access (works for any Eigen expression)
                    Scalar val;
                    if (queries.cols() == 1) {
                        val = queries(i, 0);
                    } else {
                        val = queries(0, i);
                    }
                    result(i) = eval_numeric_scalar(val);
                }
            } else {
                // N-D: handle row/col orientation
                bool is_transposed = (queries.rows() == m_dims && queries.cols() != m_dims);
                for (int i = 0; i < n_points; ++i) {
                    NumericVector point(m_dims);
                    for (int d = 0; d < m_dims; ++d) {
                        point(d) = is_transposed ? queries(d, i) : queries(i, d);
                    }
                    result(i) = eval_numeric_point(point);
                }
            }
        } else {
            if (m_use_custom_hermite) {
                throw InterpolationError(detail::hermite_symbolic_error_message());
            }
            if (m_dims == 1) {
                for (int i = 0; i < n_points; ++i) {
                    Scalar val;
                    if (queries.cols() == 1) {
                        val = queries(i, 0);
                    } else {
                        val = queries(0, i);
                    }
                    result(i) = eval_symbolic_scalar(val);
                }
            } else {
                bool is_transposed = (queries.rows() == m_dims && queries.cols() != m_dims);
                for (int i = 0; i < n_points; ++i) {
                    SymbolicVector point(m_dims);
                    for (int d = 0; d < m_dims; ++d) {
                        point(d) = is_transposed ? queries(d, i) : queries(i, d);
                    }
                    result(i) = eval_symbolic_point(point);
                }
            }
        }

        return result;
    }

  private:
    // ========================================================================
    // Setup
    // ========================================================================

    void setup_method(InterpolationMethod method) {
        if (method == InterpolationMethod::Hermite) {
            // Hermite uses custom implementation, no CasADi interpolant needed
            m_use_custom_hermite = true;
            m_valid = true;
        } else if (method == InterpolationMethod::BSpline) {
            // BSpline requires at least 4 points per dimension for cubic
            for (int d = 0; d < m_dims; ++d) {
                if (m_grid[d].size() < 4) {
                    throw InterpolationError(
                        "Interpolator: BSpline requires at least 4 grid points in dimension " +
                        std::to_string(d));
                }
            }
            m_casadi_fn = casadi::interpolant("interp", "bspline", m_grid, m_values);
            m_valid = true;
        } else if (method == InterpolationMethod::Nearest) {
            // Nearest neighbor - use linear CasADi but round in eval
            m_casadi_fn = casadi::interpolant("interp", "linear", m_grid, m_values);
            m_valid = true;
        } else {
            // Linear (default)
            m_casadi_fn = casadi::interpolant("interp", "linear", m_grid, m_values);
            m_valid = true;
        }
    }

    /**
     * @brief Compute boundary partial derivatives for N-D linear extrapolation
     *
     * For each dimension d, computes ∂f/∂x_d at the left and right boundaries
     * using finite differences from the CasADi interpolant.
     */
    void compute_boundary_slopes() {
        if (m_extrap_config.mode != ExtrapolationMode::Linear) {
            return;
        }

        m_slopes_left.resize(m_dims, 0.0);
        m_slopes_right.resize(m_dims, 0.0);

        // For each dimension, compute partial derivative at boundaries
        for (int d = 0; d < m_dims; ++d) {
            const auto &grid_d = m_grid[d];
            size_t n_d = grid_d.size();
            if (n_d < 2)
                continue;

            double x_min = grid_d.front();
            double x_max = grid_d.back();
            double h_left = grid_d[1] - grid_d[0];
            double h_right = grid_d[n_d - 1] - grid_d[n_d - 2];

            // Create query point at center of other dimensions
            std::vector<double> query_base(m_dims);
            for (int dd = 0; dd < m_dims; ++dd) {
                // Use midpoint of grid for other dimensions
                query_base[dd] = 0.5 * (m_grid[dd].front() + m_grid[dd].back());
            }

            // Left boundary: finite difference using first two grid points
            query_base[d] = x_min;
            std::vector<casadi::DM> args0 = {casadi::DM(query_base)};
            double f0 = static_cast<double>(m_casadi_fn(args0)[0]);

            query_base[d] = x_min + h_left;
            std::vector<casadi::DM> args1 = {casadi::DM(query_base)};
            double f1 = static_cast<double>(m_casadi_fn(args1)[0]);

            m_slopes_left[d] = (f1 - f0) / h_left;

            // Right boundary: finite difference using last two grid points
            query_base[d] = x_max - h_right;
            std::vector<casadi::DM> args2 = {casadi::DM(query_base)};
            double f2 = static_cast<double>(m_casadi_fn(args2)[0]);

            query_base[d] = x_max;
            std::vector<casadi::DM> args3 = {casadi::DM(query_base)};
            double f3 = static_cast<double>(m_casadi_fn(args3)[0]);

            m_slopes_right[d] = (f3 - f2) / h_right;
        }
    }

    /**
     * @brief Apply output bounds to a numeric value
     */
    double apply_output_bounds(double value) const {
        if (m_extrap_config.lower_bound.has_value()) {
            value = std::max(value, m_extrap_config.lower_bound.value());
        }
        if (m_extrap_config.upper_bound.has_value()) {
            value = std::min(value, m_extrap_config.upper_bound.value());
        }
        return value;
    }

    /**
     * @brief Apply output bounds to a symbolic value (using fmax/fmin for AD)
     */
    SymbolicScalar apply_output_bounds_sym(const SymbolicScalar &value) const {
        SymbolicScalar result = value;
        if (m_extrap_config.lower_bound.has_value()) {
            result = SymbolicScalar::fmax(result, m_extrap_config.lower_bound.value());
        }
        if (m_extrap_config.upper_bound.has_value()) {
            result = SymbolicScalar::fmin(result, m_extrap_config.upper_bound.value());
        }
        return result;
    }

    // ========================================================================
    // Numeric Evaluation
    // ========================================================================

    double eval_numeric_scalar(double query) const {
        const double x_min = m_grid[0].front();
        const double x_max = m_grid[0].back();

        // Check if we need linear extrapolation
        if (m_extrap_config.mode == ExtrapolationMode::Linear && !m_slopes_left.empty()) {
            if (query < x_min) {
                // Left extrapolation: y = y[0] + slope_left * (x - x_min)
                double result = m_values.front() + m_slopes_left[0] * (query - x_min);
                return apply_output_bounds(result);
            } else if (query > x_max) {
                // Right extrapolation: y = y[n-1] + slope_right * (x - x_max)
                double result = m_values.back() + m_slopes_right[0] * (query - x_max);
                return apply_output_bounds(result);
            }
            // In bounds: fall through to normal interpolation
        }

        // Clamp to bounds (default behavior or in-bounds query)
        double clamped = std::max(query, x_min);
        clamped = std::min(clamped, x_max);

        double result;
        if (m_use_custom_hermite) {
            result = detail::hermite_interp_1d_numeric(m_grid[0], m_values, clamped);
        } else if (m_method == InterpolationMethod::Nearest) {
            // Snap to nearest grid point
            auto it = std::lower_bound(m_grid[0].begin(), m_grid[0].end(), clamped);
            size_t idx = std::distance(m_grid[0].begin(), it);
            if (idx > 0 && (idx == m_grid[0].size() || std::abs(clamped - m_grid[0][idx - 1]) <
                                                           std::abs(clamped - m_grid[0][idx]))) {
                idx--;
            }
            result = m_values[idx];
        } else {
            // Linear/BSpline: Use CasADi
            std::vector<casadi::DM> args = {casadi::DM(clamped)};
            std::vector<casadi::DM> res = m_casadi_fn(args);
            result = static_cast<double>(res[0]);
        }

        return apply_output_bounds(result);
    }

    double eval_numeric_point(const NumericVector &query) const {
        // Check for out-of-bounds dimensions and compute extrapolation corrections
        double extrap_correction = 0.0;
        bool needs_extrapolation = false;

        if (m_extrap_config.mode == ExtrapolationMode::Linear && !m_slopes_left.empty()) {
            for (int d = 0; d < m_dims; ++d) {
                double val = query(d);
                double x_min = m_grid[d].front();
                double x_max = m_grid[d].back();

                if (val < x_min) {
                    // Left extrapolation for dimension d
                    extrap_correction += m_slopes_left[d] * (val - x_min);
                    needs_extrapolation = true;
                } else if (val > x_max) {
                    // Right extrapolation for dimension d
                    extrap_correction += m_slopes_right[d] * (val - x_max);
                    needs_extrapolation = true;
                }
            }
        }

        // Build clamped query
        std::vector<double> clamped(m_dims);
        for (int d = 0; d < m_dims; ++d) {
            double val = query(d);
            val = std::max(val, m_grid[d].front());
            val = std::min(val, m_grid[d].back());
            clamped[d] = val;
        }

        double result;
        if (m_use_custom_hermite) {
            result = detail::hermite_interpn_numeric(m_grid, m_values, clamped);
        } else if (m_method == InterpolationMethod::Nearest) {
            // TODO: Implement N-D nearest neighbor
            // For now, fall through to linear
            std::vector<casadi::DM> args = {casadi::DM(clamped)};
            std::vector<casadi::DM> res = m_casadi_fn(args);
            result = static_cast<double>(res[0]);
        } else {
            // Linear/BSpline: Use CasADi
            std::vector<casadi::DM> args = {casadi::DM(clamped)};
            std::vector<casadi::DM> res = m_casadi_fn(args);
            result = static_cast<double>(res[0]);
        }

        // Add extrapolation correction
        if (needs_extrapolation) {
            result += extrap_correction;
        }

        return apply_output_bounds(result);
    }

    // ========================================================================
    // Symbolic Evaluation
    // ========================================================================

    SymbolicScalar eval_symbolic_scalar(const SymbolicScalar &query) const {
        const double x_min = m_grid[0].front();
        const double x_max = m_grid[0].back();

        // Clamped query for interpolation
        SymbolicScalar clamped = SymbolicScalar::fmax(query, x_min);
        clamped = SymbolicScalar::fmin(clamped, x_max);

        SymbolicScalar interp_result;
        if (m_method == InterpolationMethod::Nearest) {
            // Symbolic nearest: snap to closest grid point using if_else cascade.
            // Start with the last grid value and walk backwards through midpoints.
            const auto &g = m_grid[0];
            interp_result = SymbolicScalar(m_values.back());
            for (int i = static_cast<int>(g.size()) - 2; i >= 0; --i) {
                double midpoint = 0.5 * (g[i] + g[i + 1]);
                interp_result = casadi::MX::if_else(clamped < midpoint, SymbolicScalar(m_values[i]),
                                                    interp_result);
            }
        } else {
            // Linear/BSpline: Use CasADi interpolant
            std::vector<casadi::MX> args = {clamped};
            std::vector<casadi::MX> res = m_casadi_fn(args);
            interp_result = res[0];
        }

        // Handle extrapolation if configured
        if (m_extrap_config.mode == ExtrapolationMode::Linear && !m_slopes_left.empty()) {
            // Left extrapolation: y = y[0] + slope_left * (x - x_min)
            SymbolicScalar left_extrap = m_values.front() + m_slopes_left[0] * (query - x_min);

            // Right extrapolation: y = y[n-1] + slope_right * (x - x_max)
            SymbolicScalar right_extrap = m_values.back() + m_slopes_right[0] * (query - x_max);

            // Use if_else for smooth symbolic branching
            SymbolicScalar result = casadi::MX::if_else(
                query < x_min, left_extrap,
                casadi::MX::if_else(query > x_max, right_extrap, interp_result));

            return apply_output_bounds_sym(result);
        }

        return apply_output_bounds_sym(interp_result);
    }

    SymbolicScalar eval_symbolic_point(const SymbolicVector &query) const {
        // Build clamped query
        casadi::MX clamped(m_dims, 1);
        for (int d = 0; d < m_dims; ++d) {
            casadi::MX val = query(d);
            val = casadi::MX::fmax(val, m_grid[d].front());
            val = casadi::MX::fmin(val, m_grid[d].back());
            clamped(d) = val;
        }

        // Get interpolated result at clamped location
        std::vector<casadi::MX> args = {clamped};
        std::vector<casadi::MX> res = m_casadi_fn(args);
        SymbolicScalar interp_result = res[0];

        // Handle N-D extrapolation if configured
        if (m_extrap_config.mode == ExtrapolationMode::Linear && !m_slopes_left.empty()) {
            // Add extrapolation corrections for each out-of-bounds dimension
            SymbolicScalar extrap_correction = 0.0;

            for (int d = 0; d < m_dims; ++d) {
                double x_min = m_grid[d].front();
                double x_max = m_grid[d].back();
                casadi::MX val = query(d);

                // Left extrapolation correction: slope * (x - x_min) when x < x_min
                SymbolicScalar left_corr = m_slopes_left[d] * (val - x_min);
                SymbolicScalar left_contrib = casadi::MX::if_else(val < x_min, left_corr, 0.0);

                // Right extrapolation correction: slope * (x - x_max) when x > x_max
                SymbolicScalar right_corr = m_slopes_right[d] * (val - x_max);
                SymbolicScalar right_contrib = casadi::MX::if_else(val > x_max, right_corr, 0.0);

                extrap_correction = extrap_correction + left_contrib + right_contrib;
            }

            return apply_output_bounds_sym(interp_result + extrap_correction);
        }

        return apply_output_bounds_sym(interp_result);
    }
};

// ============================================================================
// Free Function API (Backwards Compatibility)
// ============================================================================

/**
 * @brief N-dimensional interpolation on regular grids (backwards compatibility)
 *
 * @deprecated Use Interpolator class instead for better performance (cached CasADi function)
 *
 * This function creates a new Interpolator for each call. For repeated queries,
 * construct an Interpolator once and reuse it.
 *
 * @tparam Scalar Scalar type (double or casadi::MX)
 * @param points Vector of 1D coordinate arrays for each dimension
 * @param values_flat Flattened values in Fortran order (column-major)
 * @param xi Query points, shape (n_points, n_dimensions)
 * @param method Interpolation method
 * @param fill_value Optional value for out-of-bounds queries (extrapolates if nullopt)
 * @return Vector of interpolated values at query points
 */
template <typename Scalar>
JanusVector<Scalar> interpn(const std::vector<NumericVector> &points,
                            const NumericVector &values_flat, const JanusMatrix<Scalar> &xi,
                            InterpolationMethod method = InterpolationMethod::Linear,
                            std::optional<Scalar> fill_value = std::nullopt) {
    const auto grid_data = detail::build_interpn_grid(points, "interpn");
    const int n_dims = static_cast<int>(points.size());
    JanusMatrix<Scalar> xi_work = detail::normalize_query_matrix(xi, n_dims, "interpn");
    const int n_points = static_cast<int>(xi_work.rows());

    // Validate values size
    if (values_flat.size() != grid_data.expected_size) {
        throw InterpolationError("interpn: values_flat size mismatch. Expected " +
                                 std::to_string(grid_data.expected_size) + ", got " +
                                 std::to_string(values_flat.size()));
    }

    // Create interpolator
    Interpolator interp(points, values_flat, method);

    // Prepare result
    JanusVector<Scalar> result(n_points);

    // Handle fill_value for out-of-bounds
    if (fill_value.has_value()) {
        for (int i = 0; i < n_points; ++i) {
            bool out_of_bounds = false;
            if constexpr (std::is_floating_point_v<Scalar>) {
                NumericVector point = xi_work.row(i).transpose();
                out_of_bounds = detail::point_out_of_bounds_numeric(point, grid_data.grid);
            }

            if (out_of_bounds) {
                result(i) = fill_value.value();
            } else {
                JanusVector<Scalar> point = xi_work.row(i).transpose();
                result(i) = interp(point);
            }
        }
    } else {
        // No fill_value, allow extrapolation (clamp)
        for (int i = 0; i < n_points; ++i) {
            JanusVector<Scalar> point = xi_work.row(i).transpose();
            result(i) = interp(point);
        }
    }

    return result;
}

/**
 * @brief N-dimensional interpolation with symbolic table values
 *
 * This overload keeps the table coefficients symbolic so they can participate
 * in optimization and automatic differentiation. It currently supports Linear
 * and BSpline methods.
 *
 * @tparam Scalar Query scalar type (double or casadi::MX)
 * @param points Vector of 1D coordinate arrays for each dimension
 * @param values_flat Flattened symbolic table values in Fortran order
 * @param xi Query points, shape (n_points, n_dimensions)
 * @param method Interpolation method
 * @param fill_value Optional value for out-of-bounds queries
 * @return SymbolicVector of interpolated values at query points
 */
template <typename Scalar>
SymbolicVector interpn(const std::vector<NumericVector> &points, const SymbolicVector &values_flat,
                       const JanusMatrix<Scalar> &xi,
                       InterpolationMethod method = InterpolationMethod::Linear,
                       std::optional<SymbolicScalar> fill_value = std::nullopt) {
    const auto grid_data = detail::build_interpn_grid(points, "interpn");
    const int n_dims = static_cast<int>(points.size());
    JanusMatrix<Scalar> xi_work = detail::normalize_query_matrix(xi, n_dims, "interpn");
    const int n_points = static_cast<int>(xi_work.rows());

    if (values_flat.size() != grid_data.expected_size) {
        throw InterpolationError("interpn: values_flat size mismatch. Expected " +
                                 std::to_string(grid_data.expected_size) + ", got " +
                                 std::to_string(values_flat.size()));
    }

    casadi::Function interp = detail::make_parametric_interpolant(grid_data.grid, method);
    const SymbolicScalar coeffs = janus::as_mx(values_flat);

    SymbolicVector result(n_points);
    for (int i = 0; i < n_points; ++i) {
        JanusVector<Scalar> point = xi_work.row(i).transpose();
        SymbolicScalar clamped_point = detail::clamp_query_point(point, grid_data.grid);
        SymbolicScalar interp_value = interp(std::vector<SymbolicScalar>{clamped_point, coeffs})[0];

        if (fill_value.has_value()) {
            if constexpr (std::is_floating_point_v<Scalar>) {
                result(i) = detail::point_out_of_bounds_numeric(point, grid_data.grid)
                                ? fill_value.value()
                                : interp_value;
            } else {
                SymbolicScalar out_of_bounds =
                    detail::point_out_of_bounds_symbolic(point, grid_data.grid);
                result(i) =
                    SymbolicScalar::if_else(out_of_bounds, fill_value.value(), interp_value);
            }
        } else {
            result(i) = interp_value;
        }
    }

    return result;
}

} // namespace janus
