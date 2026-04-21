#pragma once
/**
 * @file ScatteredInterpolator.hpp
 * @brief RBF-based interpolation for unstructured scattered data
 * @see Interpolate.hpp
 */

#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include "metis/math/Interpolate.hpp"
#include "metis/math/Linalg.hpp"
#include <cmath>
#include <vector>

namespace metis {

// ============================================================================
// RBF Kernel Types
// ============================================================================

/**
 * @brief Radial basis function kernel types for scattered interpolation
 */
enum class RBFKernel {
    ThinPlateSpline, ///< r² log(r) - smooth, good default, no shape parameter
    Multiquadric,    ///< sqrt(1 + (ε*r)²) - adjustable shape parameter
    Gaussian,        ///< exp(-(ε*r)²) - localized influence
    Linear,          ///< r - simple, produces piecewise linear surface
    Cubic            ///< r³ - smooth, commonly used
};

// ============================================================================
// RBF Kernel Evaluation
// ============================================================================

namespace detail {

/// Regularization term for RBF interpolation matrix to prevent singularity
constexpr double rbf_regularization = 1e-10;
/// Threshold below which thin-plate spline uses linear fallback to avoid log(0)
constexpr double rbf_zero_threshold = 1e-15;

/**
 * @brief Evaluate RBF kernel φ(r) for a given distance
 *
 * @param r Distance between points (non-negative)
 * @param kernel Kernel type
 * @param epsilon Shape parameter (for Multiquadric/Gaussian)
 * @return Kernel value φ(r)
 */
inline double rbf_phi(double r, RBFKernel kernel, double epsilon = 1.0) {
    switch (kernel) {
    case RBFKernel::ThinPlateSpline:
        // φ(r) = r² log(r), with φ(0) = 0
        return (r > rbf_zero_threshold) ? r * r * std::log(r) : 0.0;
    case RBFKernel::Multiquadric:
        return std::sqrt(1.0 + (epsilon * r) * (epsilon * r));
    case RBFKernel::Gaussian:
        return std::exp(-(epsilon * r) * (epsilon * r));
    case RBFKernel::Linear:
        return r;
    case RBFKernel::Cubic:
        return r * r * r;
    default:
        return r * r * std::log(r + rbf_zero_threshold);
    }
}

/**
 * @brief Compute Euclidean distance between two points
 */
inline double euclidean_distance(const double *p1, const double *p2, int dims) {
    double sum = 0.0;
    for (int d = 0; d < dims; ++d) {
        double diff = p1[d] - p2[d];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

} // namespace detail

// ============================================================================
// Scattered Interpolator Class
// ============================================================================

/**
 * @brief Interpolator for scattered (unstructured) N-dimensional data
 *
 * Uses Radial Basis Functions (RBF) to fit scattered data points, then
 * resamples onto a regular grid for fast symbolic-compatible queries.
 *
 * The resampling approach ensures:
 * - Construction: O(N³) for N data points (RBF system solve)
 * - Query: O(1) via gridded interpolant (fast, symbolic-safe)
 *
 * @example Basic usage:
 * ```cpp
 * // Scattered wind tunnel data: (Mach, alpha) -> CL
 * NumericMatrix points(100, 2);  // 100 test points, 2D input
 * NumericVector values(100);      // CL values
 * // ... fill in data ...
 *
 * ScatteredInterpolator cl_table(points, values);
 *
 * // Query (numeric or symbolic)
 * NumericVector query(2);
 * query << 0.8, 5.0;  // Mach=0.8, alpha=5°
 * double cl = cl_table(query);
 * ```
 *
 * @example 1D usage:
 * ```cpp
 * NumericVector x(10), y(10);
 * // ... fill with scattered 1D data ...
 * ScatteredInterpolator interp(x, y);
 * double result = interp(1.5);
 * ```
 */
class ScatteredInterpolator {
  private:
    Interpolator m_gridded;
    int m_dims = 0;
    double m_reconstruction_error = 0.0;
    bool m_valid = false;

    // Store original data for error computation
    NumericMatrix m_original_points;
    NumericVector m_original_values;

  public:
    /**
     * @brief Default constructor (invalid state)
     */
    ScatteredInterpolator() = default;

    /**
     * @brief Construct from scattered N-D points with uniform grid resolution
     *
     * @param points Data locations, shape (n_points, n_dims)
     * @param values Function values at each point (length n_points)
     * @param grid_resolution Points per dimension for resampling grid
     * @param kernel RBF kernel type (default: ThinPlateSpline)
     * @param epsilon Shape parameter for Multiquadric/Gaussian kernels
     * @param method Gridded interpolation method for final queries
     * @throw InterpolationError if inputs are invalid
     */
    ScatteredInterpolator(const NumericMatrix &points, const NumericVector &values,
                          int grid_resolution = 20, RBFKernel kernel = RBFKernel::ThinPlateSpline,
                          double epsilon = 1.0,
                          InterpolationMethod method = InterpolationMethod::Linear) {
        if (points.rows() == 0 || points.cols() == 0) {
            throw InterpolationError("ScatteredInterpolator: points cannot be empty");
        }
        if (points.rows() != values.size()) {
            throw InterpolationError(
                "ScatteredInterpolator: points rows must equal values size. Got " +
                std::to_string(points.rows()) + " points, " + std::to_string(values.size()) +
                " values");
        }
        if (grid_resolution < 2) {
            throw InterpolationError("ScatteredInterpolator: grid_resolution must be >= 2");
        }

        m_dims = static_cast<int>(points.cols());
        m_original_points = points;
        m_original_values = values;

        // Build uniform grid spanning data range
        std::vector<NumericVector> grid(m_dims);
        for (int d = 0; d < m_dims; ++d) {
            double min_val = points.col(d).minCoeff();
            double max_val = points.col(d).maxCoeff();
            // Add small padding to avoid edge issues
            double padding = 0.01 * (max_val - min_val);
            if (padding < detail::rbf_regularization)
                padding = 0.1; // Handle constant dimension
            min_val -= padding;
            max_val += padding;

            grid[d] = NumericVector::LinSpaced(grid_resolution, min_val, max_val);
        }

        build_interpolator(points, values, grid, kernel, epsilon, method);
    }

    /**
     * @brief Construct with explicit per-dimension grid specification
     *
     * @param points Data locations, shape (n_points, n_dims)
     * @param values Function values at each point
     * @param grid_points Explicit grid coordinates per dimension
     * @param kernel RBF kernel type
     * @param epsilon Shape parameter for Multiquadric/Gaussian kernels
     * @param method Gridded interpolation method for final queries
     */
    ScatteredInterpolator(const NumericMatrix &points, const NumericVector &values,
                          const std::vector<NumericVector> &grid_points,
                          RBFKernel kernel = RBFKernel::ThinPlateSpline, double epsilon = 1.0,
                          InterpolationMethod method = InterpolationMethod::Linear) {
        if (points.rows() == 0 || points.cols() == 0) {
            throw InterpolationError("ScatteredInterpolator: points cannot be empty");
        }
        if (points.rows() != values.size()) {
            throw InterpolationError("ScatteredInterpolator: points rows must equal values size");
        }
        if (static_cast<int>(grid_points.size()) != points.cols()) {
            throw InterpolationError(
                "ScatteredInterpolator: grid_points dimensions must match points columns");
        }

        m_dims = static_cast<int>(points.cols());
        m_original_points = points;
        m_original_values = values;

        build_interpolator(points, values, grid_points, kernel, epsilon, method);
    }

    /**
     * @brief 1D convenience constructor
     *
     * @param x Scattered x-coordinates
     * @param y Function values at each x
     * @param grid_resolution Points for resampling grid
     * @param kernel RBF kernel type
     */
    ScatteredInterpolator(const NumericVector &x, const NumericVector &y, int grid_resolution = 50,
                          RBFKernel kernel = RBFKernel::ThinPlateSpline) {
        if (x.size() != y.size()) {
            throw InterpolationError("ScatteredInterpolator: x and y must have same size");
        }
        if (x.size() < 2) {
            throw InterpolationError("ScatteredInterpolator: need at least 2 points");
        }

        m_dims = 1;
        m_original_points.resize(x.size(), 1);
        m_original_points.col(0) = x;
        m_original_values = y;

        // Build 1D grid
        double min_val = x.minCoeff();
        double max_val = x.maxCoeff();
        double padding = 0.01 * (max_val - min_val);
        if (padding < detail::rbf_regularization)
            padding = 0.1;

        std::vector<NumericVector> grid(1);
        grid[0] = NumericVector::LinSpaced(grid_resolution, min_val - padding, max_val + padding);

        build_interpolator(m_original_points, y, grid, kernel, 1.0, InterpolationMethod::Linear);
    }

    // ========================================================================
    // Query Methods
    // ========================================================================

    /**
     * @brief Evaluate at N-D point
     * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
     * @param query Query point vector
     * @return Interpolated value
     */
    template <MetisScalar Scalar> Scalar operator()(const MetisVector<Scalar> &query) const {
        if (!m_valid) {
            throw InterpolationError("ScatteredInterpolator: not initialized");
        }
        return m_gridded(query);
    }

    /**
     * @brief Evaluate at scalar (1D only)
     * @tparam Scalar Scalar type (NumericScalar or SymbolicScalar)
     * @param query Scalar query value
     * @return Interpolated value
     */
    template <MetisScalar Scalar> Scalar operator()(const Scalar &query) const {
        if (!m_valid) {
            throw InterpolationError("ScatteredInterpolator: not initialized");
        }
        if (m_dims != 1) {
            throw InterpolationError("ScatteredInterpolator: scalar query only valid for 1D");
        }
        return m_gridded(query);
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /**
     * @brief Get number of input dimensions
     * @return Number of dimensions
     */
    int dims() const { return m_dims; }

    /**
     * @brief Check if interpolator is valid (initialized)
     * @return True if initialized
     */
    bool valid() const { return m_valid; }

    /**
     * @brief Get underlying gridded interpolator (for inspection)
     * @return Reference to gridded interpolator
     */
    const Interpolator &gridded() const { return m_gridded; }

    /**
     * @brief Get RMS reconstruction error at original scattered points
     *
     * Measures how well the gridded approximation reproduces the
     * original scattered data. Lower is better.
     */
    double reconstruction_error() const { return m_reconstruction_error; }

  private:
    /**
     * @brief Build the gridded interpolator from RBF fit
     */
    void build_interpolator(const NumericMatrix &points, const NumericVector &values,
                            const std::vector<NumericVector> &grid, RBFKernel kernel,
                            double epsilon, InterpolationMethod method) {
        int n_points = static_cast<int>(points.rows());
        int n_dims = static_cast<int>(points.cols());

        // ====================================================================
        // Step 1: Build RBF interpolation matrix and solve for weights
        // ====================================================================
        // For thin plate spline in 2D, we need polynomial terms for uniqueness
        // General form: f(x) = Σ w_i φ(||x - x_i||) + p(x)
        // We'll use the simpler pure RBF form for now

        NumericMatrix Phi(n_points, n_points);
        for (int i = 0; i < n_points; ++i) {
            for (int j = 0; j < n_points; ++j) {
                double r =
                    detail::euclidean_distance(points.row(i).data(), points.row(j).data(), n_dims);
                Phi(i, j) = detail::rbf_phi(r, kernel, epsilon);
            }
        }

        // Add small regularization for numerical stability
        Phi.diagonal().array() += detail::rbf_regularization;

        // Solve Phi * weights = values
        NumericVector weights = solve(Phi, values);

        // ====================================================================
        // Step 2: Evaluate RBF at grid points
        // ====================================================================
        // Compute total grid size
        int grid_size = 1;
        for (int d = 0; d < n_dims; ++d) {
            grid_size *= static_cast<int>(grid[d].size());
        }

        // Flatten grid values (Fortran order for Interpolator)
        NumericVector grid_values(grid_size);

        // Iterate over all grid points
        std::vector<int> dims_sizes(n_dims);
        for (int d = 0; d < n_dims; ++d) {
            dims_sizes[d] = static_cast<int>(grid[d].size());
        }

        for (int flat_idx = 0; flat_idx < grid_size; ++flat_idx) {
            // Convert flat index to multi-index (Fortran order)
            std::vector<int> multi_idx(n_dims);
            int remaining = flat_idx;
            for (int d = 0; d < n_dims; ++d) {
                multi_idx[d] = remaining % dims_sizes[d];
                remaining /= dims_sizes[d];
            }

            // Get grid point coordinates
            std::vector<double> grid_pt(n_dims);
            for (int d = 0; d < n_dims; ++d) {
                grid_pt[d] = grid[d](multi_idx[d]);
            }

            // Evaluate RBF sum at this grid point
            double val = 0.0;
            for (int i = 0; i < n_points; ++i) {
                double r = detail::euclidean_distance(grid_pt.data(), points.row(i).data(), n_dims);
                val += weights(i) * detail::rbf_phi(r, kernel, epsilon);
            }
            grid_values(flat_idx) = val;
        }

        // ====================================================================
        // Step 3: Create gridded interpolator
        // ====================================================================
        m_gridded = Interpolator(grid, grid_values, method);
        m_valid = true;

        // ====================================================================
        // Step 4: Compute reconstruction error
        // ====================================================================
        compute_reconstruction_error();
    }

    /**
     * @brief Compute RMS error at original scattered points
     */
    void compute_reconstruction_error() {
        if (!m_valid)
            return;

        double sum_sq_error = 0.0;
        int n = static_cast<int>(m_original_values.size());

        for (int i = 0; i < n; ++i) {
            NumericVector query = m_original_points.row(i).transpose();
            double predicted = m_gridded(query);
            double error = predicted - m_original_values(i);
            sum_sq_error += error * error;
        }

        m_reconstruction_error = std::sqrt(sum_sq_error / n);
    }
};

} // namespace metis
