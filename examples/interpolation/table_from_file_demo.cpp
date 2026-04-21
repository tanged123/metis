#include <fstream>
#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <sstream>
#include <vector>

/**
 * Table Interpolation from File Demo
 *
 * Demonstrates loading interpolation table data from an external file
 * at initialization time, then using it for both numeric and symbolic
 * evaluation with full automatic differentiation support.
 *
 * This pattern is ideal for:
 *   - Aerodynamic coefficient tables (Cl, Cd vs alpha, Mach)
 *   - Atmospheric property tables
 *   - Engine thrust/ISP tables
 *   - Any tabulated data loaded at startup
 */

// ============================================================================
// File Loading Utilities
// ============================================================================

/**
 * @brief Load 1D table data from a simple two-column CSV file
 *
 * Expected format (with or without header):
 *   x, y
 *   0.0, 1.0
 *   1.0, 2.5
 *   ...
 *
 * @param filepath Path to CSV file
 * @param x_out Output: independent variable values
 * @param y_out Output: dependent variable values
 * @param skip_header If true, skip the first line
 * @return true on success
 */
bool load_table_1d(const std::string &filepath, metis::NumericVector &x_out,
                   metis::NumericVector &y_out, bool skip_header = true) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filepath << "\n";
        return false;
    }

    std::vector<double> x_data, y_data;
    std::string line;

    if (skip_header && std::getline(file, line)) {
        // Skip header line
    }

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue; // Skip empty/comment lines

        std::stringstream ss(line);
        std::string cell;
        double x, y;

        if (std::getline(ss, cell, ',')) {
            x = std::stod(cell);
        } else {
            continue;
        }
        if (std::getline(ss, cell, ',')) {
            y = std::stod(cell);
        } else {
            continue;
        }

        x_data.push_back(x);
        y_data.push_back(y);
    }

    if (x_data.empty()) {
        std::cerr << "Error: No data loaded from file\n";
        return false;
    }

    // Convert to Eigen vectors
    x_out = Eigen::Map<metis::NumericVector>(x_data.data(), x_data.size());
    y_out = Eigen::Map<metis::NumericVector>(y_data.data(), y_data.size());

    return true;
}

/**
 * @brief Load 2D table data from a grid CSV file
 *
 * Expected format:
 *   , y0, y1, y2, ...        <- y-grid values in first row
 *   x0, z00, z01, z02, ...   <- x-grid value, then z values
 *   x1, z10, z11, z12, ...
 *   ...
 *
 * @param filepath Path to CSV file
 * @param x_out Output: x-grid values
 * @param y_out Output: y-grid values
 * @param z_out Output: z values in Fortran order (column-major)
 * @return true on success
 */
bool load_table_2d(const std::string &filepath, metis::NumericVector &x_out,
                   metis::NumericVector &y_out, metis::NumericVector &z_out) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filepath << "\n";
        return false;
    }

    std::vector<double> x_data, y_data;
    std::vector<std::vector<double>> z_rows;
    std::string line;

    // Read header row for y-values
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ','); // Skip empty corner cell
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                y_data.push_back(std::stod(cell));
            }
        }
    }

    // Read data rows
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        // First column is x-value
        if (std::getline(ss, cell, ',')) {
            x_data.push_back(std::stod(cell));
        }

        // Remaining columns are z-values
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(std::stod(cell));
            }
        }
        z_rows.push_back(row);
    }

    if (x_data.empty() || y_data.empty()) {
        std::cerr << "Error: No data loaded from file\n";
        return false;
    }

    // Convert to Eigen vectors
    x_out = Eigen::Map<metis::NumericVector>(x_data.data(), x_data.size());
    y_out = Eigen::Map<metis::NumericVector>(y_data.data(), y_data.size());

    // Flatten z-values in Fortran (column-major) order for CasADi
    // z_out[i + j*nx] = z_rows[i][j]
    int nx = static_cast<int>(x_data.size());
    int ny = static_cast<int>(y_data.size());
    z_out.resize(nx * ny);

    for (int j = 0; j < ny; ++j) {     // y index (columns)
        for (int i = 0; i < nx; ++i) { // x index (rows)
            z_out(i + j * nx) = z_rows[i][j];
        }
    }

    return true;
}

// ============================================================================
// Demo Main
// ============================================================================

int main(int argc, char *argv[]) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Table Interpolation from File Demo ===\n\n";

    // -------------------------------------------------------------------------
    // Option 1: Generate sample data files (if not provided)
    // In a real application, these would be your aero/engine tables
    // -------------------------------------------------------------------------

    std::string file_1d = "/tmp/sample_table_1d.csv";
    std::string file_2d = "/tmp/sample_table_2d.csv";

    // Create sample 1D table: thrust vs. throttle
    {
        std::ofstream out(file_1d);
        out << "throttle,thrust_kN\n"; // Header
        for (int i = 0; i <= 10; ++i) {
            double throttle = i * 0.1;
            double thrust = 100.0 * throttle * throttle; // Nonlinear thrust curve
            out << throttle << "," << thrust << "\n";
        }
        std::cout << "Created sample 1D table: " << file_1d << "\n";
    }

    // Create sample 2D table: Cd vs (alpha, Mach)
    {
        std::ofstream out(file_2d);
        // Header: y-values (Mach)
        out << ",0.2,0.4,0.6,0.8,1.0\n";
        // Data rows: x-values (alpha in deg), then Cd values
        for (int alpha = 0; alpha <= 20; alpha += 5) {
            out << alpha;
            for (double mach : {0.2, 0.4, 0.6, 0.8, 1.0}) {
                // Simple Cd model: Cd = 0.02 + 0.001*alpha^2 + 0.1*(mach-0.8)^2
                double cd = 0.02 + 0.001 * alpha * alpha + 0.1 * std::pow(mach - 0.8, 2);
                out << "," << cd;
            }
            out << "\n";
        }
        std::cout << "Created sample 2D table: " << file_2d << "\n";
    }

    // -------------------------------------------------------------------------
    // 1D Example: Thrust Table Lookup
    // -------------------------------------------------------------------------
    std::cout << "\n--- 1D Table Interpolation (Thrust vs Throttle) ---\n";

    metis::NumericVector throttle_pts, thrust_pts;
    if (!load_table_1d(file_1d, throttle_pts, thrust_pts)) {
        return 1;
    }

    std::cout << "Loaded " << throttle_pts.size() << " data points from file\n";

    // Create interpolator at initialization time
    metis::Interpolator thrust_interp(throttle_pts, thrust_pts,
                                      metis::InterpolationMethod::BSpline);

    // Numeric evaluation
    double throttle_query = 0.75;
    double thrust_numeric = thrust_interp(throttle_query);
    double thrust_exact = 100.0 * throttle_query * throttle_query;

    std::cout << "Query: throttle = " << throttle_query << "\n";
    std::cout << "  Interpolated thrust: " << thrust_numeric << " kN\n";
    std::cout << "  Exact thrust:        " << thrust_exact << " kN\n";

    // Symbolic evaluation with gradient
    auto throttle_sym = metis::sym("throttle");
    auto thrust_sym = thrust_interp(throttle_sym);

    // Compute derivative (d(thrust)/d(throttle))
    auto d_thrust = metis::jacobian(thrust_sym, throttle_sym);

    metis::Function thrust_fn("thrust_fn", {throttle_sym}, {thrust_sym, d_thrust});
    auto result = thrust_fn(throttle_query);

    std::cout << "  Symbolic thrust:     " << result[0](0, 0) << " kN\n";
    std::cout << "  d(thrust)/d(throttle): " << result[1](0, 0) << " kN/unit\n";
    std::cout << "  Exact derivative:    " << 200.0 * throttle_query << " kN/unit\n";

    // -------------------------------------------------------------------------
    // 2D Example: Drag Coefficient Table
    // -------------------------------------------------------------------------
    std::cout << "\n--- 2D Table Interpolation (Cd vs Alpha, Mach) ---\n";

    metis::NumericVector alpha_pts, mach_pts, cd_values;
    if (!load_table_2d(file_2d, alpha_pts, mach_pts, cd_values)) {
        return 1;
    }

    std::cout << "Loaded " << alpha_pts.size() << "x" << mach_pts.size() << " grid from file\n";

    // Create 2D interpolator
    std::vector<metis::NumericVector> cd_grid = {alpha_pts, mach_pts};
    metis::Interpolator cd_interp(cd_grid, cd_values, metis::InterpolationMethod::BSpline);

    // Numeric query
    double alpha_q = 7.5; // degrees
    double mach_q = 0.65;
    metis::NumericVector query_pt(2);
    query_pt << alpha_q, mach_q;

    double cd_numeric = cd_interp(query_pt);
    double cd_exact = 0.02 + 0.001 * alpha_q * alpha_q + 0.1 * std::pow(mach_q - 0.8, 2);

    std::cout << "Query: alpha = " << alpha_q << " deg, Mach = " << mach_q << "\n";
    std::cout << "  Interpolated Cd: " << cd_numeric << "\n";
    std::cout << "  Exact Cd:        " << cd_exact << "\n";

    // Symbolic evaluation with Jacobian
    auto alpha_sym = metis::sym("alpha");
    auto mach_sym = metis::sym("mach");

    metis::SymbolicVector query_sym(2);
    query_sym(0) = alpha_sym;
    query_sym(1) = mach_sym;

    auto cd_sym = cd_interp(query_sym);
    auto grad_cd = metis::jacobian(cd_sym, alpha_sym, mach_sym);

    metis::Function cd_fn("cd_fn", {alpha_sym, mach_sym}, {cd_sym, grad_cd});
    auto cd_result = cd_fn(alpha_q, mach_q);

    std::cout << "  Symbolic Cd:     " << cd_result[0](0, 0) << "\n";
    auto grad_mat = cd_result[1];
    std::cout << "  Gradient [dCd/dalpha, dCd/dMach]: [" << grad_mat(0, 0) << ", " << grad_mat(0, 1)
              << "]\n";

    // Exact gradients
    double dcd_dalpha = 0.002 * alpha_q;
    double dcd_dmach = 0.2 * (mach_q - 0.8);
    std::cout << "  Exact gradient:  [" << dcd_dalpha << ", " << dcd_dmach << "]\n";

    // -------------------------------------------------------------------------
    // 3. Extrapolation Example: Linear Extrapolation with Bounds
    // -------------------------------------------------------------------------
    std::cout << "\n--- Extrapolation with Safety Bounds ---\n";

    // Create thrust interpolator WITH linear extrapolation and bounds
    metis::Interpolator thrust_extrap(throttle_pts, thrust_pts, metis::InterpolationMethod::BSpline,
                                      metis::ExtrapolationConfig::linear(0.0, 150.0));

    std::cout << "Created interpolator with ExtrapolationConfig::linear(0, 150)\n";

    // Query outside bounds
    double throttle_over = 1.2;   // Above max (1.0)
    double throttle_under = -0.1; // Below min (0.0)

    double thrust_over = thrust_extrap(throttle_over);
    double thrust_under = thrust_extrap(throttle_under);

    std::cout << "Query throttle = " << throttle_over << " (outside bounds):\n";
    std::cout << "  Extrapolated thrust: " << thrust_over << " kN (clamped to upper bound)\n";

    std::cout << "Query throttle = " << throttle_under << " (outside bounds):\n";
    std::cout << "  Extrapolated thrust: " << thrust_under << " kN (clamped to lower bound)\n";

    // Compare to clamp behavior
    metis::Interpolator thrust_clamp(throttle_pts, thrust_pts, metis::InterpolationMethod::BSpline);

    double thrust_clamp_over = thrust_clamp(throttle_over);
    std::cout << "\nWith default clamping (no extrapolation):\n";
    std::cout << "  Query throttle = " << throttle_over << ": " << thrust_clamp_over
              << " kN (clamped to boundary value)\n";

    // Symbolic extrapolation with gradient
    auto t_sym = metis::sym("throttle");
    auto thrust_extrap_sym = thrust_extrap(t_sym);
    auto d_thrust_extrap = metis::jacobian(thrust_extrap_sym, t_sym);

    metis::Function extrap_fn("extrap_fn", {t_sym}, {thrust_extrap_sym, d_thrust_extrap});
    auto extrap_result = extrap_fn(throttle_over);

    std::cout << "\nSymbolic extrapolation at throttle = " << throttle_over << ":\n";
    std::cout << "  Thrust: " << extrap_result[0](0, 0) << " kN\n";
    std::cout << "  d(thrust)/d(throttle): " << extrap_result[1](0, 0)
              << " (non-zero gradient even outside bounds!)\n";

    // -------------------------------------------------------------------------
    // Key Takeaway
    // -------------------------------------------------------------------------
    std::cout << "\n=== Summary ===\n";
    std::cout << "1. Load table data from files at startup (CSV, HDF5, etc.)\n";
    std::cout << "2. Construct metis::Interpolator with loaded data\n";
    std::cout << "3. Use the interpolator for both numeric AND symbolic evaluation\n";
    std::cout << "4. Automatic differentiation works through the table lookup!\n";
    std::cout << "5. Use ExtrapolationConfig::linear() for gradient-preserving extrapolation\n";
    std::cout << "   with optional output bounds for safety\n";
    std::cout << "\n=== Demo Complete ===\n";

    return 0;
}
