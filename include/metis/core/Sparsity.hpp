/// @file Sparsity.hpp
/// @brief Sparsity pattern analysis, graph coloring, and sparse derivative evaluators
#pragma once

#include "Function.hpp"
#include "MetisError.hpp"
#include "MetisIO.hpp"
#include "MetisTypes.hpp"
#include <casadi/casadi.hpp>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace metis {

/**
 * @brief Wrapper around CasADi Sparsity for pattern analysis
 *
 * Provides query interface for Jacobian/Hessian sparsity patterns useful for
 * debugging, optimization configuration, and visualization.
 *
 * @example
 * ```cpp
 * auto x = metis::sym("x", 3);
 * auto y = metis::sym("y", 3);
 * auto f = casadi::MX::dot(x, y);  // x'*y
 *
 * auto sp = metis::jacobian_sparsity(f, casadi::MX::vertcat({x, y}));
 * std::cout << "Jacobian size: " << sp.n_rows() << "x" << sp.n_cols() << "\n";
 * std::cout << "Non-zeros: " << sp.nnz() << "\n";
 * std::cout << sp.to_string() << "\n";  // ASCII spy plot
 * ```
 */
class SparsityPattern {
  public:
    /// @brief Default-construct an empty 0x0 pattern
    SparsityPattern() : sp_(casadi::Sparsity(0, 0)) {}

    /**
     * @brief Construct from CasADi Sparsity object
     * @param sp CasADi sparsity to wrap
     */
    explicit SparsityPattern(const casadi::Sparsity &sp) : sp_(sp) {}

    /**
     * @brief Construct from CasADi MX (extracts its sparsity)
     * @param mx Symbolic expression whose sparsity is extracted
     */
    explicit SparsityPattern(const SymbolicScalar &mx) : sp_(mx.sparsity()) {}

    // === Query Interface ===

    /**
     * @brief Number of rows in the pattern
     * @return Row count
     */
    int n_rows() const { return static_cast<int>(sp_.size1()); }

    /**
     * @brief Number of columns in the pattern
     * @return Column count
     */
    int n_cols() const { return static_cast<int>(sp_.size2()); }

    /**
     * @brief Number of structural non-zeros
     * @return Non-zero count
     */
    int nnz() const { return static_cast<int>(sp_.nnz()); }

    /**
     * @brief Sparsity density (nnz / total elements)
     * @return Density in [0, 1]
     */
    double density() const {
        int total = n_rows() * n_cols();
        return total > 0 ? static_cast<double>(nnz()) / total : 0.0;
    }

    /**
     * @brief Check if a specific element is structurally non-zero
     * @param row Row index
     * @param col Column index
     * @return true if (row, col) is structurally non-zero
     */
    bool has_nz(int row, int col) const { return sp_.has_nz(row, col); }

    /**
     * @brief Get all non-zero positions as (row, col) pairs
     * @return Vector of (row, col) pairs
     */
    std::vector<std::pair<int, int>> nonzeros() const {
        std::vector<std::pair<int, int>> result;
        result.reserve(nnz());

        auto [row_idx, col_idx] = get_triplet();
        for (size_t i = 0; i < row_idx.size(); ++i) {
            result.emplace_back(row_idx[i], col_idx[i]);
        }
        return result;
    }

    // === Export Formats ===

    /**
     * @brief Get triplet format (row indices, column indices)
     * @return Tuple of (row_indices, col_indices)
     */
    std::tuple<std::vector<int>, std::vector<int>> get_triplet() const {
        std::vector<casadi_int> rows, cols;
        sp_.get_triplet(rows, cols);

        // Convert casadi_int to int
        std::vector<int> row_int(rows.begin(), rows.end());
        std::vector<int> col_int(cols.begin(), cols.end());
        return {row_int, col_int};
    }

    /**
     * @brief Get Compressed Row Storage (CRS) format
     * @return (row_ptr, col_idx) where row_ptr has size n_rows+1
     */
    std::tuple<std::vector<int>, std::vector<int>> get_crs() const {
        auto crs_sp = sp_.T(); // Transpose to get CRS from CCS
        std::vector<casadi_int> colind = crs_sp.get_colind();
        std::vector<casadi_int> row = crs_sp.get_row();

        std::vector<int> row_ptr(colind.begin(), colind.end());
        std::vector<int> col_idx(row.begin(), row.end());
        return {row_ptr, col_idx};
    }

    /**
     * @brief Get Compressed Column Storage (CCS) format
     * @return (col_ptr, row_idx) where col_ptr has size n_cols+1
     */
    std::tuple<std::vector<int>, std::vector<int>> get_ccs() const {
        std::vector<casadi_int> colind = sp_.get_colind();
        std::vector<casadi_int> row = sp_.get_row();

        std::vector<int> col_ptr(colind.begin(), colind.end());
        std::vector<int> row_idx(row.begin(), row.end());
        return {col_ptr, row_idx};
    }

    // === Visualization ===

    /**
     * @brief Generate ASCII spy plot of sparsity pattern
     *
     * Returns a string representation where '*' = non-zero, '.' = zero
     *
     * @param max_rows Maximum rows to display (default 40)
     * @param max_cols Maximum cols to display (default 80)
     * @return ASCII art string
     */
    std::string to_string(int max_rows = 40, int max_cols = 80) const {
        std::ostringstream oss;

        int rows = n_rows();
        int cols = n_cols();

        // Header
        oss << "Sparsity: " << rows << "x" << cols << ", nnz=" << nnz()
            << " (density=" << std::fixed << std::setprecision(3) << density() * 100 << "%)\n";

        // Limit display size
        int disp_rows = std::min(rows, max_rows);
        int disp_cols = std::min(cols, max_cols);

        // Top border
        oss << "┌";
        for (int j = 0; j < disp_cols; ++j)
            oss << "─";
        if (cols > max_cols)
            oss << "…";
        oss << "┐\n";

        // Rows
        for (int i = 0; i < disp_rows; ++i) {
            oss << "│";
            for (int j = 0; j < disp_cols; ++j) {
                oss << (has_nz(i, j) ? '*' : '.');
            }
            if (cols > max_cols)
                oss << "…";
            oss << "│\n";
        }

        if (rows > max_rows) {
            oss << "│";
            for (int j = 0; j < disp_cols; ++j)
                oss << "⋮";
            if (cols > max_cols)
                oss << "…";
            oss << "│\n";
        }

        // Bottom border
        oss << "└";
        for (int j = 0; j < disp_cols; ++j)
            oss << "─";
        if (cols > max_cols)
            oss << "…";
        oss << "┘\n";

        return oss.str();
    }

    /**
     * @brief Export sparsity pattern to DOT format as a matrix grid (spy plot)
     *
     * Creates a DOT file that renders as a grid where non-zeros are colored squares.
     * This mimics standard spy() plots found in MATLAB/Python.
     *
     * @param filename Output filename (without .dot extension)
     * @param name Graph name
     */
    void export_spy_dot(const std::string &filename, const std::string &name = "sparsity") const {
        std::ofstream file(filename + ".dot");
        if (!file.is_open()) {
            throw RuntimeError("Cannot open file: " + filename + ".dot");
        }

        file << "digraph " << name << " {\n";
        file << "  node [shape=plaintext];\n";
        file << "  labelloc=\"t\";\n";
        file << "  label=\"" << name << " (" << n_rows() << "x" << n_cols() << ", nnz=" << nnz()
             << ")\";\n\n";

        file << "  matrix [label=<\n";
        file << "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n";

        // Iterate rows (HTML tables are row-major)
        for (int i = 0; i < n_rows(); ++i) {
            file << "      <TR>\n";
            for (int j = 0; j < n_cols(); ++j) {
                if (has_nz(i, j)) {
                    // Non-zero element: Black (or dark blue)
                    file << "        <TD BGCOLOR=\"black\" WIDTH=\"20\" HEIGHT=\"20\"></TD>\n";
                } else {
                    // Zero element: White (or transparent)
                    file << "        <TD BGCOLOR=\"white\" WIDTH=\"20\" HEIGHT=\"20\"></TD>\n";
                }
            }
            file << "      </TR>\n";
        }

        file << "    </TABLE>\n";
        file << "  >];\n";
        file << "}\n";
        file.close();
    }

    /**
     * @brief Export/Render sparsity pattern to PDF via Graphviz
     *
     * Creates a DOT file and renders it to PDF using the `dot` command.
     * Use this for high-quality visualization of sparsity patterns.
     *
     * @param output_base Base filename (creates output_base.dot and output_base.pdf)
     * @return true if successful (requires Graphviz installed)
     */
    bool visualize_spy(const std::string &output_base) const {
        try {
            export_spy_dot(output_base, output_base);
            // render_graph is defined in MetisIO.hpp
            return render_graph(output_base + ".dot", output_base + ".pdf");
        } catch (...) {
            return false;
        }
    }

    /**
     * @brief Export sparsity pattern to an interactive HTML file
     *
     * Creates a self-contained HTML file with an SVG spy plot.
     * Supports pan, zoom, and hover info showing (row, col) coordinates.
     *
     * @param filename Output filename (without extension, .html will be added)
     * @param name Optional title for the visualization
     */
    void export_spy_html(const std::string &filename, const std::string &name = "sparsity") const {
        std::string html_filename = filename + ".html";
        std::ofstream out(html_filename);
        if (!out.is_open()) {
            throw RuntimeError("Cannot open file: " + html_filename);
        }

        int rows = n_rows();
        int cols = n_cols();
        int cell_size = 12; // pixels per cell

        // Limit cell size for very large matrices
        if (rows > 100 || cols > 100) {
            cell_size = std::max(3, 1200 / std::max(rows, cols));
        }

        // Add margins for axis labels
        int margin_left = 50;
        int margin_top = 40;
        int svg_width = cols * cell_size + margin_left + 20;
        int svg_height = rows * cell_size + margin_top + 20;

        // Determine tick interval based on matrix size
        int tick_interval = 1;
        if (rows > 50 || cols > 50)
            tick_interval = 10;
        else if (rows > 20 || cols > 20)
            tick_interval = 5;

        out << R"SPYHTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>)SPYHTML"
            << name << R"SPYHTML(</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { height: 100%; width: 100%; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; display: flex; }
        #controls { position: fixed; top: 10px; left: 10px; z-index: 100; background: rgba(0,0,0,0.8);
                    padding: 12px; border-radius: 8px; }
        #controls button { margin: 2px; padding: 8px 14px; cursor: pointer; border: none;
                           border-radius: 4px; background: #4a4a6a; color: white; font-size: 13px; }
        #controls button:hover { background: #6a6a8a; }
        #container { flex: 1; cursor: grab; overflow: hidden; height: 100%; }
        #container:active { cursor: grabbing; }
        #sidebar { width: 280px; height: 100%; background: #16213e; padding: 16px; overflow-y: auto; 
                   border-left: 2px solid #0f3460; }
        #sidebar h2 { color: #e94560; margin-bottom: 12px; font-size: 16px; }
        #sidebar .section { margin-bottom: 14px; }
        #sidebar .label { color: #888; font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }
        #sidebar .value { background: #0f3460; padding: 10px; border-radius: 6px; font-family: monospace;
                          font-size: 14px; }
        #sidebar .stat { display: flex; justify-content: space-between; padding: 6px 0;
                         border-bottom: 1px solid #0f3460; }
        #sidebar .stat-value { color: #4dc3ff; font-weight: bold; }
        #info { position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8);
                padding: 10px; border-radius: 4px; font-size: 12px; }
        .nz { cursor: pointer; }
        .nz:hover { fill: #e94560 !important; }
        .axis-label { font-family: monospace; font-size: 10px; fill: #666; }
        .grid-line { stroke: #ddd; stroke-width: 0.5; }
    </style>
</head>
<body>
    <div id="controls">
        <button onclick="zoomIn()">Zoom +</button>
        <button onclick="zoomOut()">Zoom -</button>
        <button onclick="resetView()">Reset</button>
        <button onclick="fitToScreen()">Fit</button>
    </div>
    <div id="container">
        <svg id="spy" width=")SPYHTML"
            << svg_width << R"SPYHTML(" height=")SPYHTML" << svg_height
            << R"SPYHTML(" style="background:#ffffff;">
            <g id="matrix" transform="translate()SPYHTML"
            << margin_left << "," << margin_top << R"SPYHTML()">
)SPYHTML";

        // Draw grid lines at tick intervals
        for (int r = 0; r <= rows; r += tick_interval) {
            out << "                <line class=\"grid-line\" x1=\"0\" y1=\"" << (r * cell_size)
                << "\" x2=\"" << (cols * cell_size) << "\" y2=\"" << (r * cell_size) << "\"/>\n";
        }
        for (int c = 0; c <= cols; c += tick_interval) {
            out << "                <line class=\"grid-line\" x1=\"" << (c * cell_size)
                << "\" y1=\"0\" x2=\"" << (c * cell_size) << "\" y2=\"" << (rows * cell_size)
                << "\"/>\n";
        }

        // Draw axis labels
        for (int r = 0; r <= rows; r += tick_interval) {
            out << "                <text class=\"axis-label\" x=\"-5\" y=\""
                << (r * cell_size + cell_size / 2)
                << "\" text-anchor=\"end\" dominant-baseline=\"middle\">" << r << "</text>\n";
        }
        for (int c = 0; c <= cols; c += tick_interval) {
            out << "                <text class=\"axis-label\" x=\""
                << (c * cell_size + cell_size / 2) << "\" y=\"-8\" text-anchor=\"middle\">" << c
                << "</text>\n";
        }

        // Output non-zero cells as rectangles
        auto [row_idx, col_idx] = get_triplet();
        for (size_t i = 0; i < row_idx.size(); ++i) {
            int r = row_idx[i];
            int c = col_idx[i];
            out << "                <rect class=\"nz\" x=\"" << (c * cell_size) << "\" y=\""
                << (r * cell_size) << "\" width=\"" << cell_size << "\" height=\"" << cell_size
                << "\" fill=\"#1e3a5f\" data-row=\"" << r << "\" data-col=\"" << c << "\"/>\n";
        }

        out << R"SPYHTML(            </g>
        </svg>
    </div>
    <div id="sidebar">
        <h2>Sparsity Info</h2>
        <div class="section">
            <div class="stat"><span>Matrix Size</span><span class="stat-value">)SPYHTML"
            << rows << " x " << cols << R"SPYHTML(</span></div>
            <div class="stat"><span>Non-zeros</span><span class="stat-value">)SPYHTML"
            << nnz() << R"SPYHTML(</span></div>
            <div class="stat"><span>Density</span><span class="stat-value">)SPYHTML";

        // Format density nicely
        double dens = density() * 100;
        out << std::fixed << std::setprecision(2) << dens;

        out << R"SPYHTML(%</span></div>
        </div>
        <div class="section">
            <div class="label">Selected Cell</div>
            <div class="value" id="cell-info">Click on a non-zero cell</div>
        </div>
        <div class="section">
            <div class="label">Notation</div>
            <div class="value" id="notation-info">A(row, col)</div>
        </div>
    </div>
    <div id="info">Scroll to zoom - Drag to pan - Click cells for info</div>
    <script>
        let scale = 1, panX = 0, panY = 0, isDragging = false, startX, startY;
        const container = document.getElementById('container');
        const svg = document.getElementById('spy');
        const cellInfo = document.getElementById('cell-info');
        const notationInfo = document.getElementById('notation-info');
        const svgWidth = )SPYHTML"
            << svg_width << R"SPYHTML(;
        const svgHeight = )SPYHTML"
            << svg_height << R"SPYHTML(;
        const matrixRows = )SPYHTML"
            << rows << R"SPYHTML(;
        const matrixCols = )SPYHTML"
            << cols << R"SPYHTML(;

        fitToScreen();

        function updateTransform() {
            svg.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
            svg.style.transformOrigin = '0 0';
        }
        function zoomIn() { scale *= 1.3; updateTransform(); }
        function zoomOut() { scale /= 1.3; updateTransform(); }
        function resetView() { scale = 1; panX = 0; panY = 0; updateTransform(); }
        function fitToScreen() {
            const availWidth = window.innerWidth - 280;
            const scaleX = (availWidth - 40) / svgWidth;
            const scaleY = (window.innerHeight - 40) / svgHeight;
            scale = Math.min(scaleX, scaleY);
            panX = (availWidth - svgWidth * scale) / 2;
            panY = (window.innerHeight - svgHeight * scale) / 2;
            updateTransform();
        }
        container.addEventListener('wheel', e => {
            e.preventDefault();
            const rect = container.getBoundingClientRect();
            const mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
            const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
            panX = mouseX - (mouseX - panX) * zoomFactor;
            panY = mouseY - (mouseY - panY) * zoomFactor;
            scale *= zoomFactor;
            updateTransform();
        });
        container.addEventListener('mousedown', e => { 
            if (e.target.classList.contains('nz')) return;
            isDragging = true; startX = e.clientX - panX; startY = e.clientY - panY; 
        });
        container.addEventListener('mousemove', e => { if (isDragging) { panX = e.clientX - startX; panY = e.clientY - startY; updateTransform(); } });
        container.addEventListener('mouseup', () => isDragging = false);
        container.addEventListener('mouseleave', () => isDragging = false);

        // Cell interaction
        let selectedCell = null;
        svg.querySelectorAll('.nz').forEach(rect => {
            rect.addEventListener('click', e => {
                e.stopPropagation();
                if (selectedCell) selectedCell.style.stroke = 'none';
                selectedCell = e.target;
                selectedCell.style.stroke = '#e94560';
                selectedCell.style.strokeWidth = '2';
                
                const r = parseInt(e.target.dataset.row);
                const c = parseInt(e.target.dataset.col);
                cellInfo.textContent = `Row: ${r}, Col: ${c}`;
                notationInfo.textContent = `A(${r}, ${c}) = nonzero`;
            });
            rect.addEventListener('mouseenter', e => {
                const r = e.target.dataset.row;
                const c = e.target.dataset.col;
                cellInfo.textContent = `Row: ${r}, Col: ${c}`;
            });
        });
    </script>
</body>
</html>
)SPYHTML";
        out.close();
    }

    // === Underlying Access ===

    /**
     * @brief Access underlying CasADi Sparsity object
     * @return Const reference to the wrapped CasADi sparsity
     */
    const casadi::Sparsity &casadi_sparsity() const { return sp_; }

    // === Operators ===

    bool operator==(const SparsityPattern &other) const { return sp_ == other.sp_; }
    bool operator!=(const SparsityPattern &other) const { return sp_ != other.sp_; }

  private:
    casadi::Sparsity sp_;
};

/**
 * @brief Wrapper around a CasADi graph coloring assignment.
 *
 * CasADi encodes a coloring as a sparse matrix with one nonzero per colored
 * entry. Rows correspond to original derivative entries and columns correspond
 * to compressed seed directions.
 */
class GraphColoring {
  public:
    /// @brief Default-construct an empty coloring
    GraphColoring() = default;

    /// @brief Construct from a CasADi coloring sparsity
    /// @param coloring Sparse assignment matrix from CasADi
    explicit GraphColoring(const casadi::Sparsity &coloring) : pattern_(coloring) {}

    /**
     * @brief Number of original entries being colored
     * @return Entry count
     */
    int n_entries() const { return pattern_.n_rows(); }

    /**
     * @brief Number of compressed seed directions (colors)
     * @return Color count
     */
    int n_colors() const { return pattern_.n_cols(); }

    /**
     * @brief Compression factor relative to one direction per entry
     * @return n_entries / n_colors (0 if no colors)
     */
    double compression_ratio() const {
        return n_colors() > 0 ? static_cast<double>(n_entries()) / n_colors() : 0.0;
    }

    /**
     * @brief 0-based color assignment for each original entry
     * @return Vector of color indices (-1 for uncolored entries)
     */
    std::vector<int> colorvec() const {
        std::vector<int> colors(n_entries(), -1);
        auto [rows, cols] = pattern_.get_triplet();
        for (size_t i = 0; i < rows.size(); ++i) {
            colors.at(rows[i]) = cols[i];
        }
        return colors;
    }

    /**
     * @brief Access the sparse assignment matrix behind the coloring
     * @return Const reference to the assignment pattern
     */
    const SparsityPattern &pattern() const { return pattern_; }

  private:
    SparsityPattern pattern_;
};

/**
 * @brief Preferred directional mode for a sparse Jacobian evaluator
 */
enum class SparseJacobianMode {
    Forward, ///< Forward-mode (column compression)
    Reverse  ///< Reverse-mode (row compression)
};

namespace detail {

inline std::vector<SymbolicScalar> to_mx_vector(const std::vector<SymbolicArg> &args) {
    std::vector<SymbolicScalar> ret;
    ret.reserve(args.size());
    for (const auto &arg : args) {
        ret.push_back(arg.get());
    }
    return ret;
}

inline std::vector<SymbolicArg> to_symbolic_args(const std::vector<SymbolicScalar> &args) {
    std::vector<SymbolicArg> ret;
    ret.reserve(args.size());
    for (const auto &arg : args) {
        ret.emplace_back(arg);
    }
    return ret;
}

inline SymbolicScalar stack_nonzeros(const SymbolicScalar &expr) {
    std::vector<SymbolicScalar> nz = expr.get_nonzeros();
    if (nz.empty()) {
        return SymbolicScalar(0, 1);
    }
    return SymbolicScalar::vertcat(nz);
}

inline void validate_function_indices(const Function &fn, int output_idx, int input_idx,
                                      const char *where) {
    const auto &cfn = fn.casadi_function();
    if (output_idx < 0 || output_idx >= cfn.n_out()) {
        throw InvalidArgument(std::string(where) + ": output_idx out of range");
    }
    if (input_idx < 0 || input_idx >= cfn.n_in()) {
        throw InvalidArgument(std::string(where) + ": input_idx out of range");
    }
}

inline std::vector<SymbolicScalar> symbolic_inputs_like(const Function &fn,
                                                        const std::string &prefix) {
    const auto &cfn = fn.casadi_function();
    std::vector<SymbolicScalar> inputs;
    inputs.reserve(cfn.n_in());
    for (int i = 0; i < cfn.n_in(); ++i) {
        inputs.push_back(SymbolicScalar::sym(prefix + cfn.name_in(i), cfn.sparsity_in(i)));
    }
    return inputs;
}

inline GraphColoring make_forward_coloring(const casadi::Sparsity &jac_sparsity) {
    return GraphColoring(jac_sparsity.T().uni_coloring(jac_sparsity));
}

inline GraphColoring make_reverse_coloring(const casadi::Sparsity &jac_sparsity) {
    return GraphColoring(jac_sparsity.uni_coloring(jac_sparsity.T()));
}

struct SparseJacobianArtifacts {
    SparsityPattern sparsity;
    GraphColoring forward_coloring;
    GraphColoring reverse_coloring;
    Function values_function;
};

inline SparseJacobianArtifacts
make_sparse_jacobian_artifacts(const std::vector<SymbolicArg> &expressions,
                               const std::vector<SymbolicArg> &variables, const std::string &name) {
    SymbolicScalar expr_cat = SymbolicScalar::vertcat(to_mx_vector(expressions));
    SymbolicScalar var_cat = SymbolicScalar::vertcat(to_mx_vector(variables));
    SymbolicScalar jac = SymbolicScalar::jacobian(expr_cat, var_cat);

    Function values_function = name.empty() ? Function(variables, {stack_nonzeros(jac)})
                                            : Function(name, variables, {stack_nonzeros(jac)});

    return {
        SparsityPattern(jac.sparsity()),
        make_forward_coloring(jac.sparsity()),
        make_reverse_coloring(jac.sparsity()),
        std::move(values_function),
    };
}

inline SparseJacobianArtifacts make_sparse_jacobian_artifacts(const Function &fn, int output_idx,
                                                              int input_idx,
                                                              const std::string &name) {
    validate_function_indices(fn, output_idx, input_idx, "sparse_jacobian");

    std::vector<SymbolicScalar> inputs = symbolic_inputs_like(fn, "_sj_");
    std::vector<SymbolicScalar> outputs = fn.casadi_function()(inputs);
    SymbolicScalar jac = SymbolicScalar::jacobian(outputs.at(output_idx), inputs.at(input_idx));

    std::vector<SymbolicArg> input_args = to_symbolic_args(inputs);
    Function values_function = name.empty() ? Function(input_args, {stack_nonzeros(jac)})
                                            : Function(name, input_args, {stack_nonzeros(jac)});

    return {
        SparsityPattern(jac.sparsity()),
        make_forward_coloring(jac.sparsity()),
        make_reverse_coloring(jac.sparsity()),
        std::move(values_function),
    };
}

struct SparseHessianArtifacts {
    SparsityPattern sparsity;
    GraphColoring coloring;
    Function values_function;
};

inline SparseHessianArtifacts
make_sparse_hessian_artifacts(const SymbolicArg &expression,
                              const std::vector<SymbolicArg> &variables, const std::string &name) {
    SymbolicScalar var_cat = SymbolicScalar::vertcat(to_mx_vector(variables));
    SymbolicScalar hess = SymbolicScalar::hessian(expression.get(), var_cat);

    Function values_function = name.empty() ? Function(variables, {stack_nonzeros(hess)})
                                            : Function(name, variables, {stack_nonzeros(hess)});

    return {
        SparsityPattern(hess.sparsity()),
        GraphColoring(hess.sparsity().star_coloring()),
        std::move(values_function),
    };
}

inline SparseHessianArtifacts make_sparse_hessian_artifacts(const Function &fn, int output_idx,
                                                            int input_idx,
                                                            const std::string &name) {
    validate_function_indices(fn, output_idx, input_idx, "sparse_hessian");

    std::vector<SymbolicScalar> inputs = symbolic_inputs_like(fn, "_sh_");
    std::vector<SymbolicScalar> outputs = fn.casadi_function()(inputs);
    if (!outputs.at(output_idx).is_scalar()) {
        throw InvalidArgument("sparse_hessian: selected function output must be scalar");
    }

    SymbolicScalar hess = SymbolicScalar::hessian(outputs.at(output_idx), inputs.at(input_idx));
    std::vector<SymbolicArg> input_args = to_symbolic_args(inputs);
    Function values_function = name.empty() ? Function(input_args, {stack_nonzeros(hess)})
                                            : Function(name, input_args, {stack_nonzeros(hess)});

    return {
        SparsityPattern(hess.sparsity()),
        GraphColoring(hess.sparsity().star_coloring()),
        std::move(values_function),
    };
}

inline SparsityPattern function_jacobian_sparsity(const Function &fn, int output_idx,
                                                  int input_idx) {
    validate_function_indices(fn, output_idx, input_idx, "get_jacobian_sparsity");
    std::vector<SymbolicScalar> inputs = symbolic_inputs_like(fn, "_jsp_");
    std::vector<SymbolicScalar> outputs = fn.casadi_function()(inputs);
    return SparsityPattern(
        SymbolicScalar::jacobian(outputs.at(output_idx), inputs.at(input_idx)).sparsity());
}

} // namespace detail

/**
 * @brief Cached sparse Jacobian evaluator with fixed structural ordering
 *
 * Values are returned in the CCS ordering from `sparsity().get_triplet()`.
 *
 * @see sparse_jacobian, SparsityPattern
 */
class SparseJacobianEvaluator {
  public:
    /// @brief Construct from a single symbolic expression and variable
    /// @param expression Output expression
    /// @param variables Input variables
    /// @param name Optional function name
    SparseJacobianEvaluator(const SymbolicArg &expression, const SymbolicArg &variables,
                            const std::string &name = "")
        : artifacts_(detail::make_sparse_jacobian_artifacts({expression}, {variables}, name)) {}

    /// @brief Construct from multiple symbolic expressions and variables
    /// @param expressions Output expressions
    /// @param variables Input variables
    /// @param name Optional function name
    SparseJacobianEvaluator(const std::vector<SymbolicArg> &expressions,
                            const std::vector<SymbolicArg> &variables, const std::string &name = "")
        : artifacts_(detail::make_sparse_jacobian_artifacts(expressions, variables, name)) {}

    /// @brief Construct from a metis::Function
    /// @param fn Function to differentiate
    /// @param output_idx Output index
    /// @param input_idx Input index
    /// @param name Optional function name
    SparseJacobianEvaluator(const Function &fn, int output_idx = 0, int input_idx = 0,
                            const std::string &name = "")
        : artifacts_(detail::make_sparse_jacobian_artifacts(fn, output_idx, input_idx, name)) {}

    /// @brief Get the Jacobian sparsity pattern
    /// @return Const reference to the sparsity pattern
    const SparsityPattern &sparsity() const { return artifacts_.sparsity; }

    /// @brief Number of structural non-zeros
    /// @return Non-zero count
    int nnz() const { return artifacts_.sparsity.nnz(); }

    /// @brief Get forward-mode graph coloring
    /// @return Const reference to forward coloring
    const GraphColoring &forward_coloring() const { return artifacts_.forward_coloring; }

    /// @brief Get reverse-mode graph coloring
    /// @return Const reference to reverse coloring
    const GraphColoring &reverse_coloring() const { return artifacts_.reverse_coloring; }

    /// @brief Determine the cheaper directional mode
    /// @return Forward or Reverse based on color count
    SparseJacobianMode preferred_mode() const {
        return forward_coloring().n_colors() <= reverse_coloring().n_colors()
                   ? SparseJacobianMode::Forward
                   : SparseJacobianMode::Reverse;
    }

    /// @brief Get the coloring for the preferred mode
    /// @return Const reference to the preferred coloring
    const GraphColoring &preferred_coloring() const {
        return preferred_mode() == SparseJacobianMode::Forward ? artifacts_.forward_coloring
                                                               : artifacts_.reverse_coloring;
    }

    /// @brief Access the compiled values function
    /// @return Const reference to the underlying Function
    const Function &values_function() const { return artifacts_.values_function; }

    /// @brief Evaluate and return Jacobian non-zero values
    /// @tparam Args Argument types
    /// @param args Function inputs
    /// @return Non-zero values in CCS order
    template <typename... Args> auto values(Args &&...args) const {
        return artifacts_.values_function.eval(std::forward<Args>(args)...);
    }

    /// @brief Shorthand for values()
    /// @tparam Args Argument types
    /// @param args Function inputs
    /// @return Non-zero values in CCS order
    template <typename... Args> auto operator()(Args &&...args) const {
        return values(std::forward<Args>(args)...);
    }

  private:
    detail::SparseJacobianArtifacts artifacts_;
};

/**
 * @brief Cached sparse Hessian evaluator with fixed structural ordering
 *
 * Values are returned in the CCS ordering from `sparsity().get_triplet()`.
 *
 * @see sparse_hessian, SparsityPattern
 */
class SparseHessianEvaluator {
  public:
    /// @brief Construct from a scalar expression and single variable block
    /// @param expression Scalar expression
    /// @param variables Input variables
    /// @param name Optional function name
    SparseHessianEvaluator(const SymbolicArg &expression, const SymbolicArg &variables,
                           const std::string &name = "")
        : artifacts_(detail::make_sparse_hessian_artifacts(expression, {variables}, name)) {}

    /// @brief Construct from a scalar expression and multiple variable blocks
    /// @param expression Scalar expression
    /// @param variables Input variable blocks
    /// @param name Optional function name
    SparseHessianEvaluator(const SymbolicArg &expression, const std::vector<SymbolicArg> &variables,
                           const std::string &name = "")
        : artifacts_(detail::make_sparse_hessian_artifacts(expression, variables, name)) {}

    /// @brief Construct from a metis::Function
    /// @param fn Function with scalar output
    /// @param output_idx Scalar output index
    /// @param input_idx Input index
    /// @param name Optional function name
    SparseHessianEvaluator(const Function &fn, int output_idx = 0, int input_idx = 0,
                           const std::string &name = "")
        : artifacts_(detail::make_sparse_hessian_artifacts(fn, output_idx, input_idx, name)) {}

    /// @brief Get the Hessian sparsity pattern
    /// @return Const reference to the sparsity pattern
    const SparsityPattern &sparsity() const { return artifacts_.sparsity; }

    /// @brief Number of structural non-zeros
    /// @return Non-zero count
    int nnz() const { return artifacts_.sparsity.nnz(); }

    /// @brief Get the star coloring for symmetric compression
    /// @return Const reference to the coloring
    const GraphColoring &coloring() const { return artifacts_.coloring; }

    /// @brief Access the compiled values function
    /// @return Const reference to the underlying Function
    const Function &values_function() const { return artifacts_.values_function; }

    /// @brief Evaluate and return Hessian non-zero values
    /// @tparam Args Argument types
    /// @param args Function inputs
    /// @return Non-zero values in CCS order
    template <typename... Args> auto values(Args &&...args) const {
        return artifacts_.values_function.eval(std::forward<Args>(args)...);
    }

    /// @brief Shorthand for values()
    /// @tparam Args Argument types
    /// @param args Function inputs
    /// @return Non-zero values in CCS order
    template <typename... Args> auto operator()(Args &&...args) const {
        return values(std::forward<Args>(args)...);
    }

  private:
    detail::SparseHessianArtifacts artifacts_;
};

// === Sparsity Query Functions ===

/**
 * @brief Get Jacobian sparsity without computing the full Jacobian
 *
 * This is more efficient than computing the Jacobian and extracting its sparsity.
 * Named sparsity_of_jacobian to avoid collision with casadi::jacobian_sparsity.
 *
 * @param expr Expression to differentiate (output)
 * @param vars Variables to differentiate with respect to (input)
 * @return SparsityPattern of the Jacobian
 */
inline SparsityPattern sparsity_of_jacobian(const SymbolicScalar &expr,
                                            const SymbolicScalar &vars) {
    casadi::Sparsity sp = casadi::MX::jacobian_sparsity(expr, vars);
    return SparsityPattern(sp);
}

/**
 * @brief Get Hessian sparsity of a scalar expression
 *
 * Named sparsity_of_hessian to avoid collision with potential CasADi functions.
 *
 * @param expr Scalar expression
 * @param vars Variables
 * @return SparsityPattern of the Hessian (symmetric)
 */
inline SparsityPattern sparsity_of_hessian(const SymbolicScalar &expr, const SymbolicScalar &vars) {
    // Hessian sparsity = Jacobian sparsity of the gradient
    auto grad = casadi::MX::gradient(expr, vars);
    casadi::Sparsity sp = casadi::MX::jacobian_sparsity(grad, vars);
    return SparsityPattern(sp);
}

/**
 * @brief Get sparsity of a metis::Function Jacobian
 *
 * @param fn The function
 * @param output_idx Output index (default 0)
 * @param input_idx Input index (default 0)
 * @return SparsityPattern of the function's Jacobian
 */
inline SparsityPattern get_jacobian_sparsity(const Function &fn, int output_idx = 0,
                                             int input_idx = 0) {
    return detail::function_jacobian_sparsity(fn, output_idx, input_idx);
}

/**
 * @brief Get Hessian sparsity of a scalar metis::Function output.
 *
 * @param fn The function
 * @param output_idx Scalar output index (default 0)
 * @param input_idx Input index to differentiate with respect to (default 0)
 * @return SparsityPattern of the Hessian
 */
inline SparsityPattern get_hessian_sparsity(const Function &fn, int output_idx = 0,
                                            int input_idx = 0) {
    detail::validate_function_indices(fn, output_idx, input_idx, "get_hessian_sparsity");
    std::vector<SymbolicScalar> inputs = detail::symbolic_inputs_like(fn, "_hsp_");
    std::vector<SymbolicScalar> outputs = fn.casadi_function()(inputs);
    if (!outputs.at(output_idx).is_scalar()) {
        throw InvalidArgument("get_hessian_sparsity: selected function output must be scalar");
    }
    return SparsityPattern(
        SymbolicScalar::hessian(outputs.at(output_idx), inputs.at(input_idx)).sparsity());
}

/**
 * @brief Get input sparsity of a metis::Function
 * @param fn The function
 * @param input_idx Input index (default 0)
 * @return SparsityPattern of the input slot
 */
inline SparsityPattern get_sparsity_in(const Function &fn, int input_idx = 0) {
    casadi::Sparsity sp = fn.casadi_function().sparsity_in(input_idx);
    return SparsityPattern(sp);
}

/**
 * @brief Get output sparsity of a metis::Function
 * @param fn The function
 * @param output_idx Output index (default 0)
 * @return SparsityPattern of the output slot
 */
inline SparsityPattern get_sparsity_out(const Function &fn, int output_idx = 0) {
    casadi::Sparsity sp = fn.casadi_function().sparsity_out(output_idx);
    return SparsityPattern(sp);
}

/**
 * @brief Compile a sparse Jacobian evaluator from symbolic expressions
 * @param expression Output expression
 * @param variables Input variables
 * @param name Optional function name
 * @return Configured evaluator
 */
inline SparseJacobianEvaluator sparse_jacobian(const SymbolicArg &expression,
                                               const SymbolicArg &variables,
                                               const std::string &name = "") {
    return SparseJacobianEvaluator(expression, variables, name);
}

/**
 * @brief Compile a sparse Jacobian evaluator from vector arguments
 * @param expressions Output expressions
 * @param variables Input variables
 * @param name Optional function name
 * @return Configured evaluator
 */
inline SparseJacobianEvaluator sparse_jacobian(const std::vector<SymbolicArg> &expressions,
                                               const std::vector<SymbolicArg> &variables,
                                               const std::string &name = "") {
    return SparseJacobianEvaluator(expressions, variables, name);
}

/**
 * @brief Compile a sparse Jacobian evaluator from a metis::Function
 * @param fn Function to differentiate
 * @param output_idx Output index
 * @param input_idx Input index
 * @param name Optional function name
 * @return Configured evaluator
 */
inline SparseJacobianEvaluator sparse_jacobian(const Function &fn, int output_idx = 0,
                                               int input_idx = 0, const std::string &name = "") {
    return SparseJacobianEvaluator(fn, output_idx, input_idx, name);
}

/**
 * @brief Compile a sparse Hessian evaluator from symbolic expressions
 * @param expression Scalar expression
 * @param variables Input variables
 * @param name Optional function name
 * @return Configured evaluator
 */
inline SparseHessianEvaluator sparse_hessian(const SymbolicArg &expression,
                                             const SymbolicArg &variables,
                                             const std::string &name = "") {
    return SparseHessianEvaluator(expression, variables, name);
}

/**
 * @brief Compile a sparse Hessian evaluator from vector arguments
 * @param expression Scalar expression
 * @param variables Input variable blocks
 * @param name Optional function name
 * @return Configured evaluator
 */
inline SparseHessianEvaluator sparse_hessian(const SymbolicArg &expression,
                                             const std::vector<SymbolicArg> &variables,
                                             const std::string &name = "") {
    return SparseHessianEvaluator(expression, variables, name);
}

/**
 * @brief Compile a sparse Hessian evaluator from a metis::Function
 * @param fn Function with scalar output
 * @param output_idx Scalar output index
 * @param input_idx Input index
 * @param name Optional function name
 * @return Configured evaluator
 */
inline SparseHessianEvaluator sparse_hessian(const Function &fn, int output_idx = 0,
                                             int input_idx = 0, const std::string &name = "") {
    return SparseHessianEvaluator(fn, output_idx, input_idx, name);
}

// =============================================================================
// NaN-Propagation Sparsity Detection
// =============================================================================

/**
 * @brief Options for NaN-propagation sparsity detection
 * @see nan_propagation_sparsity
 */
struct NaNSparsityOptions {
    NumericVector reference_point; ///< Point to evaluate at (default: zeros)
};

/**
 * @brief Detect Jacobian sparsity using NaN propagation
 *
 * This provides black-box sparsity detection for functions where symbolic
 * sparsity analysis is not available. The algorithm:
 *
 * 1. Evaluate f(x) at a reference point to get baseline outputs
 * 2. For each input i: set x[i] = NaN, evaluate f(x_perturbed)
 * 3. If output[j] becomes NaN, then df[j]/dx[i] ≠ 0 (structurally)
 *
 * @note This is a conservative estimate - it may report false positives
 *       if NaN propagates through unused code paths, but won't miss
 *       actual dependencies.
 *
 * @code
 * // Detect sparsity of element-wise function
 * auto sp = metis::nan_propagation_sparsity(
 *     [](const NumericVector& x) {
 *         NumericVector y(x.size());
 *         for (int i = 0; i < x.size(); ++i) y(i) = x(i) * x(i);
 *         return y;
 *     },
 *     3, 3);  // 3 inputs, 3 outputs
 *
 * EXPECT_EQ(sp.nnz(), 3);  // Diagonal pattern
 * @endcode
 *
 * @tparam Func Function type: (const NumericVector&) -> NumericVector
 * @param fn Function to analyze
 * @param n_inputs Number of scalar inputs
 * @param n_outputs Number of scalar outputs
 * @param opts Options (reference point, etc.)
 * @return SparsityPattern of the Jacobian (n_outputs x n_inputs)
 */
template <typename Func>
SparsityPattern nan_propagation_sparsity(Func &&fn, int n_inputs, int n_outputs,
                                         const NaNSparsityOptions &opts = {}) {
    // Use reference point or default to zeros
    NumericVector x0(n_inputs);
    if (opts.reference_point.size() == n_inputs) {
        x0 = opts.reference_point;
    } else {
        x0.setZero();
    }

    // Evaluate at reference point to verify the function works
    NumericVector y0 = fn(x0);
    if (y0.size() != n_outputs) {
        throw InvalidArgument("nan_propagation_sparsity: function returned " +
                              std::to_string(y0.size()) + " outputs, expected " +
                              std::to_string(n_outputs));
    }

    // Build sparsity pattern by probing each input with NaN
    std::vector<casadi_int> row_indices;
    std::vector<casadi_int> col_indices;

    for (int i = 0; i < n_inputs; ++i) {
        // Create perturbed input with NaN at position i
        NumericVector x_perturbed = x0;
        x_perturbed(i) = std::numeric_limits<double>::quiet_NaN();

        // Evaluate function
        NumericVector y_perturbed = fn(x_perturbed);

        // Check which outputs became NaN
        for (int j = 0; j < n_outputs; ++j) {
            if (std::isnan(y_perturbed(j))) {
                // Output j depends on input i
                row_indices.push_back(j);
                col_indices.push_back(i);
            }
        }
    }

    // Construct CasADi sparsity from triplets
    casadi::Sparsity sp = casadi::Sparsity::triplet(n_outputs, n_inputs, row_indices, col_indices);
    return SparsityPattern(sp);
}

/**
 * @brief Detect Jacobian sparsity of a metis::Function using NaN propagation
 *
 * Convenience overload that uses the Function's internal structure.
 *
 * @param fn Function to analyze (must have single vector input/output)
 * @param opts Options for NaN propagation
 * @return SparsityPattern of the Jacobian
 */
inline SparsityPattern nan_propagation_sparsity(const Function &fn,
                                                const NaNSparsityOptions &opts = {}) {
    // Get input/output dimensions from the function
    auto sp_in = fn.casadi_function().sparsity_in(0);
    auto sp_out = fn.casadi_function().sparsity_out(0);

    int n_inputs = static_cast<int>(sp_in.nnz());
    int n_outputs = static_cast<int>(sp_out.nnz());

    // Create wrapper that evaluates the function with a single vector input
    auto eval_fn = [&fn](const NumericVector &x) -> NumericVector {
        // Pass as a single Eigen vector (not as separate scalars)
        auto results = fn(x);
        // Flatten results to a single vector
        int total_size = 0;
        for (const auto &m : results) {
            total_size += static_cast<int>(m.size());
        }
        NumericVector y(total_size);
        int offset = 0;
        for (const auto &m : results) {
            for (int i = 0; i < m.size(); ++i) {
                y(offset + i) = m(i);
            }
            offset += static_cast<int>(m.size());
        }
        return y;
    };

    return nan_propagation_sparsity(eval_fn, n_inputs, n_outputs, opts);
}

} // namespace metis
