#pragma once
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

/**
 * @file MetisIO.hpp
 * @brief IO Utilities and Traits for Metis
 *
 * Provides:
 * 1. Helper functions for printing/displaying matrices with wrappers.
 * 2. Evaluation utilities (eval).
 * 3. Graph visualization utilities for CasADi expressions.
 *
 * Note: Eigen::NumTraits definitions are in MetisTypes.hpp usually.
 */

namespace metis {

// Forward declaration
class Function;

/**
 * @brief Print a matrix to stdout with a label
 * @tparam Derived Eigen expression type
 * @param label Descriptive label printed before the matrix
 * @param mat Matrix to display
 */
template <typename Derived>
void print(const std::string &label, const Eigen::MatrixBase<Derived> &mat) {
    std::cout << label << ":\n" << mat << "\n" << std::endl;
}

/**
 * @brief Deprecated alias for print
 * @tparam Derived Eigen expression type
 * @param label Descriptive label
 * @param mat Matrix to display
 * @see print
 */
template <typename Derived>
void disp(const std::string &label, const Eigen::MatrixBase<Derived> &mat) {
    print(label, mat);
}

/**
 * @brief Evaluate a symbolic matrix to a numeric Eigen matrix
 * @tparam Derived Eigen expression type
 * @param mat Symbolic or numeric matrix (must contain no free variables)
 * @return NumericMatrix with evaluated values
 */
template <typename Derived> auto eval(const Eigen::MatrixBase<Derived> &mat) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_same_v<Scalar, SymbolicScalar>) {
        // Flatten to MX, evaluate, map back
        SymbolicScalar flat = SymbolicScalar::zeros(mat.rows(), mat.cols());
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                flat(i, j) = mat(i, j);
            }
        }

        try {
            casadi::Function f("f", {}, {flat});
            auto res = f(std::vector<casadi::DM>{});
            casadi::DM res_dm = res[0];

            NumericMatrix res_eigen(mat.rows(), mat.cols());
            for (int i = 0; i < mat.rows(); ++i) {
                for (int j = 0; j < mat.cols(); ++j) {
                    res_eigen(i, j) = static_cast<double>(res_dm(i, j));
                }
            }
            return res_eigen;
        } catch (const std::exception &e) {
            throw RuntimeError("eval failed (expression contains free variables): " +
                               std::string(e.what()));
        }
    } else {
        return mat.eval();
    }
}

/// @brief Evaluate a symbolic scalar to a double
/// @param val Symbolic scalar (must be 1x1 and contain no free variables)
/// @return Numeric value
/// @throws RuntimeError if val is not 1x1 or contains free variables
inline double eval(const SymbolicScalar &val) {
    if (val.size1() != 1 || val.size2() != 1) {
        throw RuntimeError("eval scalar failed: expected 1x1 symbolic expression, got " +
                           std::to_string(val.size1()) + "x" + std::to_string(val.size2()));
    }
    try {
        casadi::Function f("f", {}, {val});
        auto res = f(std::vector<casadi::DM>{});
        casadi::DM res_dm = res[0];
        return static_cast<double>(res_dm);
    } catch (const std::exception &e) {
        throw RuntimeError("eval scalar failed: " + std::string(e.what()));
    }
}

/// @brief Passthrough eval for numeric types
/// @tparam T Arithmetic type
/// @param val Numeric value
/// @return Same value unchanged
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0> T eval(const T &val) {
    return val;
}

// ======================================================================
// Graph Visualization
// ======================================================================

namespace detail {

/**
 * @brief Escape special characters for DOT format
 */
inline std::string escape_dot_label(const std::string &s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '<':
            result += "&lt;";
            break;
        case '>':
            result += "&gt;";
            break;
        case '{':
        case '}':
            result += ' ';
            break;
        default:
            result += c;
        }
    }
    // Truncate long labels
    if (result.size() > 40) {
        result = result.substr(0, 37) + "...";
    }
    return result;
}

/**
 * @brief Get a short description of an MX operation type
 */
inline std::string get_op_name(const SymbolicScalar &mx) {
    if (mx.is_symbolic()) {
        std::ostringstream ss;
        mx.disp(ss, false);
        return ss.str();
    }
    if (mx.is_constant()) {
        std::ostringstream ss;
        mx.disp(ss, false);
        std::string s = ss.str();
        if (s.size() > 20)
            s = s.substr(0, 17) + "...";
        return s;
    }
    // Try to get operation name from string representation
    std::ostringstream ss;
    mx.disp(ss, false);
    std::string s = ss.str();

    // Common operations
    if (s.find("sq(") == 0)
        return "sq";
    if (s.find("sin(") == 0)
        return "sin";
    if (s.find("cos(") == 0)
        return "cos";
    if (s.find("exp(") == 0)
        return "exp";
    if (s.find("log(") == 0)
        return "log";
    if (s.find("sqrt(") == 0)
        return "sqrt";
    if (s.find("tanh(") == 0)
        return "tanh";

    // For binary ops, return truncated expression
    if (s.size() > 30)
        s = s.substr(0, 27) + "...";
    return s;
}

} // namespace detail

/**
 * @brief Export a symbolic expression to DOT format for visualization
 *
 * Creates a DOT file representing the computational graph of the expression.
 * Traverses the full expression tree showing all intermediate operations.
 *
 * @param expr The symbolic expression to visualize
 * @param filename Output filename (without extension, .dot will be added)
 * @param name Optional graph name for DOT file
 */
inline void export_graph_dot(const SymbolicScalar &expr, const std::string &filename,
                             const std::string &name = "expression") {
    // Get all free variables
    std::vector<SymbolicScalar> free_vars = SymbolicScalar::symvar(expr);

    // Build DOT file
    std::string dot_filename = filename + ".dot";
    std::ofstream out(dot_filename);
    if (!out.is_open()) {
        throw RuntimeError("Failed to open file for writing: " + dot_filename);
    }

    out << "digraph \"" << name << "\" {\n";
    out << "  rankdir=BT;\n"; // Bottom to top (inputs at bottom)
    out << "  splines=ortho;\n";
    out << "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];\n";
    out << "  edge [color=\"#666666\", arrowsize=0.7];\n\n";

    // Title
    out << "  labelloc=\"t\";\n";
    out << "  label=\"" << detail::escape_dot_label(name) << "\";\n";
    out << "  fontsize=16;\n\n";

    // Collect nodes by traversing the expression
    std::map<std::string, int> node_ids;    // ptr-based ID -> sequential ID
    std::vector<std::pair<int, int>> edges; // edges as sequential IDs
    std::set<const void *> visited;
    int node_counter = 0;

    // BFS traversal to collect all nodes
    std::vector<SymbolicScalar> queue = {expr};
    std::map<const void *, int> ptr_to_id;

    while (!queue.empty()) {
        SymbolicScalar current = queue.back();
        queue.pop_back();

        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);

        int current_id = node_counter++;
        ptr_to_id[ptr] = current_id;

        // Get dependencies
        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            SymbolicScalar dep = current.dep(i);
            queue.push_back(dep);
        }
    }

    // Second pass: generate nodes and edges
    visited.clear();
    queue = {expr};

    while (!queue.empty()) {
        SymbolicScalar current = queue.back();
        queue.pop_back();

        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);

        int current_id = ptr_to_id[ptr];
        std::string label = detail::get_op_name(current);

        // Determine node style based on type
        std::string color = "#87CEEB"; // light blue default
        std::string shape = "box";

        if (current.is_symbolic()) {
            color = "#90EE90"; // light green for inputs
            shape = "ellipse";
        } else if (current.is_constant()) {
            color = "#FFE4B5"; // moccasin for constants
            shape = "ellipse";
        } else if (current.n_dep() == 0) {
            color = "#DDA0DD"; // plum for leaf operations
        }

        out << "  node_" << current_id << " [label=\"" << detail::escape_dot_label(label)
            << "\", fillcolor=\"" << color << "\", shape=" << shape << "];\n";

        // Add edges from dependencies
        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            SymbolicScalar dep = current.dep(i);
            const void *dep_ptr = dep.get();
            int dep_id = ptr_to_id[dep_ptr];
            out << "  node_" << dep_id << " -> node_" << current_id << ";\n";
            queue.push_back(dep);
        }
    }

    // Mark output node specially
    if (!ptr_to_id.empty()) {
        const void *out_ptr = expr.get();
        int out_id = ptr_to_id[out_ptr];
        out << "\n  // Output marker\n";
        out << "  output [label=\"Output\", shape=doublecircle, fillcolor=\"#FFD700\"];\n";
        out << "  node_" << out_id << " -> output;\n";
    }

    out << "}\n";
    out.close();
}

/**
 * @brief Export a metis::Function to DOT format for visualization
 *
 * @param func The function to visualize (uses underlying CasADi function)
 * @param filename Output filename (without extension)
 */
// Note: This overload is implemented after Function class is defined
// Use the casadi::Function version directly for now

/**
 * @brief Render a DOT file to an image using Graphviz
 *
 * Requires Graphviz to be installed and `dot` command available in PATH.
 *
 * @param dot_file Path to the DOT file (with .dot extension)
 * @param output_file Path for output image (extension determines format: .pdf, .png, .svg)
 * @return true if rendering succeeded, false otherwise
 */
inline bool render_graph(const std::string &dot_file, const std::string &output_file) {
    // Determine output format from extension
    std::string format = "pdf"; // default
    size_t dot_pos = output_file.rfind('.');
    if (dot_pos != std::string::npos) {
        format = output_file.substr(dot_pos + 1);
    }

    // Build command: dot -Tformat input.dot -o output.ext
    std::string cmd = "dot -T" + format + " \"" + dot_file + "\" -o \"" + output_file + "\"";

    int result = std::system(cmd.c_str());
    return result == 0;
}

/**
 * @brief Convenience function: export expression to DOT and render to PDF
 *
 * Creates both a .dot file and a .pdf file with the given base name.
 *
 * @param expr The symbolic expression to visualize
 * @param output_base Base filename (creates output_base.dot and output_base.pdf)
 * @return true if both export and render succeeded
 */
inline bool visualize_graph(const SymbolicScalar &expr, const std::string &output_base) {
    try {
        export_graph_dot(expr, output_base);
        return render_graph(output_base + ".dot", output_base + ".pdf");
    } catch (const std::exception &) {
        return false;
    }
}

namespace detail {

/// @brief Escape a string for embedding in a JavaScript string literal
/// @param content Raw string
/// @return Escaped string safe for use inside a \<script\> tag
inline std::string escape_for_js(const std::string &content) {
    std::string escaped;
    for (char c : content) {
        if (c == '\\')
            escaped += "\\\\";
        else if (c == '"')
            escaped += "\\\"";
        else if (c == '\n')
            escaped += "\\n";
        else if (c == '\r')
            escaped += "\\r";
        else if (c == '<')
            escaped += "\\u003C";
        else
            escaped += c;
    }
    return escaped;
}

/// @brief Escape a string for embedding in JSON
/// @param s Raw string
/// @return Escaped string safe for use inside a \<script\> tag
inline std::string escape_for_json(const std::string &s) {
    std::string result;
    for (char c : s) {
        if (c == '"')
            result += "\\\"";
        else if (c == '\\')
            result += "\\\\";
        else if (c == '\n')
            result += "\\n";
        else if (c == '\r')
            result += "\\r";
        else if (c == '\t')
            result += "\\t";
        else if (c == '<')
            result += "\\u003C";
        else
            result += c;
    }
    return result;
}

/// @brief Write a complete interactive graph HTML page
/// @param out Output stream
/// @param title Page title
/// @param escaped_dot DOT source (JS-escaped)
/// @param node_data_json Node metadata as JSON
/// @param edges_json Edge list as JSON
/// @param extra_header_js Optional JS inserted after data declarations
inline void write_graph_html(std::ostream &out, const std::string &title,
                             const std::string &escaped_dot, const std::string &node_data_json,
                             const std::string &edges_json,
                             const std::string &extra_header_js = "") {
    out << R"HTMLSTART(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>)HTMLSTART"
        << title << R"HTMLMID(</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/viz.js/2.1.2/viz.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/viz.js/2.1.2/full.render.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { height: 100%; width: 100%; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; display: flex; }
        #controls { position: fixed; top: 10px; left: 10px; z-index: 100; background: rgba(0,0,0,0.8);
                    padding: 12px; border-radius: 8px; }
        #controls button { margin: 2px; padding: 8px 14px; cursor: pointer; border: none;
                           border-radius: 4px; background: #4a4a6a; color: white; font-size: 13px; }
        #controls button:hover { background: #6a6a8a; }
        #graph { flex: 1; cursor: grab; overflow: hidden; height: 100%; }
        #graph:active { cursor: grabbing; }
        #graph svg { display: block; }
        #sidebar { width: 320px; height: 100%; background: #16213e; padding: 16px; overflow-y: auto;
                   border-left: 2px solid #0f3460; }
        #sidebar h2 { color: #e94560; margin-bottom: 12px; font-size: 16px; }
        #sidebar .section { margin-bottom: 16px; }
        #sidebar .label { color: #888; font-size: 11px; text-transform: uppercase; margin-bottom: 4px; }
        #sidebar .value { background: #0f3460; padding: 10px; border-radius: 6px; font-family: monospace;
                          font-size: 13px; word-break: break-all; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
        #sidebar .type-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px;
                               margin-left: 8px; }
        .type-input { background: #90EE90; color: #000; }
        .type-constant { background: #FFE4B5; color: #000; }
        .type-operation { background: #87CEEB; color: #000; }
        .type-leaf { background: #DDA0DD; color: #000; }
        #info { position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8);
                padding: 10px; border-radius: 4px; font-size: 12px; }
        #stats { position: fixed; top: 10px; right: 340px; background: rgba(0,0,0,0.8);
                 padding: 10px; border-radius: 4px; font-size: 12px; }
        .node-highlighted polygon, .node-highlighted ellipse, .node-highlighted path { stroke: #e94560 !important; stroke-width: 3px !important; }
        .edge-highlighted path { stroke: #e94560 !important; stroke-width: 2px !important; }
        .edge-highlighted polygon { stroke: #e94560 !important; fill: #e94560 !important; }
        svg .node { cursor: pointer; }
        svg .node:hover polygon, svg .node:hover ellipse { stroke: #fff !important; stroke-width: 2px !important; }
    </style>
</head>
<body>
    <div id="controls">
        <button onclick="zoomIn()">Zoom +</button>
        <button onclick="zoomOut()">Zoom -</button>
        <button onclick="resetView()">Reset</button>
        <button onclick="fitToScreen()">Fit</button>
    </div>
    <div id="graph"></div>
    <div id="sidebar">
        <h2>Node Info</h2>
        <div id="node-info">
            <p style="color:#666; font-style:italic;">Click on a node to see details</p>
        </div>
    </div>
    <div id="info">Scroll to zoom - Drag to pan - Click nodes for details</div>
    <div id="stats"></div>
    <script>
        const dotSrc = ")HTMLMID"
        << escaped_dot << R"HTMLMID2(";
        const nodeData = )HTMLMID2"
        << node_data_json << R"HTMLMID3(;
        const edges = )HTMLMID3"
        << edges_json << R"HTMLEND(;
)HTMLEND";

    // Insert optional extra JS (e.g. stats computation for SX graphs)
    if (!extra_header_js.empty()) {
        out << "        " << extra_header_js << "\n";
    }

    out << R"HTMLEND2(
        let scale = 1, panX = 0, panY = 0, isDragging = false, startX, startY;
        let selectedNode = null;
        const container = document.getElementById('graph');
        const sidebar = document.getElementById('node-info');

        new Viz().renderSVGElement(dotSrc).then(svg => {
            container.appendChild(svg);
            svg.style.transformOrigin = '0 0';
            fitToScreen();
            setupPanZoom(svg);
            setupNodeInteraction(svg);
        });

        function updateTransform(svg) {
            svg.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        }
        function zoomIn() { scale *= 1.3; updateTransform(container.querySelector('svg')); }
        function zoomOut() { scale /= 1.3; updateTransform(container.querySelector('svg')); }
        function resetView() { scale = 1; panX = 0; panY = 0; updateTransform(container.querySelector('svg')); }
        function fitToScreen() {
            const svg = container.querySelector('svg');
            if (!svg) return;
            const bbox = svg.getBBox();
            const availWidth = window.innerWidth - 320;
            const scaleX = (availWidth - 40) / (bbox.width + 40);
            const scaleY = (window.innerHeight - 40) / (bbox.height + 40);
            scale = Math.min(scaleX, scaleY);
            panX = (availWidth - bbox.width * scale) / 2;
            panY = (window.innerHeight - bbox.height * scale) / 2;
            updateTransform(svg);
        }

        function setupPanZoom(svg) {
            container.addEventListener('wheel', e => {
                e.preventDefault();
                const rect = container.getBoundingClientRect();
                const mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
                const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
                panX = mouseX - (mouseX - panX) * zoomFactor;
                panY = mouseY - (mouseY - panY) * zoomFactor;
                scale *= zoomFactor;
                updateTransform(svg);
            });
            container.addEventListener('mousedown', e => {
                if (e.target.closest('.node')) return;
                isDragging = true; startX = e.clientX - panX; startY = e.clientY - panY;
            });
            container.addEventListener('mousemove', e => { if (isDragging) { panX = e.clientX - startX; panY = e.clientY - startY; updateTransform(svg); } });
            container.addEventListener('mouseup', () => isDragging = false);
            container.addEventListener('mouseleave', () => isDragging = false);
        }

        function setupNodeInteraction(svg) {
            const nodes = svg.querySelectorAll('.node');
            nodes.forEach(node => {
                const nodeId = node.id;
                node.addEventListener('click', e => {
                    e.stopPropagation();
                    selectNode(svg, nodeId);
                });
            });
        }

        function selectNode(svg, nodeId) {
            svg.querySelectorAll('.node-highlighted').forEach(n => n.classList.remove('node-highlighted'));
            svg.querySelectorAll('.edge-highlighted').forEach(e => e.classList.remove('edge-highlighted'));

            const node = svg.getElementById(nodeId);
            if (!node) return;

            node.classList.add('node-highlighted');
            selectedNode = nodeId;

            const nodeNum = parseInt(nodeId.replace('node_', '').replace('output_', '-'));
            edges.forEach(([from, to]) => {
                if (from === nodeNum || to === nodeNum || to === -nodeNum - 1) {
                    svg.querySelectorAll('.edge').forEach(edge => {
                        const title = edge.querySelector('title');
                        if (title) {
                            const edgeStr = title.textContent;
                            const toId = to < 0 ? `output_${-to - 1}` : `node_${to}`;
                            if (edgeStr.includes(`node_${from}`) && edgeStr.includes(toId)) {
                                edge.classList.add('edge-highlighted');
                            }
                        }
                    });
                }
            });

            const data = nodeData[nodeId];
            if (data) {
                const nodeLabel = data.short || data.label || '';
                const fullExpr = data.full || data.label || '';
                sidebar.innerHTML = `
                    <div class="section">
                        <div class="label">Node ID</div>
                        <div class="value">${data.id} <span class="type-badge type-${data.type}">${data.type}</span></div>
                    </div>
                    <div class="section">
                        <div class="label">Label</div>
                        <div class="value">${escapeHtml(nodeLabel)}</div>
                    </div>
                    ${fullExpr !== nodeLabel ? `<div class="section">
                        <div class="label">Full Expression</div>
                        <div class="value">${escapeHtml(fullExpr)}</div>
                    </div>` : ''}
                    <div class="section">
                        <div class="label">Dependencies (${data.deps.length})</div>
                        <div class="value">${data.deps.length > 0 ? data.deps.map(d => 'node_' + d).join(', ') : 'None'}</div>
                    </div>
                `;
            } else if (nodeId === 'output' || nodeId.startsWith('output_')) {
                const outIdx = nodeId.replace('output_', '').replace('output', '0');
                sidebar.innerHTML = `
                    <div class="section">
                        <div class="label">Node</div>
                        <div class="value">Output[${outIdx}] <span class="type-badge" style="background:#FFD700;color:#000;">output</span></div>
                    </div>
                    <div class="section">
                        <div class="label">Description</div>
                        <div class="value">Output element ${outIdx} of the expression</div>
                    </div>
                `;
            }
        }

        function escapeHtml(str) {
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }
    </script>
</body>
</html>
)HTMLEND2";
}

} // namespace detail

/**
 * @brief Export a symbolic expression to an interactive HTML file
 *
 * Creates a self-contained HTML file with embedded Viz.js for rendering
 * the computational graph. Supports pan and zoom via mouse interaction.
 *
 * @param expr The symbolic expression to visualize
 * @param filename Output filename (without extension, .html will be added)
 * @param name Optional graph name
 */
inline void export_graph_html(const SymbolicScalar &expr, const std::string &filename,
                              const std::string &name = "expression") {
    // First generate the DOT content and collect node metadata
    std::ostringstream dot_stream;
    std::ostringstream node_data_stream; // JSON for node metadata

    // Get all free variables
    std::vector<SymbolicScalar> free_vars = SymbolicScalar::symvar(expr);

    dot_stream << "digraph \"" << name << "\" {\n";
    dot_stream << "  rankdir=BT;\n";
    dot_stream << "  splines=ortho;\n";
    dot_stream << "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];\n";
    dot_stream << "  edge [color=\"#666666\", arrowsize=0.7];\n\n";
    dot_stream << "  labelloc=\"t\";\n";
    dot_stream << "  label=\"" << detail::escape_dot_label(name) << "\";\n";
    dot_stream << "  fontsize=16;\n\n";

    // Collect nodes
    std::set<const void *> visited;
    std::map<const void *, int> ptr_to_id;
    int node_counter = 0;

    std::vector<SymbolicScalar> queue = {expr};
    while (!queue.empty()) {
        SymbolicScalar current = queue.back();
        queue.pop_back();

        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);
        ptr_to_id[ptr] = node_counter++;

        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            queue.push_back(current.dep(i));
        }
    }

    // Build node data JSON and edges JSON
    node_data_stream << "{";
    std::ostringstream edges_stream;
    edges_stream << "[";
    bool first_node = true;
    bool first_edge = true;

    // Second pass: generate nodes and edges
    visited.clear();
    queue = {expr};
    while (!queue.empty()) {
        SymbolicScalar current = queue.back();
        queue.pop_back();

        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);

        int current_id = ptr_to_id[ptr];

        // Get full expression string (no truncation)
        std::ostringstream full_expr;
        current.disp(full_expr, false);
        std::string full_label = full_expr.str();

        // Get short label for display
        std::string short_label = detail::get_op_name(current);
        std::string node_type = "operation";
        std::string color = "#87CEEB";
        std::string shape = "box";

        if (current.is_symbolic()) {
            color = "#90EE90";
            shape = "ellipse";
            node_type = "input";
        } else if (current.is_constant()) {
            color = "#FFE4B5";
            shape = "ellipse";
            node_type = "constant";
        } else if (current.n_dep() == 0) {
            color = "#DDA0DD";
            node_type = "leaf";
        }

        // Add to node data JSON
        if (!first_node)
            node_data_stream << ",";
        first_node = false;
        node_data_stream << "\"node_" << current_id << "\":{";
        node_data_stream << "\"id\":" << current_id << ",";
        node_data_stream << "\"short\":\"" << detail::escape_for_json(short_label) << "\",";
        node_data_stream << "\"full\":\"" << detail::escape_for_json(full_label) << "\",";
        node_data_stream << "\"type\":\"" << node_type << "\",";
        node_data_stream << "\"deps\":[";

        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            if (i > 0)
                node_data_stream << ",";
            SymbolicScalar dep = current.dep(i);
            node_data_stream << ptr_to_id[dep.get()];
        }
        node_data_stream << "]}";

        dot_stream << "  node_" << current_id << " [label=\""
                   << detail::escape_dot_label(short_label) << "\", fillcolor=\"" << color
                   << "\", shape=" << shape << ", id=\"node_" << current_id << "\"];\n";

        for (casadi_int i = 0; i < n; ++i) {
            SymbolicScalar dep = current.dep(i);
            const void *dep_ptr = dep.get();
            int dep_id = ptr_to_id[dep_ptr];
            dot_stream << "  node_" << dep_id << " -> node_" << current_id << ";\n";

            if (!first_edge)
                edges_stream << ",";
            first_edge = false;
            edges_stream << "[" << dep_id << "," << current_id << "]";

            queue.push_back(dep);
        }
    }

    // Mark output
    if (!ptr_to_id.empty()) {
        const void *out_ptr = expr.get();
        int out_id = ptr_to_id[out_ptr];
        dot_stream << "\n  output [label=\"Output\", shape=doublecircle, fillcolor=\"#FFD700\", "
                      "id=\"output\"];\n";
        dot_stream << "  node_" << out_id << " -> output;\n";

        if (!first_edge)
            edges_stream << ",";
        edges_stream << "[" << out_id << ",-1]";
    }
    dot_stream << "}\n";
    node_data_stream << "}";
    edges_stream << "]";

    std::string dot_content = dot_stream.str();

    // Write HTML file
    std::string html_filename = filename + ".html";
    std::ofstream out(html_filename);
    if (!out.is_open()) {
        throw RuntimeError("Failed to open file for writing: " + html_filename);
    }

    detail::write_graph_html(out, detail::escape_dot_label(name),
                             detail::escape_for_js(dot_content), node_data_stream.str(),
                             edges_stream.str());
    out.close();
}

// ======================================================================
// Deep Graph Visualization (SX-based)
// ======================================================================

namespace detail {

/**
 * @brief Map CasADi SX operation codes to readable labels
 *
 * Uses casadi_math.hpp operation codes. Returns a short human-readable
 * string for each operation type.
 */
/// Map a CasADi operation code to a short display symbol.
/// Shared by the symbolic graph exporters and the numeric eval-algorithm panel
/// so both label identical operations identically.
/// CasADi operation codes are defined in casadi/core/calculus.hpp.
inline std::string op_symbol(casadi_int op) {
    switch (op) {
    // Unary operations
    case casadi::OP_ASSIGN:
        return "=";
    case casadi::OP_NEG:
        return "neg";
    case casadi::OP_NOT:
        return "not";
    case casadi::OP_SQ:
        return "sq";
    case casadi::OP_SQRT:
        return "sqrt";
    case casadi::OP_EXP:
        return "exp";
    case casadi::OP_LOG:
        return "log";
    case casadi::OP_SIN:
        return "sin";
    case casadi::OP_COS:
        return "cos";
    case casadi::OP_TAN:
        return "tan";
    case casadi::OP_ASIN:
        return "asin";
    case casadi::OP_ACOS:
        return "acos";
    case casadi::OP_ATAN:
        return "atan";
    case casadi::OP_SINH:
        return "sinh";
    case casadi::OP_COSH:
        return "cosh";
    case casadi::OP_TANH:
        return "tanh";
    case casadi::OP_ASINH:
        return "asinh";
    case casadi::OP_ACOSH:
        return "acosh";
    case casadi::OP_ATANH:
        return "atanh";
    case casadi::OP_FABS:
        return "abs";
    case casadi::OP_FLOOR:
        return "floor";
    case casadi::OP_CEIL:
        return "ceil";
    case casadi::OP_SIGN:
        return "sign";
    case casadi::OP_ERF:
        return "erf";
    case casadi::OP_ERFINV:
        return "erfinv";
    case casadi::OP_INV:
        return "inv";

    // Binary operations
    case casadi::OP_ADD:
        return "+";
    case casadi::OP_SUB:
        return "-";
    case casadi::OP_MUL:
        return "*";
    case casadi::OP_DIV:
        return "/";
    case casadi::OP_POW:
        return "pow";
    case casadi::OP_ATAN2:
        return "atan2";
    case casadi::OP_FMIN:
        return "min";
    case casadi::OP_FMAX:
        return "max";
    case casadi::OP_FMOD:
        return "mod";
    case casadi::OP_COPYSIGN:
        return "copysign";
    case casadi::OP_HYPOT:
        return "hypot";

    // Comparison operations
    case casadi::OP_LT:
        return "<";
    case casadi::OP_LE:
        return "<=";
    case casadi::OP_EQ:
        return "==";
    case casadi::OP_NE:
        return "!=";
    case casadi::OP_AND:
        return "&&";
    case casadi::OP_OR:
        return "||";

    // Conditional
    case casadi::OP_IF_ELSE_ZERO:
        return "if_else_zero";

    default:
        return "op" + std::to_string(op);
    }
}

inline std::string get_sx_operation(const casadi::SXElem &elem) {
    if (elem.is_symbolic()) {
        return elem.name();
    }
    if (elem.is_constant()) {
        double val = static_cast<double>(elem);
        if (val == 0.0)
            return "0";
        if (val == 1.0)
            return "1";
        if (val == -1.0)
            return "-1";
        std::ostringstream oss;
        oss << std::setprecision(4) << val;
        return oss.str();
    }
    if (!elem.is_leaf()) {
        return op_symbol(elem.op());
    }
    return "?";
}

/**
 * @brief Get node styling based on SX element type
 */
inline void get_sx_node_style(const casadi::SXElem &elem, std::string &color, std::string &shape) {
    if (elem.is_symbolic()) {
        color = "#90EE90"; // light green for inputs
        shape = "ellipse";
    } else if (elem.is_constant()) {
        color = "#FFE4B5"; // moccasin for constants
        shape = "ellipse";
    } else if (!elem.is_leaf()) {
        // Non-leaf: operation node
        casadi_int op = elem.op();
        // Color by operation category
        if (op == casadi::OP_ADD || op == casadi::OP_SUB || op == casadi::OP_MUL ||
            op == casadi::OP_DIV || op == casadi::OP_NEG) {
            color = "#87CEEB"; // light blue for arithmetic
        } else if (op == casadi::OP_SIN || op == casadi::OP_COS || op == casadi::OP_TAN ||
                   op == casadi::OP_ASIN || op == casadi::OP_ACOS || op == casadi::OP_ATAN) {
            color = "#DDA0DD"; // plum for trig
        } else if (op == casadi::OP_EXP || op == casadi::OP_LOG || op == casadi::OP_SQRT ||
                   op == casadi::OP_SQ || op == casadi::OP_POW) {
            color = "#FFB6C1"; // light pink for power/exp
        } else if (op == casadi::OP_LT || op == casadi::OP_LE || op == casadi::OP_EQ ||
                   op == casadi::OP_NE || op == casadi::OP_AND || op == casadi::OP_OR) {
            color = "#98FB98"; // pale green for comparison
        } else {
            color = "#B0C4DE"; // light steel blue for other
        }
        shape = "box";
    } else {
        color = "#D3D3D3"; // light gray for unknown
        shape = "box";
    }
}

} // namespace detail

/**
 * @brief Visual theme for deep (SX) computational-graph exports.
 *
 * Defaults reproduce the historical Metis look (`classic()`); `conceptual()`
 * produces a clean left-to-right "blueprint" style with white nodes, orange
 * operation borders and subtle gray edges, matching slide-style diagrams.
 *
 * Colors are hex strings; an empty string omits that attribute (so the
 * Graphviz default applies).
 */
struct GraphStyle {
    std::string rankdir = "BT";
    std::string splines = "ortho";
    std::string bgcolor = ""; // empty -> transparent / Graphviz default
    std::string fontname = "Helvetica";
    std::string node_style = "rounded,filled";
    std::string edge_color = "#666666";
    double edge_arrowsize = 0.7;

    // When false, operation nodes are colored by category (classic behavior).
    // When true, every operation node uses op_fill/op_border/op_fontcolor.
    bool uniform_op_color = false;

    std::string input_fill = "#90EE90", input_border = "", input_fontcolor = "",
                input_shape = "ellipse";
    std::string const_fill = "#FFE4B5", const_border = "", const_fontcolor = "",
                const_shape = "ellipse";
    std::string op_fill = "#87CEEB", op_border = "", op_fontcolor = "", op_shape = "box";

    bool show_output_marker = true;
    std::string output_fill = "#FFD700", output_border = "", output_fontcolor = "",
                output_shape = "doublecircle";

    static GraphStyle classic() { return GraphStyle{}; }

    static GraphStyle conceptual() {
        GraphStyle s;
        s.rankdir = "LR";
        s.splines = "spline";
        s.bgcolor = "#FAFCFE";
        s.node_style = "filled";
        s.edge_color = "#9AA7B4";
        s.edge_arrowsize = 0.8;
        s.uniform_op_color = true;
        s.input_fill = "#FFFFFF";
        s.input_border = "#5F7E9B";
        s.input_fontcolor = "#33495C";
        s.input_shape = "box";
        s.const_fill = "#FFFFFF";
        s.const_border = "#5F7E9B";
        s.const_fontcolor = "#33495C";
        s.const_shape = "box";
        s.op_fill = "#FFFFFF";
        s.op_border = "#E08A1E";
        s.op_fontcolor = "#D17A12";
        s.op_shape = "box";
        s.output_fill = "#FFFFFF";
        s.output_border = "#E08A1E";
        s.output_fontcolor = "#D17A12";
        s.output_shape = "box";
        return s;
    }
};

namespace detail {

/// Build the per-node attribute list, omitting border/fontcolor when empty.
inline std::string node_attrs(const std::string &fill, const std::string &border,
                              const std::string &fontcolor) {
    std::string a = "fillcolor=\"" + fill + "\"";
    if (!border.empty())
        a += ", color=\"" + border + "\"";
    if (!fontcolor.empty())
        a += ", fontcolor=\"" + fontcolor + "\"";
    return a;
}

/// Theme-aware SX node styling. Falls back to the classic per-category colors
/// when the style does not request a uniform operation color.
inline void get_sx_node_style_themed(const casadi::SXElem &elem, const GraphStyle &style,
                                     std::string &fill, std::string &border, std::string &fontcolor,
                                     std::string &shape) {
    if (elem.is_symbolic()) {
        fill = style.input_fill;
        border = style.input_border;
        fontcolor = style.input_fontcolor;
        shape = style.input_shape;
    } else if (elem.is_constant()) {
        fill = style.const_fill;
        border = style.const_border;
        fontcolor = style.const_fontcolor;
        shape = style.const_shape;
    } else if (style.uniform_op_color) {
        fill = style.op_fill;
        border = style.op_border;
        fontcolor = style.op_fontcolor;
        shape = style.op_shape;
    } else {
        // Classic: color operations by category.
        get_sx_node_style(elem, fill, shape);
        border = "";
        fontcolor = "";
    }
}

} // namespace detail

/**
 * @brief Export an SX expression to DOT format for deep visualization
 *
 * Traverses the full SX expression tree showing all primitive operations.
 * Unlike MX graphs, this shows the fully-expanded computational graph.
 *
 * @param expr The SX expression to visualize
 * @param filename Output filename (without extension, .dot will be added)
 * @param name Optional graph name for DOT file
 */
inline void export_sx_graph_dot(const casadi::SX &expr, const std::string &filename,
                                const std::string &name = "expression",
                                const GraphStyle &style = GraphStyle::classic()) {
    std::string dot_filename = filename + ".dot";
    std::ofstream out(dot_filename);
    if (!out.is_open()) {
        throw RuntimeError("Failed to open file for writing: " + dot_filename);
    }

    out << "digraph \"" << name << "\" {\n";
    out << "  rankdir=" << style.rankdir << ";\n";
    if (!style.splines.empty())
        out << "  splines=" << style.splines << ";\n";
    if (!style.bgcolor.empty())
        out << "  bgcolor=\"" << style.bgcolor << "\";\n";
    out << "  node [style=\"" << style.node_style << "\", fontname=\"" << style.fontname
        << "\"];\n";
    out << "  edge [color=\"" << style.edge_color << "\", arrowsize=" << style.edge_arrowsize
        << "];\n\n";
    out << "  labelloc=\"t\";\n";
    out << "  label=\"" << detail::escape_dot_label(name) << "\";\n";
    out << "  fontsize=16;\n\n";

    // Traverse all elements in the SX matrix
    std::map<const void *, int> ptr_to_id;
    std::set<const void *> visited;
    std::vector<casadi::SXElem> queue;
    int node_counter = 0;

    // Get nonzeros as SXElem vector
    const std::vector<casadi::SXElem> &nz = expr.nonzeros();
    casadi_int n_elem = static_cast<casadi_int>(nz.size());

    // Collect all output elements
    for (casadi_int i = 0; i < n_elem; ++i) {
        queue.push_back(nz[i]);
    }

    // First pass: assign IDs to all nodes
    size_t queue_idx = 0;
    while (queue_idx < queue.size()) {
        casadi::SXElem current = queue[queue_idx++];
        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);
        ptr_to_id[ptr] = node_counter++;

        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            queue.push_back(current.dep(i));
        }
    }

    // Second pass: generate nodes and edges
    visited.clear();
    queue.clear();
    for (casadi_int i = 0; i < n_elem; ++i) {
        queue.push_back(nz[i]);
    }

    std::vector<int> output_node_ids;
    for (casadi_int i = 0; i < n_elem; ++i) {
        const void *ptr = nz[i].get();
        output_node_ids.push_back(ptr_to_id[ptr]);
    }

    queue_idx = 0;
    while (queue_idx < queue.size()) {
        casadi::SXElem current = queue[queue_idx++];
        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);

        int current_id = ptr_to_id[ptr];
        std::string label = detail::get_sx_operation(current);
        std::string fill, border, fontcolor, shape;
        detail::get_sx_node_style_themed(current, style, fill, border, fontcolor, shape);

        out << "  node_" << current_id << " [label=\"" << detail::escape_dot_label(label) << "\", "
            << detail::node_attrs(fill, border, fontcolor) << ", shape=" << shape << "];\n";

        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            casadi::SXElem dep = current.dep(i);
            const void *dep_ptr = dep.get();
            int dep_id = ptr_to_id[dep_ptr];
            out << "  node_" << dep_id << " -> node_" << current_id << ";\n";
            queue.push_back(dep);
        }
    }

    // Mark output nodes
    if (style.show_output_marker && !output_node_ids.empty()) {
        out << "\n  // Output markers\n";
        bool single = output_node_ids.size() == 1;
        for (size_t i = 0; i < output_node_ids.size(); ++i) {
            std::string olabel =
                single ? detail::escape_dot_label(name) : ("out[" + std::to_string(i) + "]");
            out << "  output_" << i << " [label=\"" << olabel << "\", "
                << detail::node_attrs(style.output_fill, style.output_border,
                                      style.output_fontcolor)
                << ", shape=" << style.output_shape << "];\n";
            out << "  node_" << output_node_ids[i] << " -> output_" << i << ";\n";
        }
    }

    out << "}\n";
    out.close();
}

/**
 * @brief Export an SX expression to an interactive HTML file for deep visualization
 *
 * Creates a self-contained HTML file with embedded Viz.js for rendering
 * the fully-expanded computational graph.
 *
 * @param expr The SX expression to visualize
 * @param filename Output filename (without extension, .html will be added)
 * @param name Optional graph name
 */
inline void export_sx_graph_html(const casadi::SX &expr, const std::string &filename,
                                 const std::string &name = "expression",
                                 const GraphStyle &style = GraphStyle::classic()) {
    std::ostringstream dot_stream;
    std::ostringstream node_data_stream;
    std::ostringstream edges_stream;

    dot_stream << "digraph \"" << name << "\" {\n";
    dot_stream << "  rankdir=" << style.rankdir << ";\n";
    if (!style.splines.empty())
        dot_stream << "  splines=" << style.splines << ";\n";
    if (!style.bgcolor.empty())
        dot_stream << "  bgcolor=\"" << style.bgcolor << "\";\n";
    dot_stream << "  node [style=\"" << style.node_style << "\", fontname=\"" << style.fontname
               << "\"];\n";
    dot_stream << "  edge [color=\"" << style.edge_color << "\", arrowsize=" << style.edge_arrowsize
               << "];\n\n";
    dot_stream << "  labelloc=\"t\";\n";
    dot_stream << "  label=\"" << detail::escape_dot_label(name) << "\";\n";
    dot_stream << "  fontsize=16;\n\n";

    // Traverse all elements
    std::map<const void *, int> ptr_to_id;
    std::set<const void *> visited;
    std::vector<casadi::SXElem> queue;
    int node_counter = 0;

    // Get nonzeros as SXElem vector
    const std::vector<casadi::SXElem> &nz = expr.nonzeros();
    casadi_int n_elem = static_cast<casadi_int>(nz.size());

    for (casadi_int i = 0; i < n_elem; ++i) {
        queue.push_back(nz[i]);
    }

    // First pass: assign IDs
    size_t queue_idx = 0;
    while (queue_idx < queue.size()) {
        casadi::SXElem current = queue[queue_idx++];
        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);
        ptr_to_id[ptr] = node_counter++;

        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            queue.push_back(current.dep(i));
        }
    }

    // Collect output node IDs
    std::vector<int> output_node_ids;
    for (casadi_int i = 0; i < n_elem; ++i) {
        const void *ptr = nz[i].get();
        output_node_ids.push_back(ptr_to_id[ptr]);
    }

    // Second pass: generate nodes and edges
    visited.clear();
    queue.clear();
    for (casadi_int i = 0; i < n_elem; ++i) {
        queue.push_back(nz[i]);
    }

    node_data_stream << "{";
    edges_stream << "[";
    bool first_node = true;
    bool first_edge = true;

    queue_idx = 0;
    while (queue_idx < queue.size()) {
        casadi::SXElem current = queue[queue_idx++];
        const void *ptr = current.get();
        if (visited.count(ptr))
            continue;
        visited.insert(ptr);

        int current_id = ptr_to_id[ptr];
        std::string label = detail::get_sx_operation(current);
        std::string fill, border, fontcolor, shape;
        detail::get_sx_node_style_themed(current, style, fill, border, fontcolor, shape);

        // Determine node type
        std::string node_type = "operation";
        if (current.is_symbolic())
            node_type = "input";
        else if (current.is_constant())
            node_type = "constant";

        // Add to node data JSON
        if (!first_node)
            node_data_stream << ",";
        first_node = false;
        node_data_stream << "\"node_" << current_id << "\":{";
        node_data_stream << "\"id\":" << current_id << ",";
        node_data_stream << "\"label\":\"" << detail::escape_for_json(label) << "\",";
        node_data_stream << "\"type\":\"" << node_type << "\",";
        node_data_stream << "\"deps\":[";

        casadi_int n = current.n_dep();
        for (casadi_int i = 0; i < n; ++i) {
            if (i > 0)
                node_data_stream << ",";
            casadi::SXElem dep = current.dep(i);
            node_data_stream << ptr_to_id[dep.get()];
        }
        node_data_stream << "]}";

        dot_stream << "  node_" << current_id << " [label=\"" << detail::escape_dot_label(label)
                   << "\", " << detail::node_attrs(fill, border, fontcolor) << ", shape=" << shape
                   << ", id=\"node_" << current_id << "\"];\n";

        for (casadi_int i = 0; i < n; ++i) {
            casadi::SXElem dep = current.dep(i);
            const void *dep_ptr = dep.get();
            int dep_id = ptr_to_id[dep_ptr];
            dot_stream << "  node_" << dep_id << " -> node_" << current_id << ";\n";

            if (!first_edge)
                edges_stream << ",";
            first_edge = false;
            edges_stream << "[" << dep_id << "," << current_id << "]";

            queue.push_back(dep);
        }
    }

    // Mark outputs
    bool single_output = output_node_ids.size() == 1;
    for (size_t i = 0; style.show_output_marker && i < output_node_ids.size(); ++i) {
        std::string olabel =
            single_output ? detail::escape_dot_label(name) : ("out[" + std::to_string(i) + "]");
        dot_stream << "  output_" << i << " [label=\"" << olabel << "\", "
                   << detail::node_attrs(style.output_fill, style.output_border,
                                         style.output_fontcolor)
                   << ", shape=" << style.output_shape << ", id=\"output_" << i << "\"];\n";
        dot_stream << "  node_" << output_node_ids[i] << " -> output_" << i << ";\n";

        if (!first_edge)
            edges_stream << ",";
        first_edge = false;
        edges_stream << "[" << output_node_ids[i] << ",-" << (i + 1) << "]";
    }

    dot_stream << "}\n";
    node_data_stream << "}";
    edges_stream << "]";

    std::string dot_content = dot_stream.str();

    // Write HTML file
    std::string html_filename = filename + ".html";
    std::ofstream out(html_filename);
    if (!out.is_open()) {
        throw RuntimeError("Failed to open file for writing: " + html_filename);
    }

    std::string stats_js = "const nodeCount = Object.keys(nodeData).length;\n"
                           "        document.getElementById('stats').textContent = "
                           "'Nodes: ' + nodeCount + ' | Edges: ' + edges.length;";
    detail::write_graph_html(out, detail::escape_dot_label(name) + " - Deep Graph",
                             detail::escape_for_js(dot_content), node_data_stream.str(),
                             edges_stream.str(), stats_js);
    out.close();
}

/**
 * @brief Export format for deep graph visualization
 */
enum class DeepGraphFormat {
    DOT,  ///< Graphviz DOT text format
    HTML, ///< Self-contained interactive HTML
    PDF   ///< Rendered PDF via Graphviz
};

/**
 * @brief Export a CasADi Function to deep graph format showing all operations
 *
 * This function expands the given Function to SX form, inlining all nested
 * function calls, then exports the fully-expanded computational graph.
 *
 * @param fn The CasADi Function to visualize
 * @param filename Output filename (without extension)
 * @param format Output format (DOT, HTML, or PDF)
 * @param name Optional graph name
 */
inline void export_graph_deep(const casadi::Function &fn, const std::string &filename,
                              DeepGraphFormat format = DeepGraphFormat::HTML,
                              const std::string &name = "",
                              const GraphStyle &style = GraphStyle::classic()) {
    // Use function name if no name provided
    std::string graph_name = name.empty() ? fn.name() : name;

    // Expand function to SX (inlines all nested calls)
    casadi::Function expanded = fn.expand();

    // Create SX symbolic inputs matching the function signature
    std::vector<casadi::SX> sx_inputs;
    for (casadi_int i = 0; i < expanded.n_in(); ++i) {
        sx_inputs.push_back(
            casadi::SX::sym(expanded.name_in(i), expanded.size1_in(i), expanded.size2_in(i)));
    }

    // Evaluate to get SX outputs
    std::vector<casadi::SX> sx_outputs = expanded(sx_inputs);

    // Combine all outputs into a single SX for visualization
    casadi::SX combined = casadi::SX::vertcat(sx_outputs);

    // Export using SX-specific traversal
    switch (format) {
    case DeepGraphFormat::DOT:
        export_sx_graph_dot(combined, filename, graph_name, style);
        break;
    case DeepGraphFormat::HTML:
        export_sx_graph_html(combined, filename, graph_name, style);
        break;
    case DeepGraphFormat::PDF:
        export_sx_graph_dot(combined, filename, graph_name, style);
        render_graph(filename + ".dot", filename + ".pdf");
        break;
    }
}

/**
 * @brief Convenience function: export Function to deep graph and render to PDF
 *
 * @param fn The CasADi Function to visualize
 * @param output_base Base filename (creates output_base.dot and output_base.pdf)
 * @return true if both export and render succeeded
 */
inline bool visualize_graph_deep(const casadi::Function &fn, const std::string &output_base) {
    try {
        export_graph_deep(fn, output_base, DeepGraphFormat::PDF);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

// =============================================================================
// Numeric-equivalent exports
//
// A symbolic Function has a "numeric twin": the same computation run as native
// arithmetic. CasADi can emit that twin for ANY Function, so these are the
// numeric counterparts of export_graph_deep -- one Function, two views:
//   export_graph_deep(fn)      -> the graph it traces       (DAG)
//   export_function_code(fn)   -> the code it compiles to    (portable C)
//   export_eval_algorithm(fn)  -> the algorithm it runs      (SSA work-vector)
// =============================================================================

/// Target language for export_function_code.
enum class CodeLanguage {
    C ///< Portable C99 source (CasADi codegen)
};

/**
 * @brief Emit source that numerically evaluates @p fn -- the "numeric twin".
 *
 * Thin wrapper over casadi::Function::generate(): writes self-contained,
 * dependency-free C that evaluates the function with native arithmetic. Works
 * for any function, mirroring how export_graph_deep visualizes any graph.
 *
 * @param fn       The Function to emit code for
 * @param filename Output base name (".c" is appended by CasADi)
 * @param lang     Output language (currently only C)
 * @return The path of the file written (as reported by CasADi)
 * @throws RuntimeError if code generation fails
 */
inline std::string export_function_code(const casadi::Function &fn, const std::string &filename,
                                        CodeLanguage lang = CodeLanguage::C) {
    if (lang != CodeLanguage::C) {
        throw RuntimeError("export_function_code: only C output is currently supported");
    }
    try {
        return fn.generate(filename);
    } catch (const std::exception &e) {
        throw RuntimeError("export_function_code failed for '" + fn.name() + "': " + e.what());
    }
}

namespace detail {

/// A rendered token and the role that determines its color.
enum class TokRole { Text, Op, Io, Const };
using AlgoToken = std::pair<TokRole, std::string>;
using AlgoLine = std::vector<AlgoToken>;

inline bool is_infix_op(const std::string &s) {
    return s == "+" || s == "-" || s == "*" || s == "/" || s == "<" || s == "<=" || s == ">" ||
           s == ">=" || s == "==" || s == "!=" || s == "&&" || s == "||";
}

inline std::string fmt_algo_num(double v) {
    std::ostringstream oss;
    oss << std::setprecision(6) << v;
    return oss.str();
}

inline std::string html_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '&':
            out += "&amp;";
            break;
        case '<':
            out += "&lt;";
            break;
        case '>':
            out += "&gt;";
            break;
        default:
            out += c;
        }
    }
    return out;
}

/**
 * @brief Decode a Function's evaluation algorithm into colorable token lines.
 *
 * Walks instruction_id/input/output/constant. Work-vector slots are shown as
 * `aN` (matching CasADi's C codegen register style); OP_INPUT/OP_OUTPUT use the
 * supplied input/output names. Best-effort for non-expandable (MX) functions.
 */
inline std::vector<AlgoLine> decode_algorithm(const casadi::Function &f,
                                              const std::vector<std::string> &names_in,
                                              const std::vector<std::string> &names_out) {
    std::vector<AlgoLine> lines;
    auto reg = [](casadi_int w) { return "a" + std::to_string(w); };
    auto name_at = [](const std::vector<std::string> &names, casadi_int idx) {
        return (idx >= 0 && idx < static_cast<casadi_int>(names.size()))
                   ? names[idx]
                   : ("x" + std::to_string(idx));
    };

    for (casadi_int k = 0; k < f.n_instructions(); ++k) {
        casadi_int id = f.instruction_id(k);
        std::vector<casadi_int> in = f.instruction_input(k);
        std::vector<casadi_int> out = f.instruction_output(k);
        AlgoLine line;

        if (id == casadi::OP_INPUT) {
            std::string nm = name_at(names_in, in.empty() ? 0 : in[0]);
            if (in.size() > 1 && in[1] > 0)
                nm += "[" + std::to_string(in[1]) + "]";
            line.push_back({TokRole::Text, reg(out[0]) + " = "});
            line.push_back({TokRole::Io, nm});
        } else if (id == casadi::OP_OUTPUT) {
            std::string nm = name_at(names_out, out.empty() ? 0 : out[0]);
            if (out.size() > 1 && out[1] > 0)
                nm += "[" + std::to_string(out[1]) + "]";
            line.push_back({TokRole::Io, nm});
            line.push_back({TokRole::Text, " = " + reg(in[0])});
        } else if (id == casadi::OP_CONST) {
            line.push_back({TokRole::Text, reg(out[0]) + " = "});
            line.push_back({TokRole::Const, fmt_algo_num(f.instruction_constant(k))});
        } else {
            std::string lbl = op_symbol(id);
            line.push_back({TokRole::Text, reg(out[0]) + " = "});
            if (is_infix_op(lbl) && in.size() == 2) {
                line.push_back({TokRole::Text, reg(in[0]) + " "});
                line.push_back({TokRole::Op, lbl});
                line.push_back({TokRole::Text, " " + reg(in[1])});
            } else {
                line.push_back({TokRole::Op, lbl});
                line.push_back({TokRole::Text, "("});
                for (size_t i = 0; i < in.size(); ++i) {
                    line.push_back({TokRole::Text, reg(in[i]) + (i + 1 < in.size() ? ", " : "")});
                }
                line.push_back({TokRole::Text, ")"});
            }
        }
        lines.push_back(std::move(line));
    }
    return lines;
}

/// Resolved panel colors, taken from the GraphStyle with readable fallbacks.
struct PanelColors {
    std::string bg, card, title, text, op, io, num;
};

inline PanelColors resolve_panel_colors(const GraphStyle &s) {
    auto pick = [](const std::string &v, const char *d) { return v.empty() ? std::string(d) : v; };
    PanelColors c;
    c.bg = pick(s.bgcolor, "#FAFCFE");
    c.card = "#FFFFFF";
    c.title = pick(s.input_fontcolor, "#33495C");
    c.text = "#2B3A48";
    c.op = pick(s.op_fontcolor, "#D17A12");
    c.io = pick(s.input_fontcolor, "#5F7E9B");
    c.num = pick(s.op_fontcolor, "#D17A12");
    return c;
}

inline std::string token_color(const AlgoToken &t, const PanelColors &c) {
    switch (t.first) {
    case TokRole::Op:
        return c.op;
    case TokRole::Io:
        return c.io;
    case TokRole::Const:
        return c.num;
    default:
        return c.text;
    }
}

/// Render a line as a Graphviz HTML-like-label fragment (colored <font> tokens).
inline std::string render_line_graphviz(const AlgoLine &line, const PanelColors &c) {
    std::string out;
    for (const auto &t : line) {
        if (t.first == TokRole::Text) {
            out += html_escape(t.second);
        } else {
            out += "<font color=\"" + token_color(t, c) + "\">" + html_escape(t.second) + "</font>";
        }
    }
    return out;
}

/// Render a line as an HTML-page fragment (colored <span> tokens).
inline std::string render_line_html(const AlgoLine &line, const PanelColors &c) {
    std::string out;
    for (const auto &t : line) {
        if (t.first == TokRole::Text) {
            out += html_escape(t.second);
        } else {
            out += "<span style=\"color:" + token_color(t, c) + "\">" + html_escape(t.second) +
                   "</span>";
        }
    }
    return out;
}

inline void write_algorithm_dot(const std::string &filename, const std::string &title,
                                const std::vector<AlgoLine> &lines, const PanelColors &c) {
    std::string dot_filename = filename + ".dot";
    std::ofstream out(dot_filename);
    if (!out.is_open()) {
        throw RuntimeError("Failed to open file for writing: " + dot_filename);
    }
    out << "digraph numeric_twin {\n";
    out << "  bgcolor=\"" << c.bg << "\";\n  margin=0;\n";
    out << "  node [shape=plaintext];\n";
    out << "  panel [label=<\n";
    out << "    <table border=\"1\" color=\"#D7DEE6\" cellborder=\"0\" cellspacing=\"0\" "
           "cellpadding=\"5\" bgcolor=\""
        << c.card << "\">\n";
    out << "      <tr><td align=\"left\"><font face=\"Helvetica-Bold\" point-size=\"15\" color=\""
        << c.title << "\">  " << html_escape(title) << "  </font></td></tr>\n";
    out << "      <tr><td><font point-size=\"3\"> </font></td></tr>\n";
    for (const auto &line : lines) {
        out << "      <tr><td align=\"left\"><font face=\"Courier\" point-size=\"13\" color=\""
            << c.text << "\">  " << render_line_graphviz(line, c) << "  </font></td></tr>\n";
    }
    out << "    </table>\n  >];\n}\n";
    out.close();
}

inline void write_algorithm_html(const std::string &filename, const std::string &title,
                                 const std::vector<AlgoLine> &lines, const PanelColors &c) {
    std::string html_filename = filename + ".html";
    std::ofstream out(html_filename);
    if (!out.is_open()) {
        throw RuntimeError("Failed to open file for writing: " + html_filename);
    }
    out << "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"UTF-8\">\n";
    out << "<title>" << html_escape(title) << "</title></head>\n";
    out << "<body style=\"background:" << c.bg
        << ";margin:0;padding:32px;font-family:Helvetica,"
           "Arial,sans-serif;\">\n";
    out << "<div style=\"display:inline-block;background:" << c.card
        << ";border:1px solid #D7DEE6;border-radius:8px;padding:20px 28px;\">\n";
    out << "<div style=\"color:" << c.title
        << ";font-weight:bold;font-size:18px;margin-bottom:14px;\">" << html_escape(title)
        << "</div>\n";
    out << "<pre style=\"margin:0;font-family:'Courier "
           "New',monospace;font-size:14px;line-height:1.6;"
           "color:"
        << c.text << ";\">\n";
    for (const auto &line : lines) {
        out << render_line_html(line, c) << "\n";
    }
    out << "</pre>\n</div>\n</body>\n</html>\n";
    out.close();
}

} // namespace detail

/**
 * @brief Render a Function's numeric evaluation algorithm as a code panel.
 *
 * Decodes the SSA work-vector trace (`aN = op(aI, aJ)`) the numeric backend
 * runs -- the linear counterpart to the deep graph, using the same operation
 * labels. The function is expanded to primitive ops first (falling back to the
 * MX-level trace if it is not expandable).
 *
 * @param fn       The Function whose algorithm to render
 * @param filename Output base name (extension added per format)
 * @param format   DOT, HTML, or PDF (PDF requires Graphviz `dot`)
 * @param style    Visual theme (colors); defaults to the conceptual blueprint
 * @param name     Panel title (defaults to the function name)
 */
inline void export_eval_algorithm(const casadi::Function &fn, const std::string &filename,
                                  DeepGraphFormat format = DeepGraphFormat::HTML,
                                  const GraphStyle &style = GraphStyle::conceptual(),
                                  const std::string &name = "") {
    std::string title = name.empty() ? fn.name() : name;

    casadi::Function f = fn;
    try {
        f = fn.expand(); // primitive ops; mirrors export_graph_deep
    } catch (const std::exception &) {
        f = fn; // best-effort: render the MX-level trace
    }

    std::vector<std::string> names_in, names_out;
    for (casadi_int i = 0; i < fn.n_in(); ++i)
        names_in.push_back(fn.name_in(i));
    for (casadi_int i = 0; i < fn.n_out(); ++i)
        names_out.push_back(fn.name_out(i));

    auto lines = detail::decode_algorithm(f, names_in, names_out);
    auto colors = detail::resolve_panel_colors(style);

    switch (format) {
    case DeepGraphFormat::DOT:
        detail::write_algorithm_dot(filename, title, lines, colors);
        break;
    case DeepGraphFormat::HTML:
        detail::write_algorithm_html(filename, title, lines, colors);
        break;
    case DeepGraphFormat::PDF:
        detail::write_algorithm_dot(filename, title, lines, colors);
        render_graph(filename + ".dot", filename + ".pdf");
        break;
    }
}

} // namespace metis
