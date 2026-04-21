# Metis Refinement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all fixes from the refinement audit across 5 phases: bug fixes, dead code removal, architectural refactoring, naming standardization, and minor cleanup.

**Architecture:** Header-only C++20 library using CasADi + Eigen. All changes are in `include/metis/`. Tests in `tests/`, examples in `examples/`. Build with `cmake -B build && cmake --build build`.

**Tech Stack:** C++20, CasADi (symbolic), Eigen 3.4 (linear algebra), GTest (testing)

**Build & Test:**
```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

**Consumer Impact:** Vulcan uses `num_iterations()` in 9 places (examples only). Icarus is unaffected. Both are clean of deprecated APIs.

**IMPORTANT:** Leave all doxygen comments (`@brief`, `@param`, `@return`, etc.) intact throughout. They are user-facing documentation.

---

## Phase 1: Critical Bug Fixes

### Task 1: Fix solve_sweep() Silent Data Loss (C6)

**Files:**
- Modify: `include/metis/optimization/OptiSweep.hpp` (add error tracking fields)
- Modify: `include/metis/optimization/Opti.hpp:672-691` (fix catch block)
- Modify: `tests/optimization/test_opti.cpp` (add sweep error test)

- [ ] **Step 1: Add error tracking to SweepResult**

In `include/metis/optimization/OptiSweep.hpp`, add fields to `SweepResult`:

```cpp
struct SweepResult {
    std::vector<double> param_values;
    std::vector<OptiSol> solutions;
    std::vector<int> iterations;
    bool all_converged = true;

    /// Per-point convergence status (true = converged)
    std::vector<bool> converged;

    /// Error messages for failed points (empty string if converged)
    std::vector<std::string> errors;

    size_t size() const { return param_values.size(); }
    // ... rest unchanged
```

- [ ] **Step 2: Fix the catch block in solve_sweep()**

In `include/metis/optimization/Opti.hpp`, replace lines 672-691 (the for loop in `solve_sweep`):

```cpp
for (size_t idx = 0; idx < values.size(); ++idx) {
    double val = values[idx];
    opti_.set_value(param, val);

    try {
        auto sol = opti_.solve();
        result.solutions.emplace_back(sol);
        result.iterations.push_back(sol.stats().count("iter_count")
                                        ? static_cast<int>(sol.stats().at("iter_count"))
                                        : -1);
        result.converged.push_back(true);
        result.errors.emplace_back();
    } catch (const std::exception &e) {
        result.all_converged = false;
        result.converged.push_back(false);
        result.errors.push_back(e.what());
        // Store a default-constructed OptiSol placeholder is not feasible;
        // instead, record the index gap - solutions.size() < param_values.size()
        // means the failed index is: converged[i] == false
        result.iterations.push_back(-1);
    }
}
```

- [ ] **Step 3: Build and verify compilation**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
```

- [ ] **Step 4: Run existing tests to ensure no regressions**

```bash
cd build && ctest --output-on-failure -R test_opti
```

- [ ] **Step 5: Commit**

```bash
git add include/metis/optimization/OptiSweep.hpp include/metis/optimization/Opti.hpp
git commit -m "fix: solve_sweep continues on failure, reports per-point errors (C6)"
```

---

### Task 2: Fix OptiSweep::objective() Hardcoded 0.0 (C7)

**Files:**
- Modify: `include/metis/optimization/OptiSweep.hpp:39-52`
- Modify: `include/metis/optimization/Opti.hpp` (store objective expression in SweepResult)

- [ ] **Step 1: Replace hardcoded objective with proper evaluation**

The issue is that CasADi's `OptiSol` doesn't expose the objective directly, but `sol.value(expr)` can evaluate any expression. The objective expression must be passed in.

Replace `SweepResult::objective()` in `OptiSweep.hpp`:

```cpp
/// Get objective value at sweep index.
/// Requires the objective expression that was passed to minimize/maximize.
double objective(size_t i, const SymbolicScalar &objective_expr) const {
    if (i >= solutions.size()) {
        throw InvalidArgument("SweepResult::objective: index out of range");
    }
    return solutions[i].value(objective_expr);
}
```

- [ ] **Step 2: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure -R test_opti
```

- [ ] **Step 3: Commit**

```bash
git add include/metis/optimization/OptiSweep.hpp
git commit -m "fix: SweepResult::objective() evaluates actual expression instead of returning 0.0 (C7)"
```

---

### Task 3: Fix OptiSol Silent Fallbacks (M9)

**Files:**
- Modify: `include/metis/optimization/OptiSol.hpp:87-104`

- [ ] **Step 1: Change return type to std::optional<int>**

Replace both methods in `OptiSol.hpp`:

```cpp
/**
 * @brief Get number of function evaluations
 * @return Number of function evaluations, or nullopt if not available
 */
std::optional<int> num_function_evals() const {
    auto s = stats();
    if (s.count("n_call_nlp_f")) {
        return static_cast<int>(s.at("n_call_nlp_f"));
    }
    return std::nullopt;
}

/**
 * @brief Get number of iterations
 * @return Number of iterations, or nullopt if not available
 */
std::optional<int> num_iterations() const {
    auto s = stats();
    if (s.count("iter_count")) {
        return static_cast<int>(s.at("iter_count"));
    }
    return std::nullopt;
}
```

- [ ] **Step 2: Update any internal uses of num_iterations() returning -1**

Search for uses in `Opti.hpp` solve_sweep. The iterations vector already handles -1 from the catch block, which is fine as a sentinel in the vector. No internal code calls `num_iterations()`.

- [ ] **Step 3: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add include/metis/optimization/OptiSol.hpp
git commit -m "fix: OptiSol num_iterations/num_function_evals return optional instead of -1 (M9)"
```

---

### Task 4: Fix Quaternion from_rotation_matrix() Symbolic Branch (C8)

**Files:**
- Modify: `include/metis/math/Quaternion.hpp:230-244`

- [ ] **Step 1: Implement full 4-branch symbolic conversion using metis::where**

Replace the symbolic branch (lines 230-244) in `from_rotation_matrix()`:

```cpp
} else {
    // Symbolic: Full 4-branch using nested metis::where (Shepperd's method)
    // Branch 0: trace > 0
    Scalar s0 = half / metis::sqrt(trace + one);
    Scalar w0 = static_cast<Scalar>(0.25) / s0;
    Scalar x0 = (mat(2, 1) - mat(1, 2)) * s0;
    Scalar y0 = (mat(0, 2) - mat(2, 0)) * s0;
    Scalar z0 = (mat(1, 0) - mat(0, 1)) * s0;

    // Branch 1: mat(0,0) is largest diagonal
    Scalar s1 = two * metis::sqrt(one + mat(0, 0) - mat(1, 1) - mat(2, 2));
    Scalar w1 = (mat(2, 1) - mat(1, 2)) / s1;
    Scalar x1 = static_cast<Scalar>(0.25) * s1;
    Scalar y1 = (mat(0, 1) + mat(1, 0)) / s1;
    Scalar z1 = (mat(0, 2) + mat(2, 0)) / s1;

    // Branch 2: mat(1,1) is largest diagonal
    Scalar s2 = two * metis::sqrt(one + mat(1, 1) - mat(0, 0) - mat(2, 2));
    Scalar w2 = (mat(0, 2) - mat(2, 0)) / s2;
    Scalar x2 = (mat(0, 1) + mat(1, 0)) / s2;
    Scalar y2 = static_cast<Scalar>(0.25) * s2;
    Scalar z2 = (mat(1, 2) + mat(2, 1)) / s2;

    // Branch 3: mat(2,2) is largest diagonal
    Scalar s3 = two * metis::sqrt(one + mat(2, 2) - mat(0, 0) - mat(1, 1));
    Scalar w3 = (mat(1, 0) - mat(0, 1)) / s3;
    Scalar x3 = (mat(0, 2) + mat(2, 0)) / s3;
    Scalar y3 = (mat(1, 2) + mat(2, 1)) / s3;
    Scalar z3 = static_cast<Scalar>(0.25) * s3;

    // Select using nested where: trace>0 ? branch0 : (R00>R11 && R00>R22) ? branch1 : R11>R22 ? branch2 : branch3
    auto cond_trace = trace > static_cast<Scalar>(0.0);
    auto cond_r00 = metis::logical_and(mat(0, 0) > mat(1, 1), mat(0, 0) > mat(2, 2));
    auto cond_r11 = mat(1, 1) > mat(2, 2);

    // Inner: branch2 vs branch3
    Scalar wi = metis::where(cond_r11, w2, w3);
    Scalar xi = metis::where(cond_r11, x2, x3);
    Scalar yi = metis::where(cond_r11, y2, y3);
    Scalar zi = metis::where(cond_r11, z2, z3);

    // Middle: branch1 vs inner
    Scalar wm = metis::where(cond_r00, w1, wi);
    Scalar xm = metis::where(cond_r00, x1, xi);
    Scalar ym = metis::where(cond_r00, y1, yi);
    Scalar zm = metis::where(cond_r00, z1, zi);

    // Outer: branch0 vs middle
    q_w = metis::where(cond_trace, w0, wm);
    q_x = metis::where(cond_trace, x0, xm);
    q_y = metis::where(cond_trace, y0, ym);
    q_z = metis::where(cond_trace, z0, zm);
}
```

- [ ] **Step 2: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure -R test_math
```

- [ ] **Step 3: Commit**

```bash
git add include/metis/math/Quaternion.hpp
git commit -m "fix: from_rotation_matrix() symbolic path handles all rotation angles via nested where (C8)"
```

---

### Task 5: Fix Arithmetic fmod() Infinite Recursion (C8)

**Files:**
- Modify: `include/metis/math/Arithmetic.hpp:365-375`

- [ ] **Step 1: Fix the recursive call**

The matrix fmod for CasADi calls itself recursively. Fix by using element-wise application:

```cpp
template <typename Derived, typename Scalar>
auto fmod(const Eigen::MatrixBase<Derived> &x, const Scalar &y) {
    if constexpr (std::is_same_v<typename Derived::Scalar, casadi::MX>) {
        // Element-wise fmod for CasADi matrices
        using MatScalar = typename Derived::Scalar;
        Eigen::Matrix<MatScalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> result(
            x.rows(), x.cols());
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            for (Eigen::Index j = 0; j < x.cols(); ++j) {
                result(i, j) = metis::fmod(x(i, j), static_cast<MatScalar>(y));
            }
        }
        return result;
    } else {
        return x.binaryExpr(Eigen::MatrixBase<Derived>::Constant(x.rows(), x.cols(), y),
                            [](double a, double b) { return std::fmod(a, b); });
    }
}
```

- [ ] **Step 2: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure -R test_math
```

- [ ] **Step 3: Commit**

```bash
git add include/metis/math/Arithmetic.hpp
git commit -m "fix: fmod matrix variant uses element-wise dispatch instead of infinite recursion (C8)"
```

---

### Task 6: Fix SurrogateModel softplus_scalar() Symbolic Stability (C8)

**Files:**
- Modify: `include/metis/math/SurrogateModel.hpp:147-150`

- [ ] **Step 1: Add stabilization to symbolic softplus**

Replace the symbolic `softplus_scalar` (line 147-150):

```cpp
inline auto softplus_scalar(const SymbolicScalar &x, double beta) {
    const auto bx = beta * x;
    // logsumexp(0, bx) = log(exp(0) + exp(bx)) which is numerically stable
    // CasADi's logsumexp handles the shift internally
    return SymbolicScalar::logsumexp(SymbolicScalar::vertcat({SymbolicScalar(0.0), bx})) / beta;
}
```

Actually, looking at CasADi's `logsumexp` implementation — it computes `log(sum(exp(x)))` with the standard max-shift stabilization internally. The current implementation is correct. Add a validating comment instead:

```cpp
inline auto softplus_scalar(const SymbolicScalar &x, double beta) {
    const auto bx = beta * x;
    // CasADi's logsumexp applies max-shift stabilization internally:
    // logsumexp(a) = max(a) + log(sum(exp(a - max(a))))
    return SymbolicScalar::logsumexp(SymbolicScalar::vertcat({SymbolicScalar(0.0), bx})) / beta;
}
```

- [ ] **Step 2: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure -R test_math
```

- [ ] **Step 3: Commit**

```bash
git add include/metis/math/SurrogateModel.hpp
git commit -m "docs: clarify softplus_scalar symbolic stability (C8)"
```

---

## Phase 2: Dead Code Removal

### Task 7: Remove DiffOps.hpp (M3)

**Files:**
- Delete: `include/metis/math/DiffOps.hpp`
- Verify: `include/metis/math/MetisMath.hpp` (already commented out)

- [ ] **Step 1: Delete the file**

Delete `include/metis/math/DiffOps.hpp` (15-line deprecated redirect).

- [ ] **Step 2: Verify MetisMath.hpp doesn't include it**

Confirm line 32 in `MetisMath.hpp` is still commented out:
```cpp
// #include "metis/math/DiffOps.hpp"
```

- [ ] **Step 3: Search for any remaining references**

```bash
grep -r "DiffOps" include/ tests/ examples/
```

- [ ] **Step 4: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git rm include/metis/math/DiffOps.hpp
git commit -m "chore: delete deprecated DiffOps.hpp redirect (M3)"
```

---

### Task 8: Remove Backward-Compat Aliases (M3)

**Files:**
- Modify: `include/metis/math/Interpolate.hpp:1093-1106` (remove aliases)
- Modify: `include/metis/math/Interpolate.hpp:1237-1275` (remove interpn2d)
- Modify: `include/metis/math/Logic.hpp:548-552` (remove clip)

- [ ] **Step 1: Remove Interp1D and MetisInterpolator aliases**

In `Interpolate.hpp`, delete lines 1093-1106:
```cpp
// ============================================================================
// Backwards Compatibility
// ============================================================================

/**
 * @brief Alias for 1D interpolation (backwards compatibility)
 * @deprecated Use Interpolator directly
 */
using Interp1D = Interpolator;

/**
 * @deprecated Use Interpolator directly
 */
using MetisInterpolator = Interpolator;
```

- [ ] **Step 2: Remove interpn2d deprecated wrappers**

In `Interpolate.hpp`, delete lines 1237-1275 (the two `interpn2d` function templates).

- [ ] **Step 3: Remove clip() alias**

In `Logic.hpp`, delete lines 548-552:
```cpp
// --- Clip ---
/**
 * @brief Alias for clamp
 */
template <typename... Args> auto clip(Args &&...args) { return clamp(std::forward<Args>(args)...); }
```

- [ ] **Step 4: Update using.hpp if it exports clip**

Check if `using.hpp` exports `clip` and remove if so.

- [ ] **Step 5: Search for any internal uses**

```bash
grep -r "Interp1D\|MetisInterpolator\|interpn2d\|metis::clip\b" include/ tests/ examples/
```

Fix any found references.

- [ ] **Step 6: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 7: Commit**

```bash
git add include/metis/math/Interpolate.hpp include/metis/math/Logic.hpp
git commit -m "chore: remove deprecated aliases Interp1D, MetisInterpolator, interpn2d, clip (M3)"
```

---

### Task 9: Inline OptiCache into OptiSol (M3)

**Files:**
- Modify: `include/metis/optimization/OptiSol.hpp` (add static load method)
- Delete: `include/metis/optimization/OptiCache.hpp`
- Modify: `include/metis/optimization/Opti.hpp:1` (remove OptiCache include)
- Modify: `tests/optimization/test_opti_cache.cpp` (update references)
- Modify: `tests/CMakeLists.txt` if needed

- [ ] **Step 1: Add static load method to OptiSol**

In `OptiSol.hpp`, add a static method:

```cpp
/**
 * @brief Load solution data from JSON file
 *
 * @param filename JSON file path
 * @return Map of variable names to value vectors
 * @throws RuntimeError if file cannot be read or parsed
 */
static std::map<std::string, std::vector<double>> load(const std::string &filename) {
    return metis::utils::read_json(filename);
}
```

- [ ] **Step 2: Update Opti.hpp include**

Change `#include "OptiCache.hpp"` to remove it from `Opti.hpp` line 3.

- [ ] **Step 3: Update test file**

In `tests/optimization/test_opti_cache.cpp`, replace `OptiCache::load` with `OptiSol::load`.

- [ ] **Step 4: Delete OptiCache.hpp**

- [ ] **Step 5: Search for remaining references**

```bash
grep -r "OptiCache" include/ tests/ examples/
```

- [ ] **Step 6: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 7: Commit**

```bash
git rm include/metis/optimization/OptiCache.hpp
git add include/metis/optimization/OptiSol.hpp include/metis/optimization/Opti.hpp tests/
git commit -m "chore: inline OptiCache::load into OptiSol, delete OptiCache.hpp (M3)"
```

---

## Phase 3: Architectural Refactoring

### Task 10: Deduplicate MetisIO.hpp HTML Template (C3)

**Files:**
- Modify: `include/metis/core/MetisIO.hpp`

The two HTML export functions (`export_graph_html` at line 375 and `export_sx_graph_html` at line 1055) share nearly identical HTML/CSS/JS (~200 lines each). Extract the shared template.

- [ ] **Step 1: Create shared HTML template helper in detail namespace**

Add a helper function in the `detail` namespace (around line 108) that generates the HTML page given data parameters:

```cpp
namespace detail {

/// Shared HTML template for interactive graph visualization.
/// Takes the escaped DOT string, node data JSON, and edges JSON.
inline void write_graph_html(std::ostream &out, const std::string &title,
                             const std::string &escaped_dot,
                             const std::string &node_data_json,
                             const std::string &edges_json,
                             const std::string &extra_stats_js = "") {
    // Write the full HTML page with embedded Viz.js
    // This is the single source of truth for the interactive graph viewer
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
        )HTMLEND"
        << extra_stats_js << R"HTMLFINAL(

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
            svg.querySelectorAll('.node').forEach(node => {
                node.addEventListener('click', e => {
                    e.stopPropagation();
                    selectNode(svg, node.id);
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
                const label = data.short || data.label || '';
                const full = data.full || data.label || label;
                sidebar.innerHTML = `
                    <div class="section">
                        <div class="label">Node ID</div>
                        <div class="value">${data.id} <span class="type-badge type-${data.type}">${data.type}</span></div>
                    </div>
                    <div class="section">
                        <div class="label">Label</div>
                        <div class="value">${escapeHtml(label)}</div>
                    </div>
                    ${full !== label ? `<div class="section"><div class="label">Full Expression</div><div class="value">${escapeHtml(full)}</div></div>` : ''}
                    <div class="section">
                        <div class="label">Dependencies (${data.deps.length})</div>
                        <div class="value">${data.deps.length > 0 ? data.deps.map(d => 'node_' + d).join(', ') : 'None'}</div>
                    </div>
                `;
            } else if (nodeId.startsWith('output')) {
                const outIdx = nodeId.replace('output_', '').replace('output', '');
                sidebar.innerHTML = `
                    <div class="section">
                        <div class="label">Node</div>
                        <div class="value">Output${outIdx ? '[' + outIdx + ']' : ''} <span class="type-badge" style="background:#FFD700;color:#000;">output</span></div>
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
)HTMLFINAL";
}

/// Escape a string for embedding in JavaScript
inline std::string escape_for_js(const std::string &content) {
    std::string escaped;
    for (char c : content) {
        if (c == '\\') escaped += "\\\\";
        else if (c == '"') escaped += "\\\"";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else escaped += c;
    }
    return escaped;
}

/// Escape a string for embedding in JSON
inline std::string escape_for_json(const std::string &s) {
    std::string result;
    for (char c : s) {
        if (c == '"') result += "\\\"";
        else if (c == '\\') result += "\\\\";
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else result += c;
    }
    return result;
}

} // namespace detail
```

- [ ] **Step 2: Refactor export_graph_html() to use shared template**

Replace the HTML writing portion of `export_graph_html()` (lines 552-757). Keep the DOT/JSON generation logic, but replace the HTML output with:

```cpp
std::string html_filename = filename + ".html";
std::ofstream out(html_filename);
if (!out.is_open()) {
    throw RuntimeError("Failed to open file for writing: " + html_filename);
}
detail::write_graph_html(out, detail::escape_dot_label(name),
                         detail::escape_for_js(dot_content),
                         node_data_stream.str(),
                         edges_stream.str());
out.close();
```

- [ ] **Step 3: Refactor export_sx_graph_html() to use shared template**

Same approach — replace lines 1238-1434 HTML output with call to `detail::write_graph_html`. Add stats JS as the `extra_stats_js` parameter:

```cpp
std::string stats_js = "const nodeCount = Object.keys(nodeData).length;\n"
                       "document.getElementById('stats').textContent = "
                       "'Nodes: ' + nodeCount + ' | Edges: ' + edges.length;";
detail::write_graph_html(out, detail::escape_dot_label(name) + " - Deep Graph",
                         detail::escape_for_js(dot_content),
                         node_data_stream.str(),
                         edges_stream.str(),
                         stats_js);
```

- [ ] **Step 4: Remove duplicated escape_json lambdas**

Both functions define identical `escape_json` lambdas. Replace usages with `detail::escape_for_json()`.

- [ ] **Step 5: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure -R test_core
```

- [ ] **Step 6: Commit**

```bash
git add include/metis/core/MetisIO.hpp
git commit -m "refactor: deduplicate HTML graph template in MetisIO.hpp, ~400 lines removed (C3)"
```

---

### Task 11: Consolidate Opti.hpp variable() Overloads (C4, M7)

**Files:**
- Modify: `include/metis/optimization/Opti.hpp:191-337`

The 4 `variable()` overloads share duplicated scaling/bounds logic. Consolidate into 2 clean overloads (scalar + vector) that delegate to a shared internal helper.

- [ ] **Step 1: No structural change needed**

Looking at the code more carefully, the overloads serve genuinely different signatures:
1. `variable(double init_guess, ...)` — scalar, no options
2. `variable(double init_guess, VariableOptions, ...)` — scalar with options (freeze/category)
3. `variable(int n_vars, double init_guess, ...)` — vector with uniform init
4. `variable(NumericVector init_guess, ...)` — vector with per-element init

Overload 1 already delegates to overload 2. The vector overloads (3, 4) share scaling/bounds logic but differ in init guess handling. Extract the shared pattern into a private helper:

```cpp
private:
    SymbolicVector create_vector_variable(int n_vars, const NumericVector &init_guess,
                                          std::optional<double> scale,
                                          std::optional<double> lower_bound,
                                          std::optional<double> upper_bound,
                                          const std::string &category = "Uncategorized") {
        double s = scale.has_value()
                       ? opti_detail::validate_positive_scale(scale.value(), "Opti::variable")
                       : opti_detail::suggest_vector_scale(init_guess, lower_bound, upper_bound);

        SymbolicScalar raw_var = opti_.variable(n_vars, 1);
        SymbolicScalar scaled_var = s * raw_var;

        // Set initial guess
        std::vector<double> init_vec(init_guess.data(), init_guess.data() + init_guess.size());
        for (auto &v : init_vec) { v /= s; }
        opti_.set_initial(raw_var, init_vec);

        if (lower_bound.has_value()) { opti_.subject_to(scaled_var >= lower_bound.value()); }
        if (upper_bound.has_value()) { opti_.subject_to(scaled_var <= upper_bound.value()); }

        register_variable_block(init_guess, category, s, scale.has_value(), lower_bound, upper_bound);
        return metis::to_eigen(scaled_var);
    }
```

Then simplify the two vector overloads to delegate to it.

- [ ] **Step 2: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure -R test_opti
```

- [ ] **Step 3: Commit**

```bash
git add include/metis/optimization/Opti.hpp
git commit -m "refactor: consolidate variable() vector overloads via shared helper (C4, M7)"
```

---

### Task 12: Make casadi_opti() Return Const Reference (M11)

**Files:**
- Modify: `include/metis/optimization/Opti.hpp:951`

- [ ] **Step 1: Remove the mutable overload**

Change line 951 from:
```cpp
casadi::Opti &casadi_opti() { return opti_; }
const casadi::Opti &casadi_opti() const { return opti_; }
```
To:
```cpp
const casadi::Opti &casadi_opti() const { return opti_; }
```

- [ ] **Step 2: Search for any mutable usage**

```bash
grep -r "casadi_opti()" include/ tests/ examples/
```

If any code calls `casadi_opti()` and modifies the result, it will need updating.

- [ ] **Step 3: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add include/metis/optimization/Opti.hpp
git commit -m "refactor: casadi_opti() returns const ref to prevent bypassing Metis invariants (M11)"
```

---

## Phase 4: Naming Standardization

### Task 13: Unify Detail Namespaces to Plain `detail` (C2)

**Files (8 renames):**
- Modify: `include/metis/core/StructuralTransforms.hpp` — `structural_detail` → `detail`
- Modify: `include/metis/core/Diagnostics.hpp` — `diagnostics_detail` → `detail`
- Modify: `include/metis/optimization/Opti.hpp` — `opti_detail` → `detail`
- Modify: `include/metis/math/PolynomialChaos.hpp` — `polynomial_chaos_detail` → `detail`
- Modify: `include/metis/math/Integrate.hpp` — `integrate_detail` → `detail`
- Modify: `include/metis/math/Quadrature.hpp` — `quadrature_detail` → `detail`
- Modify: `include/metis/math/AutoDiff.hpp` — `autodiff_detail` → `detail`
- Modify: `include/metis/math/Logic.hpp` — `logic_detail` → `detail`

- [ ] **Step 1: Rename all prefixed detail namespaces**

For each file, replace `namespace <prefix>_detail` with `namespace detail` and update all references. Use find-and-replace:

For each file:
- `structural_detail::` → `detail::`
- `diagnostics_detail::` → `detail::`
- `opti_detail::` → `detail::`
- `polynomial_chaos_detail::` → `detail::`
- `integrate_detail::` → `detail::`
- `quadrature_detail::` → `detail::`
- `autodiff_detail::` → `detail::`
- `logic_detail::` → `detail::`

Also rename the namespace declarations themselves.

**Note:** Multiple files already use `namespace detail` — this is fine in C++ as long as they're all within `namespace metis`. Multiple `namespace detail` blocks in different headers just extend the same namespace.

- [ ] **Step 2: Handle AutoDiff.hpp which uses both patterns**

In `AutoDiff.hpp`, there's both `namespace detail` (line 119) and `namespace autodiff_detail` (line 129). Merge `autodiff_detail` into `detail`.

- [ ] **Step 3: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add include/
git commit -m "refactor: unify all *_detail namespaces to plain detail (C2)"
```

---

### Task 14: Standardize Enum Values to PascalCase (C1)

**Files:**
- Modify: `include/metis/optimization/OptiOptions.hpp:14-18`
- Modify: All files that reference `Solver::IPOPT`, `Solver::SNOPT`, `Solver::QPOASES`

- [ ] **Step 1: Rename Solver enum values**

In `OptiOptions.hpp`:
```cpp
enum class Solver {
    Ipopt,   ///< Interior Point OPTimizer (default, always available)
    Snopt,   ///< Sparse Nonlinear OPTimizer (requires license)
    QpOases  ///< QP solver for QP subproblems
};
```

- [ ] **Step 2: Update all references in the codebase**

```bash
grep -rn "Solver::IPOPT\|Solver::SNOPT\|Solver::QPOASES" include/ tests/ examples/
```

Replace all occurrences:
- `Solver::IPOPT` → `Solver::Ipopt`
- `Solver::SNOPT` → `Solver::Snopt`
- `Solver::QPOASES` → `Solver::QpOases`

Key files to update:
- `OptiOptions.hpp` (enum definition + default value on line 103)
- `Opti.hpp` (switch statements in solve(), solve_sweep(), configure_*_opts())
- `tests/optimization/test_opti_solvers.cpp`
- Any examples that use solver selection

- [ ] **Step 3: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 4: Commit**

```bash
git add include/ tests/ examples/
git commit -m "refactor: rename Solver enum values to PascalCase: Ipopt, Snopt, QpOases (C1)"
```

---

## Phase 5: Minor Cleanup

### Task 15: Fix FiniteDifference.hpp Relative Includes (M4)

**Files:**
- Modify: `include/metis/math/FiniteDifference.hpp:3-4`

- [ ] **Step 1: Replace relative with absolute paths**

Change:
```cpp
#include "../core/MetisError.hpp"
#include "../core/MetisTypes.hpp"
```
To:
```cpp
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
```

- [ ] **Step 2: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure -R test_math
```

- [ ] **Step 3: Commit**

```bash
git add include/metis/math/FiniteDifference.hpp
git commit -m "fix: use absolute include paths in FiniteDifference.hpp (M4)"
```

---

### Task 16: Name Magic Numbers (M5)

**Files:**
- Modify: `include/metis/math/ScatteredInterpolator.hpp`
- Modify: `include/metis/math/IntegrateDiscrete.hpp`
- Modify: `include/metis/math/OrthogonalPolynomials.hpp`

- [ ] **Step 1: Add named constants in ScatteredInterpolator.hpp**

Near the top of the file (inside namespace metis or the relevant detail namespace), add:

```cpp
namespace detail {
/// Regularization term for RBF interpolation matrix to prevent singularity
constexpr double rbf_regularization = 1e-10;
/// Threshold below which thin-plate spline uses linear fallback to avoid log(0)
constexpr double rbf_zero_threshold = 1e-15;
} // namespace detail
```

Replace the literal `1e-10` at line ~327 and `1e-15` at lines ~46,56 with the named constants.

- [ ] **Step 2: Add named constants in IntegrateDiscrete.hpp**

```cpp
namespace detail {
/// Small epsilon to prevent division by zero in RMS fusion
constexpr double rms_fusion_epsilon = 1e-100;
} // namespace detail
```

Replace `1e-100` at lines ~229 and ~409.

- [ ] **Step 3: Add named constant in OrthogonalPolynomials.hpp**

```cpp
namespace detail {
/// Tolerance for Legendre root convergence (Newton iteration)
constexpr double legendre_root_tolerance = 1e-15;
} // namespace detail
```

Replace the literal at the relevant location.

- [ ] **Step 4: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure -R test_math
```

- [ ] **Step 5: Commit**

```bash
git add include/metis/math/ScatteredInterpolator.hpp include/metis/math/IntegrateDiscrete.hpp include/metis/math/OrthogonalPolynomials.hpp
git commit -m "refactor: name magic numbers with constexpr constants (M5)"
```

---

### Task 17: Rename BirkhoffOptions and Normalize MultiShootingOptions (M10)

**Files:**
- Modify: `include/metis/optimization/BirkhoffPseudospectral.hpp`
- Modify: `include/metis/optimization/MultiShooting.hpp`
- Update any test/example files that use these options

- [ ] **Step 1: Rename BirkhoffOptions to BirkhoffPseudospectralOptions**

In `BirkhoffPseudospectral.hpp`, rename the struct and update all references.

- [ ] **Step 2: Normalize MultiShootingOptions field name**

In `MultiShooting.hpp`, if `n_intervals` exists, add an alias or rename to `n_nodes` for consistency with other transcription options. Check what the other options structs use.

- [ ] **Step 3: Update all references**

```bash
grep -rn "BirkhoffOptions\|n_intervals" include/ tests/ examples/
```

- [ ] **Step 4: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 5: Commit**

```bash
git add include/ tests/ examples/
git commit -m "refactor: rename BirkhoffOptions to BirkhoffPseudospectralOptions, normalize options (M10)"
```

---

### Task 18: Fix .cursorrules Stale Reference (m1)

**Files:**
- Modify: `.cursorrules`

- [ ] **Step 1: Fix the stale reference**

Replace any reference to `include/metis/linalg/` with `include/metis/math/Linalg.hpp`.

- [ ] **Step 2: Commit**

```bash
git add .cursorrules
git commit -m "docs: fix stale linalg path reference in .cursorrules (m1)"
```

---

### Task 19: Remove Redundant Include from MetisMath.hpp (m2)

**Files:**
- Modify: `include/metis/math/MetisMath.hpp:9`

- [ ] **Step 1: Remove the redundant include**

Remove `#include "metis/core/MetisError.hpp"` from line 9 — it's already transitively included through every math header.

- [ ] **Step 2: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 3: Commit**

```bash
git add include/metis/math/MetisMath.hpp
git commit -m "chore: remove redundant MetisError.hpp include from MetisMath.hpp (m2)"
```

---

### Task 20: Mark JsonUtils as Internal (M12)

**Files:**
- Modify: `include/metis/utils/JsonUtils.hpp`

- [ ] **Step 1: Fix silent error handling**

In `JsonUtils.hpp` line 117, replace the silent catch:

```cpp
try {
    vec.push_back(std::stod(number_str));
} catch (const std::exception &e) {
    throw metis::RuntimeError("Malformed JSON: could not parse number '" +
                              number_str + "': " + e.what());
}
```

- [ ] **Step 2: Build and test**

```bash
cmake -B build -DBUILD_EXAMPLES=OFF && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

- [ ] **Step 3: Commit**

```bash
git add include/metis/utils/JsonUtils.hpp
git commit -m "fix: JsonUtils throws on parse errors instead of silently dropping values (M12)"
```

---

### Task 21: Final Build Verification

- [ ] **Step 1: Full build with examples**

```bash
cmake -B build && cmake --build build -j$(nproc)
```

- [ ] **Step 2: Run all tests**

```bash
cd build && ctest --output-on-failure
```

- [ ] **Step 3: Verify no regressions**

All tests should pass. If any fail, investigate and fix.
