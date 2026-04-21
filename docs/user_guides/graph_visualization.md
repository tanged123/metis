# Graph Visualization

Metis uses **computational graphs** (directed acyclic graphs built by CasADi) to represent symbolic expressions. This guide explains how to export and visualize those graphs for debugging, optimization analysis, teaching, and documentation. Graph visualization works in **symbolic mode** only, since numeric-mode expressions do not build a traceable graph.

## Quick Start

```cpp
#include <metis/metis.hpp>

auto x = metis::sym("x");
auto expr = metis::sin(x) * x;

// One-step: export DOT and render PDF (requires Graphviz)
metis::visualize_graph(expr, "my_graph");

// Or interactive HTML (no dependencies, recommended for exploration)
metis::export_graph_html(expr, "my_graph", "SinExpression");
```

## Core API

All functions live in `<metis/core/MetisIO.hpp>`.

| Function | Description |
|----------|-------------|
| `metis::export_graph_dot(expr, filename, title)` | Export a symbolic expression to DOT format |
| `metis::render_graph(dot_file, output_file)` | Render a DOT file to PDF, PNG, or SVG |
| `metis::visualize_graph(expr, base)` | Convenience wrapper: export DOT then render to PDF |
| `metis::export_graph_html(expr, filename, title)` | Export an interactive HTML graph with pan/zoom and node details |

> [!NOTE]
> `render_graph` and `visualize_graph` require **Graphviz** installed. In NixOS, it is included in the dev shell. The HTML export has no external dependencies.

## Usage Patterns

### What is a Computational Graph?

A **computational graph** is a DAG where each node is either:

- **Input nodes** (leaves): Symbolic variables or constants
- **Operation nodes** (internal): Mathematical operations (+, *, sin, etc.)
- **Output node** (root): The final result

When you write symbolic expressions in Metis, CasADi builds this graph internally. It enables:

1. **Automatic differentiation** -- Traverse the graph to compute gradients
2. **Code generation** -- Compile the graph to efficient C code
3. **Optimization** -- Symbolic solvers operate on the graph structure

### Simple Expression Graph

```cpp
auto x = metis::sym("x");
auto y = x * x + 2.0 * x + 1.0;  // Builds a graph with 5+ nodes
```

This creates a tree structure:

```
        (+)
       /   \
     (+)    1.0
    /   \
  (*)    (*)
  / \    / \
 x   x  2   x
```

### Two-Step Export (DOT then Render)

```cpp
metis::export_graph_dot(expr, "my_graph", "SinExpression");
metis::render_graph("my_graph.dot", "my_graph.pdf");
```

### Interactive HTML Export

The **HTML output** is recommended for exploring complex graphs:
- **Click nodes** to see the full expression in the sidebar
- **Pan/zoom** with mouse drag and scroll
- **Connection highlighting** when a node is selected

```cpp
metis::export_graph_html(expr, "my_graph", "SinExpression");
```

### Understanding Graph Layout

The graph uses **bottom-to-top** layout (`rankdir=BT`):
- Inputs (variables and constants) are at the bottom
- Operations build upward
- The final output is at the top

**Node colors:**
- Green ellipses: Input variables (id, iq, Rs, Ld, etc.)
- Yellow ellipses: Constants (1.5, etc.)
- Blue boxes: Operations (multiply, add, subtract)
- Gold circle: Output node

### Shared Subexpressions

CasADi automatically detects **common subexpressions**. If `iq` appears in multiple places, the graph reuses the same node -- this is a key optimization for automatic differentiation.

### Deep vs. Shallow Graphs

| Expression | Depth | Use Case |
|------------|-------|----------|
| `x + y` | 1 | Simple operations |
| `sin(x) * cos(y)` | 2 | Transcendental functions |
| `motor.voltage_d(...)` | 3-4 | Engineering models |
| `P_elec` (full motor) | 6+ | Nested function calls |

### Real Example: Electric Motor Power

The `graph_visualization.cpp` example models a **Permanent Magnet Synchronous Motor (PMSM)**.

```cpp
// Create symbolic motor parameters
auto Rs = metis::sym("Rs");
auto Ld = metis::sym("Ld");
// ... more parameters

MotorModel<metis::SymbolicScalar> motor{Rs, Ld, Lq, lambda, p, J, B};

// Compute voltage equations
auto Vd = motor.voltage_d(id, iq, did_dt, omega_e);
auto Vq = motor.voltage_q(id, iq, diq_dt, omega_e);

// Build power expression
auto P_elec = motor.electrical_power(Vd, Vq, id, iq);

// Visualize the computational graph
metis::visualize_graph(P_elec, "graph_power");      // PDF
metis::export_graph_html(P_elec, "graph_power");   // Interactive HTML
```

The power expression `P = 1.5 * (Vd*id + Vq*iq)` expands to include all the intermediate terms from the voltage equations, creating a deep graph.

## Advanced Usage

### Jacobian Graphs

You can visualize the graph of **derivatives**:

```cpp
auto T_e = motor.electromagnetic_torque(id, iq);

// Compute Jacobian: dT/d[id, iq]
auto dT_dq = metis::jacobian({T_e}, {id, iq});

// Visualize the derivative graph
metis::visualize_graph(dT_dq, "jacobian_graph");
```

The Jacobian graph shows how CasADi computes gradients by applying the chain rule symbolically.

### Practical Applications

1. **Debugging** -- Verify your expression has the expected structure
2. **Optimization analysis** -- See which variables affect the output
3. **Teaching** -- Demonstrate automatic differentiation concepts
4. **Documentation** -- Generate figures for papers and reports

### Running the Example

```bash
cd /path/to/metis
./scripts/build.sh
./build/examples/graph_visualization

# View generated graphs (PDF requires Graphviz)
xdg-open graph_power.pdf
xdg-open graph_dynamics.pdf

# Or open interactive HTML in browser (no dependencies)
xdg-open graph_power.html     # Or: explorer.exe graph_power.html (WSL)
xdg-open graph_dynamics.html
```

## See Also

- [Symbolic Computing Guide](symbolic_computing.md) -- Symbolic mode fundamentals
- [Sparsity Guide](sparsity.md) -- Visualize sparsity patterns of Jacobians and Hessians
- [graph_visualization.cpp](../../examples/math/graph_visualization.cpp) -- Full example source
- [MetisIO.hpp](../../include/metis/core/MetisIO.hpp) -- API reference
