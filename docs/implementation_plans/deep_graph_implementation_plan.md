# Deep Computational Graph Generation - Implementation Strategy

## Overview

This document outlines the implementation strategy for adding deep computational graph visualization to Metis. The goal is to export CasADi `Function` objects as fully-expanded graphs showing all mathematical operations, rather than opaque function call nodes.

## Problem Statement

Current `export_graph_dot()` and `export_graph_html()` functions traverse MX expression graphs using `.n_dep()` and `.dep()`. When a graph contains calls to other `Function` objects, these appear as single "function call" nodes without exposing their internal operations.

**Example - Current (Shallow):**

```
[t] ──► [dynamics(t, x)] ──► [xdot]
[x] ──►
```

**Desired (Deep):**

```
[t]
[x[0..2]] ─► [norm] ─► [r³] ─► [μ/r³] ─► [×] ─► [accel] ─► [xdot[3..5]]
[x[3..5]] ────────────────────────────────────────────────► [xdot[0..2]]
```

## Technical Background

### CasADi Expression Types

| Type | Description | Use Case |
|------|-------------|----------|
| **SX** | Scalar symbolic expression | Scalar operations, efficient for small dense problems |
| **MX** | Matrix symbolic expression | Sparse operations, function calls, larger problems |
| **Function** | Callable symbolic function | Encapsulates expressions, provides evaluation interface |

### Key Insight

`Function::expand()` converts an MX-based function to an SX-based function, inlining all nested calls. However:

- SX uses different graph traversal API (`SXElem::n_dep()`, `SXElem::dep()`)
- Converting SX back to MX via `MX(sx)` wraps it as a constant, losing structure
- SX elements within a matrix are accessed via `.at(i)` or similar

## Proposed Solution

### Approach 1: Native SX Graph Traversal (Recommended)

Create dedicated SX graph export functions that understand the SX expression tree.

#### API Design

```cpp
namespace metis {

// Primary API - works with any Function
void export_graph_deep(const Function& fn, 
                       const std::string& filename,
                       ExportFormat format = ExportFormat::HTML);

// Low-level APIs
void export_sx_graph_dot(const casadi::SX& expr, 
                         const std::string& filename,
                         const std::string& name = "expression");

void export_sx_graph_html(const casadi::SX& expr,
                          const std::string& filename, 
                          const std::string& name = "expression");

enum class ExportFormat { DOT, HTML, PDF, JSON };

} // namespace metis
```

#### Implementation Steps

1. **Add SX Graph Traversal** (`MetisIO.hpp`)

   ```cpp
   // Traverse SX expression tree
   void traverse_sx_graph(const casadi::SX& expr,
                          std::set<const void*>& visited,
                          std::vector<SXNode>& nodes,
                          std::vector<SXEdge>& edges);
   ```

2. **Implement SX Element Inspection**

   ```cpp
   struct SXNodeInfo {
       std::string operation;  // "sin", "mul", "add", etc.
       std::string value;      // For constants/symbols
       bool is_leaf;
       int n_children;
   };
   
   SXNodeInfo get_sx_node_info(const casadi::SXElem& elem);
   ```

3. **Add Deep Export Function**

   ```cpp
   void export_graph_deep(const Function& fn, 
                          const std::string& filename,
                          ExportFormat format) {
       // 1. Expand function to SX
       casadi::Function expanded = fn.casadi_function().expand();
       
       // 2. Create SX symbolic inputs
       std::vector<casadi::SX> sx_inputs;
       for (casadi_int i = 0; i < expanded.n_in(); ++i) {
           sx_inputs.push_back(casadi::SX::sym(
               expanded.name_in(i), 
               expanded.size1_in(i), 
               expanded.size2_in(i)));
       }
       
       // 3. Evaluate to get SX outputs
       std::vector<casadi::SX> sx_outputs = expanded(sx_inputs);
       
       // 4. Combine outputs
       casadi::SX combined = casadi::SX::vertcat(sx_outputs);
       
       // 5. Export using SX-specific traversal
       switch (format) {
           case ExportFormat::DOT:
               export_sx_graph_dot(combined, filename);
               break;
           case ExportFormat::HTML:
               export_sx_graph_html(combined, filename);
               break;
           // ...
       }
   }
   ```

### Approach 2: Code Generation (Alternative)

Use `casadi::CodeGenerator` to produce readable algorithm representation.

```cpp
void export_algorithm_text(const Function& fn, 
                           const std::string& filename) {
    casadi::Function expanded = fn.casadi_function().expand();
    
    casadi::CodeGenerator cg(filename);
    cg.add(expanded);
    
    std::string code = cg.dump();
    
    std::ofstream out(filename + ".c");
    out << code;
}
```

**Pros:** Detailed operation-by-operation listing
**Cons:** C code format, not visual graph

### Approach 3: Hybrid (Recommended for Full Feature Set)

Combine both approaches:

- SX graph traversal for visual DOT/HTML output
- CodeGenerator for detailed algorithm text

## File Changes

### [MODIFY] `include/metis/core/MetisIO.hpp`

Add the following sections:

```cpp
// ============================================================================
// SX Graph Traversal (Deep Visualization)
// ============================================================================

namespace detail {

struct SXGraphNode {
    int id;
    std::string label;
    std::string fillcolor;
    std::string shape;
    bool is_input;
    bool is_output;
};

struct SXGraphEdge {
    int from_id;
    int to_id;
};

// Get operation name from SXElem
inline std::string get_sx_operation(const casadi::SXElem& elem) {
    if (elem.is_symbolic()) return elem.name();
    if (elem.is_constant()) {
        std::ostringstream oss;
        oss << std::setprecision(4) << static_cast<double>(elem);
        return oss.str();
    }
    if (elem.is_op()) {
        // Map CasADi operation codes to readable names
        int op = elem.op();
        // Return operation name based on op code
        // See casadi/core/calculus.hpp for OP_* constants
    }
    return "?";
}

} // namespace detail

// Primary deep export API
inline void export_graph_deep(const casadi::Function& fn,
                              const std::string& filename,
                              const std::string& name = "expression") {
    // Implementation as described above
}
```

### [NEW] Test File

Create `tests/graph/deep_graph_test.cpp`:

```cpp
TEST(DeepGraphTest, ExpandedFunctionHasMoreNodes) {
    // Create a function with nested calls
    auto x = metis::sym("x", 3);
    auto inner = metis::sin(x(0)) + metis::cos(x(1));
    auto outer = inner * x(2);
    
    metis::Function fn("test", {x}, {outer});
    
    // Export shallow
    metis::export_graph_dot(outer, "shallow_test");
    
    // Export deep  
    metis::export_graph_deep(fn.casadi_function(), "deep_test");
    
    // Count nodes in each file
    int shallow_nodes = count_nodes("shallow_test.dot");
    int deep_nodes = count_nodes("deep_test.dot");
    
    EXPECT_GT(deep_nodes, shallow_nodes);
}
```

## Implementation Phases

### Phase 1: SX Traversal Infrastructure

- [ ] Implement `detail::get_sx_operation()` for all CasADi operation types
- [ ] Implement `traverse_sx_graph()` for recursive SX tree traversal
- [ ] Add unit tests for basic SX expressions

### Phase 2: Deep Export Functions

- [ ] Implement `export_sx_graph_dot()` using SX traversal
- [ ] Implement `export_sx_graph_html()` with interactive visualization
- [ ] Implement `export_graph_deep()` wrapper for `Function` objects

### Phase 3: Algorithm Text Export

- [ ] Add `export_algorithm_text()` using CodeGenerator
- [ ] Parse and format C code output for readability

### Phase 4: Integration & Testing

- [ ] Add comprehensive tests for various function types
- [ ] Update Icarus demo to use new deep export
- [ ] Documentation updates

## CasADi Operation Codes Reference

For `get_sx_operation()`, map these common operation codes:

| Code | Operation | Label |
|------|-----------|-------|
| OP_CONST | Constant | value |
| OP_INPUT | Input | name |
| OP_ADD | Addition | + |
| OP_SUB | Subtraction | − |
| OP_MUL | Multiplication | × |
| OP_DIV | Division | ÷ |
| OP_NEG | Negation | − |
| OP_SQ | Square | x² |
| OP_SQRT | Square root | √ |
| OP_SIN | Sine | sin |
| OP_COS | Cosine | cos |
| OP_TAN | Tangent | tan |
| OP_EXP | Exponential | exp |
| OP_LOG | Logarithm | log |
| OP_POW | Power | ^ |
| OP_FABS | Absolute value | |x| |

## Success Criteria

1. **Visual Verification**: Deep graph shows individual operations (sin, cos, mul, etc.) instead of single function call nodes
2. **Node Count**: Deep graph has significantly more nodes than shallow graph for complex functions
3. **Correctness**: Graph structure correctly represents the mathematical operations
4. **Performance**: Export completes in reasonable time for typical simulation models (< 100ms for 6-DOF dynamics)

## Dependencies

- CasADi headers: `casadi/core/sx.hpp`, `casadi/core/sx_elem.hpp`
- Graphviz (optional): For PDF/PNG rendering

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Very large graphs for complex functions | Add node count limit with warning |
| CasADi internal API changes | Pin to specific CasADi version, add API compatibility tests |
| Performance for large expressions | Add async export option, progress callback |
