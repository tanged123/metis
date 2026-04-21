# Graph Visualization for MetisIO.hpp

**Status**: ✅ COMPLETE

Implements user-friendly tools to visualize CasADi computational graphs as specified in Phase 4 Milestone 5.

---

## Implemented Changes

### MetisIO

#### [MODIFY] [MetisIO.hpp](file:///home/tanged/sources/metis/include/metis/core/MetisIO.hpp)

Added graph visualization functions:

```cpp
// Export symbolic expression to DOT format
void export_graph_dot(const casadi::MX& expr, const std::string& filename,
                      const std::string& name = "expression");

// Render DOT file to PDF/PNG/SVG using Graphviz
bool render_graph(const std::string& dot_file, const std::string& output_file);

// Convenience: export + render in one call
bool visualize_graph(const casadi::MX& expr, const std::string& output_base);
```

/**
 * @brief Render DOT file to PDF/PNG/SVG using Graphviz
 */
bool render_graph(const std::string& dot_file, const std::string& output_file);

/**
 * @brief Convenience: export + render in one call
 */
bool visualize_graph(const SymbolicScalar& expr, const std::string& output_base);
bool visualize_graph(const Function& func, const std::string& output_base);
```

**Implementation approach:**
1. Create a `casadi::Function` from the expression with empty inputs
2. Use `casadi::Function::get_str(true)` for detailed representation
3. Generate DOT format manually by parsing the string representation
4. Call Graphviz `dot` command via `std::system()` or `popen()`

---

### Tests

#### [MODIFY] [test_metis_io.cpp](file:///home/tanged/sources/metis/tests/core/test_metis_io.cpp)

Add tests to existing `test_metis_io.cpp`:

```cpp
TEST(MetisIOTests, ExportGraphDot) {
    auto x = metis::sym("x");
    auto y = x * x + 2 * x + 1;
    
    std::string dot_file = "/tmp/test_graph.dot";
    metis::export_graph_dot(y, dot_file, "quadratic");
    
    std::ifstream f(dot_file);
    EXPECT_TRUE(f.good());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("digraph") != std::string::npos);
}

TEST(MetisIOTests, RenderGraph) {
    if (std::system("which dot > /dev/null 2>&1") != 0) {
        GTEST_SKIP() << "Graphviz not available";
    }
    
    auto x = metis::sym("x");
    bool success = metis::visualize_graph(x * x, "/tmp/test_viz");
    EXPECT_TRUE(success);
}
```

---

### Examples

#### [NEW] [graph_visualization.cpp](file:///home/tanged/sources/metis/examples/graph_visualization.cpp)

```cpp
#include <metis/metis.hpp>
#include <iostream>

int main() {
    auto x = metis::sym("x");
    auto y = metis::sym("y");
    auto z = metis::sin(x) * metis::cos(y) + x * y;
    
    metis::export_graph_dot(z, "expression_graph.dot", "trig_expr");
    metis::visualize_graph(z, "expression");
    
    return 0;
}
```

---

## Verification Plan

### Automated Tests

```bash
./scripts/test.sh
cd build && ctest -R test_metis_io --output-on-failure
```

### Manual Verification

```bash
./build/bin/graph_visualization
cat expression_graph.dot  # Verify DOT syntax
xdg-open expression_graph.pdf  # View rendered graph
```
