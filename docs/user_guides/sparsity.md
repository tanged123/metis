# Sparsity

Understanding the sparsity pattern of your Jacobian and Hessian matrices is crucial for high-performance optimization. Metis provides tools to inspect sparsity patterns from symbolic graphs, visualize them as ASCII/PDF/HTML spy plots, compile sparse derivative evaluators that return only structural nonzeros, and surface CasADi graph coloring metadata. Sparsity analysis works in **symbolic mode**; the NaN-propagation fallback also supports **numeric mode** for black-box functions.

## Quick Start

```cpp
#include <metis/metis.hpp>

auto x = metis::sym("x", 10);
auto f = metis::SymbolicScalar::vertcat({
    x(1) - x(0),
    x(2) - metis::sin(x(1)),
    x(3) - x(2)
});

// Extract Jacobian sparsity pattern
metis::SparsityPattern sp = metis::sparsity_of_jacobian(f, x);

// Print ASCII spy plot
std::cout << sp.to_string() << std::endl;

// Build a sparse Jacobian evaluator (returns only nonzeros)
auto J = metis::sparse_jacobian(f, x);
std::cout << "nnz = " << J.nnz() << "\n";
```

## Core API

The core class is `metis::SparsityPattern` in `<metis/core/Sparsity.hpp>`.

### Extracting Sparsity

| Function | Description |
|----------|-------------|
| `metis::sparsity_of_jacobian(f, x)` | Jacobian sparsity from symbolic expressions |
| `metis::sparsity_of_hessian(f, x)` | Hessian sparsity from symbolic expressions |
| `metis::get_jacobian_sparsity(func, out_idx, in_idx)` | Jacobian sparsity from a compiled `metis::Function` |
| `metis::get_hessian_sparsity(func, out_idx, in_idx)` | Hessian sparsity from a compiled `metis::Function` |
| `metis::nan_propagation_sparsity(callable, n_in, n_out)` | Black-box sparsity via NaN propagation |
| `metis::nan_propagation_sparsity(func)` | NaN-propagation sparsity from `metis::Function` |

### Querying a SparsityPattern

| Method | Description |
|--------|-------------|
| `sp.nnz()` | Number of structural nonzeros |
| `sp.density()` | Fraction of nonzeros |
| `sp.n_rows()` / `sp.n_cols()` | Dimensions |
| `sp.get_triplet()` | Row/column index vectors |
| `sp.get_ccs()` | Compressed column storage |

### Sparse Derivative Evaluators

| Function | Description |
|----------|-------------|
| `metis::sparse_jacobian(f, x)` | Build a sparse Jacobian evaluator from expressions |
| `metis::sparse_hessian(phi, x)` | Build a sparse Hessian evaluator from a scalar expression |
| `metis::sparse_jacobian(func, out_idx, in_idx)` | Build from a `metis::Function` block |
| `metis::sparse_hessian(func, out_idx, in_idx)` | Build from a `metis::Function` block |

### Visualization

| Method | Description |
|--------|-------------|
| `sp.to_string()` | ASCII spy plot for terminal output |
| `sp.visualize_spy(filename)` | Render spy plot to PDF (requires Graphviz) |
| `sp.export_spy_html(filename, title)` | Interactive HTML spy plot with pan/zoom |

## Usage Patterns

### From Symbolic Expressions

```cpp
auto x = metis::sym("x", 10);
auto f = ...; // some expression depending on x

// Jacobian sparsity (df/dx)
metis::SparsityPattern J_sp = metis::sparsity_of_jacobian(f, x);

// Hessian sparsity (d^2 f/dx^2)
metis::SparsityPattern H_sp = metis::sparsity_of_hessian(f, x);
```

### From a Compiled Function

```cpp
metis::Function func(inputs, outputs);
auto sp = metis::get_jacobian_sparsity(func);
```

For multi-input or multi-output functions, use explicit block selection:

```cpp
auto J_sp = metis::get_jacobian_sparsity(func, output_idx, input_idx);
auto H_sp = metis::get_hessian_sparsity(func, scalar_output_idx, input_idx);
```

### From CasADi Types Directly

```cpp
casadi::Sparsity raw = ...;
metis::SparsityPattern sp(raw);

metis::SymbolicScalar expr = ...;
metis::SparsityPattern expr_sp(expr); // Extract from MX sparsity
```

### Sparse Jacobian Pipeline

```cpp
auto x = metis::sym("x", 6);
auto f = metis::SymbolicScalar::vertcat({
    x(1) - x(0),
    x(2) - metis::sin(x(1)),
    x(3) - x(2)
});

auto J = metis::sparse_jacobian(f, x);

std::cout << "nnz = " << J.nnz() << "\n";
std::cout << "forward colors = " << J.forward_coloring().n_colors() << "\n";
std::cout << "reverse colors = " << J.reverse_coloring().n_colors() << "\n";
std::cout << "preferred mode = "
          << (J.preferred_mode() == metis::SparseJacobianMode::Forward ? "forward" : "reverse")
          << "\n";

metis::NumericMatrix x_val(6, 1);
x_val << 0.0, 0.2, 0.5, 0.9, 0.0, 0.0;

metis::NumericMatrix jac_nz = J.values(x_val);
```

`jac_nz` is a column vector of derivative values in the same CCS ordering reported by `J.sparsity().get_triplet()` and `J.sparsity().get_ccs()`. That ordering is fixed, so the sparsity structure can be reused across many evaluations.

### Sparse Hessian Pipeline

```cpp
auto x = metis::sym("x", 5);
metis::SymbolicScalar phi = 0;
for (int k = 0; k < 4; ++k) {
    auto diff = x(k + 1) - x(k);
    phi = phi + diff * diff;
}

auto H = metis::sparse_hessian(phi, x);

std::cout << "nnz = " << H.nnz() << "\n";
std::cout << "star colors = " << H.coloring().n_colors() << "\n";

metis::NumericMatrix x_val(5, 1);
x_val << 0.0, 0.1, 0.3, 0.7, 1.0;

metis::NumericMatrix hess_nz = H.values(x_val);
```

For Hessians, Metis exposes CasADi's star coloring through `H.coloring()`.

### From Function Blocks

Sparse derivative evaluators can also be built from an already-compiled `metis::Function`, selecting a specific output block and input block:

```cpp
metis::Function terms("terms", {x, u}, {defects, objective});

auto ddefects_dx = metis::sparse_jacobian(terms, 0, 0);
auto ddefects_du = metis::sparse_jacobian(terms, 0, 1);
auto hobjective_xx = metis::sparse_hessian(terms, 1, 0);
```

This is the most useful form for optimization pipelines where one compiled function already exposes multiple residual, constraint, and objective blocks.

### Reconstructing a Dense Matrix

When debugging, it is often useful to reconstruct the dense matrix from sparse values:

```cpp
auto nz = J.values(x_val);
auto [rows, cols] = J.sparsity().get_triplet();

metis::NumericMatrix dense =
    metis::NumericMatrix::Zero(J.sparsity().n_rows(), J.sparsity().n_cols());

for (Eigen::Index k = 0; k < nz.size(); ++k) {
    dense(rows[static_cast<size_t>(k)], cols[static_cast<size_t>(k)]) = nz(k);
}
```

In production you usually keep the structural ordering and pass the nonzero vector straight into a sparse solver or downstream callback.

### Visualization Workflows

**ASCII spy plot** for quick terminal debugging:
```cpp
std::cout << sp.to_string() << std::endl;
```

Output:
```
Sparsity: 10x10, nnz=28 (density=28.000%)
+----------+
|**........|
|***.......
|.***......|
...
+----------+
```

**PDF rendering** for reports:
```cpp
sp.visualize_spy("my_pattern"); // Creates my_pattern.pdf
```

**Interactive HTML** for exploring large matrices:
```cpp
sp.export_spy_html("my_pattern", "My Jacobian"); // Creates my_pattern.html
```

The HTML output includes pan/zoom, clickable cells with row/col details, axis labels, and a stats panel.

## Advanced Usage

### Reading Coloring Metadata

`metis::GraphColoring` exposes:
- `n_entries()` for the uncompressed derivative size
- `n_colors()` for the compressed directional count
- `compression_ratio()` for a quick summary
- `colorvec()` for the per-entry color assignment

This is useful when comparing derivative blocks and deciding whether sparse directional sweeps are worth it.

### NaN-Propagation Sparsity Detection

Sometimes you have **black-box functions** where symbolic sparsity analysis is not possible (external library calls, non-traceable operations, functions with runtime branching). Metis provides `nan_propagation_sparsity()` for these cases.

**How it works:**
1. Evaluate f(x) at a reference point
2. For each input i: set x[i] = NaN, evaluate f(x)
3. If output[j] becomes NaN, then it depends on input i, so Jacobian(j, i) is nonzero

```cpp
// For lambda/callable functions
auto sp = metis::nan_propagation_sparsity(
    [](const metis::NumericVector& x) {
        metis::NumericVector y(x.size());
        for (int i = 0; i < x.size(); ++i) y(i) = x(i) * x(i);
        return y;
    },
    n_inputs, n_outputs);

// For metis::Function
metis::Function fn(...);
auto sp = metis::nan_propagation_sparsity(fn);

// With custom options (reference point)
metis::NaNSparsityOptions opts;
opts.reference_point = metis::NumericVector{{1.0, 2.0, 3.0}};
auto sp = metis::nan_propagation_sparsity(fn, opts);
```

**Verifying symbolic sparsity:**

```cpp
auto x = metis::sym("x", 4);
auto f = x * x;  // Element-wise square

// Symbolic sparsity
auto sp_symbolic = metis::sparsity_of_jacobian(f, x);

// NaN-propagation sparsity (black-box equivalent)
metis::Function fn({x}, {f});
auto sp_nan = metis::nan_propagation_sparsity(fn);

// They should match!
assert(sp_symbolic == sp_nan);
```

> [!TIP]
> Use NaN-propagation to validate that your symbolic expressions have the expected structure, or when interfacing with external numerical code.

### Example Walkthrough: `sparsity_intro.cpp`

The example `examples/intro/sparsity_intro.cpp` demonstrates four common structures found in optimization:

1. **Simple Jacobian** -- Shows how dependencies map to nonzeros. `f_0 = x_0^2` depends only on `x_0`, so row 0 has a nonzero at column 0.

2. **Chain Structure (Tridiagonal)** -- `f = sum((x[i] - x[i+1])^2)` creates a tridiagonal Hessian. This band structure is typical in trajectory optimization where each state depends only on its neighbors.

3. **Independent Systems (Block Diagonal)** -- Two completely separate systems stacked together form a block-diagonal matrix. Solvers can parallelize this trivially.

4. **2D Laplacian (5-Point Stencil)** -- Typical in PDE constraints. Each node depends on itself and its 4 neighbors. Uses `metis::sym_vec_pair` for 2D indexing:

```cpp
auto [x_vec, x_mx] = metis::sym_vec_pair("x", n_vars);
// Build equations using x_vec(k), create Function using raw x_mx
metis::Function f_pde({x_mx}, {metis::SymbolicScalar::vertcat(eqs)});
```

### Example Walkthrough: `sparse_derivative_pipeline.cpp`

The example `examples/math/sparse_derivative_pipeline.cpp` shows the full workflow:

1. Build a trajectory-style residual with local coupling
2. Compile sparse Jacobian blocks and a sparse Hessian block
3. Print `nnz` versus dense size
4. Inspect forward, reverse, and star coloring counts
5. Reuse the same sparse kernels at two different evaluation points

Build and run:

```bash
ninja -C build sparse_derivative_pipeline
./build/examples/sparse_derivative_pipeline
```

## See Also

- [Graph Visualization Guide](graph_visualization.md) -- Visualize the computational graph itself
- [Structural Diagnostics Guide](structural_diagnostics.md) -- Use sparsity for observability and identifiability analysis
- [sparsity_intro.cpp](../../examples/intro/sparsity_intro.cpp) -- Introductory sparsity patterns example
- [sparse_derivative_pipeline.cpp](../../examples/math/sparse_derivative_pipeline.cpp) -- Full sparse derivative pipeline example
- [Sparsity.hpp](../../include/metis/core/Sparsity.hpp) -- API reference
