# Structural Transforms

Metis provides a structural-analysis layer for dense square residual systems built as `metis::Function`s. The current pipeline covers alias elimination on trivial affine rows, block-triangular decomposition (BLT), and tearing recommendations inside coupled blocks. This works in **symbolic mode** only, operating on the Jacobian sparsity pattern of a compiled function. Code lives in `<metis/core/StructuralTransforms.hpp>`.

## Quick Start

```cpp
#include <metis/metis.hpp>

auto x = metis::sym("x", 3, 1);
auto p = metis::sym("p");
metis::SymbolicScalar x0 = x(0), x1 = x(1), x2 = x(2);

auto residual = metis::SymbolicScalar::vertcat({
    x0 - x1,
    x2 - p,
    metis::sin(x0) + x2 - 2.0,
});

metis::Function fn("system", {x, p}, {residual});

// Full pipeline: alias elimination -> BLT -> tearing
auto analysis = metis::structural_analyze(fn);
```

## Core API

```cpp
#include <metis/metis.hpp>

metis::StructuralTransformOptions opts;
opts.input_idx = 0;   // which input block is the variable vector
opts.output_idx = 0;  // which output block is the residual vector

auto alias = metis::alias_eliminate(fn, opts);
auto blt   = metis::block_triangularize(fn, opts);
auto analysis = metis::structural_analyze(fn, opts);
```

**`StructuralTransformOptions`** exposes:
- `input_idx`: which function input block is treated as the structural variable vector
- `output_idx`: which function output block is treated as the residual vector
- `max_alias_row_nnz`: maximum number of structural variable coefficients allowed in an alias row
- `require_constant_alias_coefficients`: whether alias rows must have constant coefficients

The selected input/output pair must be dense column vectors with equal dimension. If not, Metis throws `metis::InvalidArgument`.

**`AliasEliminationResult`** returns:
- `reduced_function`: same input ordering, but with the selected input block reduced to kept variables and the output containing only kept residual rows
- `reconstruct_full_input`: maps reduced selected input block plus untouched original inputs back to the full original selected input block
- kept/eliminated variable and residual indices
- the explicit substitution list

**`BLTDecomposition`** returns:
- the structural incidence pattern
- row and column permutations
- fine and coarse block offsets
- a vector of `StructuralBlock`s (each with `residual_indices`, `variable_indices`, `tear_variable_indices`)

## Usage Patterns

### When To Use This

Use structural transforms when you already have a residual function `F(x, p) -> r` and the selected state block `x` and residual block `r` are both dense column vectors with the same dimension.

Typical use cases:
- simplifying algebraic systems before passing them to a nonlinear solver
- identifying independent subsystems in a large residual graph
- choosing candidate tear variables for block-coupled algebraic loops

### Alias Elimination

`alias_eliminate()` removes rows that are affine in the selected variables and structurally simple enough to solve directly.

```cpp
auto x = metis::sym("x", 3, 1);
auto p = metis::sym("p");
metis::SymbolicScalar x0 = x(0), x1 = x(1), x2 = x(2);

auto residual = metis::SymbolicScalar::vertcat({
    x0 - x1,
    x2 - p,
    metis::sin(x0) + x2 - 2.0,
});

metis::Function fn("alias_system", {x, p}, {residual});
auto alias = metis::alias_eliminate(fn);
```

For this system:
- `x(1)` is replaced by `x(0)`
- `x(2)` is replaced by `p`
- the reduced system becomes one nonlinear equation in one kept variable

### Reconstructing The Full State

```cpp
metis::NumericMatrix x_reduced(1, 1);
x_reduced(0, 0) = 1.25;

metis::NumericMatrix x_full = alias.reconstruct_full_input.eval(x_reduced, 0.5);
metis::NumericMatrix r_reduced = alias.reduced_function.eval(x_reduced, 0.5);
```

This is useful when a solver operates on the reduced coordinates but downstream code still expects the original state layout.

### BLT Decomposition

`block_triangularize()` uses the Jacobian sparsity of the selected residual block with respect to the selected variable block and runs CasADi's block-triangular factorization.

```cpp
auto x = metis::sym("x", 4, 1);
auto residual = metis::SymbolicScalar::vertcat({
    x(0) - 1.0,
    x(1) + x(2),
    x(1) - x(2),
    x(3) - 3.0,
});

metis::Function fn("blt_blocks", {x}, {residual});
auto blt = metis::block_triangularize(fn);
```

Each `StructuralBlock` stores `residual_indices`, `variable_indices`, and `tear_variable_indices`. These indices are local to the selected input/output block, not global NLP variable numbers.

### Tearing Recommendations

Every coupled BLT block also gets a tearing recommendation. Today this is a heuristic pass:
- build the variable-dependency graph implied by the block incidence pattern
- find strongly connected components
- greedily remove the highest-degree variable inside cyclic SCCs until the block becomes acyclic

The result is a recommendation, not a mandatory solve policy. It gives you a starting point for choosing iteration variables in algebraic loops.

### Full Pipeline

`structural_analyze()` runs the current pass ordering:

1. alias elimination
2. BLT on the reduced residual system
3. tearing inside each reduced BLT block

```cpp
auto analysis = metis::structural_analyze(fn);
```

This ordering matters. Alias elimination is first so obvious substitutions simplify the incidence graph before block detection and tearing run.

### Example Walkthrough: `structural_transforms_demo.cpp`

The example `examples/math/structural_transforms_demo.cpp` shows the three core workflows:

1. Alias elimination on a square residual with two trivial affine rows and one nonlinear reduced row.
2. BLT decomposition on a system with two scalar blocks and one coupled `2x2` block.
3. Tearing on a pure `3x3` algebraic cycle.

It prints kept versus eliminated variables, the explicit substitution list, reconstructed full-state values, and BLT block membership with tear-variable recommendations.

Build and run:

```bash
ninja -C build structural_transforms_demo
./build/examples/structural_transforms_demo
```

## Diagnostics & Troubleshooting

### Current Limits

The current implementation is deliberately conservative:
- only dense square selected blocks are supported
- alias elimination only handles structurally small affine rows
- tearing returns recommendations only
- simplified residual/Jacobian code generation is not part of this pass yet

That makes this layer useful for inspection and reduction now without pretending it is a full symbolic compiler pipeline.

## See Also

- [Structural Diagnostics Guide](structural_diagnostics.md) -- Observability and identifiability analysis
- [Sparsity Guide](sparsity.md) -- Sparsity pattern extraction underpinning these transforms
- [structural_transforms_demo.cpp](../../examples/math/structural_transforms_demo.cpp) -- Full example source
- [StructuralTransforms.hpp](../../include/metis/core/StructuralTransforms.hpp) -- API reference
