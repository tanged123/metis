# Structural Diagnostics

Metis exposes a structural preflight layer that answers two common model-quality questions: (1) are the selected states structurally observable from the chosen outputs, and (2) are the selected parameters structurally identifiable from the chosen outputs. The implementation works from the symbolic Jacobian sparsity pattern of a `metis::Function` and lives in `<metis/core/Diagnostics.hpp>`. This is a **symbolic-mode** analysis -- it answers whether the measurement layout can separate variables based on symbolic dependence alone, not numeric coefficient values.

## Quick Start

```cpp
#include <metis/metis.hpp>

auto x = metis::sym("x", 3, 1);
auto y = metis::SymbolicScalar::vertcat({
    x(0) + x(1),
    x(1),
});

metis::Function h("sensor_model", {x}, {y});
auto obs = metis::analyze_structural_observability(h);

// obs.structural_rank, obs.rank_deficiency, obs.deficient_local_indices, etc.
```

## Core API

```cpp
#include <metis/metis.hpp>

// Observability analysis on input block 0
auto obs = metis::analyze_structural_observability(h, 0);

// Identifiability analysis on input block 1
auto id = metis::analyze_structural_identifiability(h, 1);

// Combined diagnostics
metis::StructuralDiagnosticsOptions opts;
opts.state_input_idx = 0;
opts.parameter_input_idx = 1;
auto both = metis::analyze_structural_diagnostics(h, opts);
```

**`StructuralSensitivityOptions`** exposes:
- `output_indices`: optional subset of function outputs to analyze; defaults to all outputs

**`StructuralDiagnosticsOptions`** adds:
- `state_input_idx`: input block interpreted as the state vector
- `parameter_input_idx`: input block interpreted as the parameter vector

At least one of `state_input_idx` or `parameter_input_idx` must be provided for the combined helper.

**`StructuralSensitivityReport`** returns:
- `structural_rank`: structural rank of the selected Jacobian block
- `rank_deficiency`: `n_variables - structural_rank`
- `deficient_local_indices`: all input entries in a deficient structural component
- `zero_sensitivity_local_indices`: entries with no structural dependence on chosen outputs
- `deficiency_groups`: connected variable/output groups where structural rank is too small
- `issues`: user-facing remediation hints

The report also carries flattened input and output labels so you can connect a deficiency back to the original function blocks.

## Usage Patterns

### When To Use This

Use structural diagnostics before:
- fitting parameters against a measurement model
- introducing estimator states into a filter or smoother
- sending a large reduced-order model into an optimizer and wondering why some degrees of freedom drift

Typical setup:

```cpp
metis::Function h("h", {x, p}, {y});
```

where `x` is a dense column-vector state block, `p` is a dense column-vector parameter block, and `y` contains the measured or otherwise selected outputs.

### Observability Analysis

```cpp
auto x = metis::sym("x", 3, 1);
auto y = metis::SymbolicScalar::vertcat({
    x(0) + x(1),
    x(1),
});

metis::Function h("sensor_model", {x}, {y});
auto obs = metis::analyze_structural_observability(h);
```

Here:
- `x[2]` has zero structural sensitivity
- the structural rank is `2 / 3`
- the diagnostic suggests adding a sensor depending on `x[2]` or constraining/fixing it

### Identifiability Analysis

```cpp
auto x = metis::sym("x");
auto p = metis::sym("p", 4, 1);
auto y = metis::SymbolicScalar::vertcat({
    p(0) + p(1) + x,
    p(1) + p(2),
});

metis::Function h("calibration_model", {x, p}, {y});
auto id = metis::analyze_structural_identifiability(h, 1);
```

Here:
- `p[3]` is unused and therefore immediately unidentifiable
- `p[0]`, `p[1]`, and `p[2]` share only two structurally independent output rows
- the report surfaces one coupled deficiency group and recommends adding measurements that separate that block

### Combined Diagnostics

If one measurement model carries both estimation states and calibration parameters, run both checks together:

```cpp
metis::StructuralDiagnosticsOptions opts;
opts.state_input_idx = 0;
opts.parameter_input_idx = 1;

auto report = metis::analyze_structural_diagnostics(h, opts);
// report.observability
// report.identifiability
// report.has_deficiency()
```

### Example Walkthrough: `structural_diagnostics_demo.cpp`

The example `examples/math/structural_diagnostics_demo.cpp` demonstrates:

1. An observability gap caused by an unmeasured state.
2. An identifiability gap caused by a coupled parameter block plus an unused parameter.
3. The combined state-plus-parameter report on a shared measurement model.

Build and run:

```bash
ninja -C build structural_diagnostics_demo
./build/examples/structural_diagnostics_demo
```

## Diagnostics & Troubleshooting

### Current Limits

The current implementation is deliberately structural and local:
- selected input blocks must be dense column vectors
- selected outputs must be dense
- conclusions are based on symbolic Jacobian sparsity, not numeric coefficient values
- a structurally full-rank report does not guarantee good conditioning or practical estimator quality

That makes this pass useful as an early symbolic filter before you spend time on solver tuning, experiment design, or estimator debugging.

## See Also

- [Sparsity Guide](sparsity.md) -- Sparsity pattern extraction and visualization
- [Structural Transforms Guide](structural_transforms.md) -- Alias elimination, BLT decomposition, and tearing
- [structural_diagnostics_demo.cpp](../../examples/math/structural_diagnostics_demo.cpp) -- Full example source
- [Diagnostics.hpp](../../include/metis/core/Diagnostics.hpp) -- API reference
