# N-Dimensional Interpolation

The `interpn` function provides N-dimensional interpolation on regular grids, supporting dimensions from 1D up to 7D and beyond. It works seamlessly with Metis's dual-backend architecture, allowing the same code to run in both fast numeric mode and symbolic trace mode for optimization. For unstructured point clouds, `ScatteredInterpolator` fits RBF models and resamples onto grids for symbolic-compatible queries.

## Quick Start

```cpp
#include <metis/math/Interpolate.hpp>

// Create a 2D grid
metis::NumericVector x_pts(3), y_pts(3);
x_pts << 0.0, 1.0, 2.0;
y_pts << 0.0, 1.0, 2.0;

std::vector<metis::NumericVector> points = {x_pts, y_pts};

// Grid values: z = x + y (Fortran order)
metis::NumericVector values(9);
values << 0, 1, 2, 1, 2, 3, 2, 3, 4;

// Query points
metis::NumericMatrix xi(2, 2);
xi << 0.5, 0.5,   // Point 1
      1.5, 1.0;   // Point 2

// Interpolate
auto result = metis::interpn<double>(points, values, xi);
// result(0) ~ 1.0, result(1) ~ 2.5
```

## Core API

### `interpn<Scalar>()`

```cpp
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
interpn(const std::vector<Eigen::VectorXd>& points,
        const Eigen::VectorXd& values_flat,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& xi,
        InterpolationMethod method = InterpolationMethod::Linear,
        std::optional<Scalar> fill_value = std::nullopt);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `std::vector<Eigen::VectorXd>` | Grid coordinates for each dimension. Each vector must be sorted. |
| `values_flat` | `Eigen::VectorXd` | Flattened grid values in **Fortran order** (column-major). |
| `xi` | `Eigen::Matrix<Scalar, ...>` | Query points, shape `(n_points, n_dims)` or `(n_dims, n_points)`. |
| `method` | `InterpolationMethod` | Interpolation algorithm (default: `Linear`). |
| `fill_value` | `std::optional<Scalar>` | Value for out-of-bounds queries. If `nullopt`, clamps to boundary. |

Returns `Eigen::Matrix<Scalar, Dynamic, 1>` -- vector of interpolated values at each query point.

### Interpolation Methods

| Method | Enum Value | Continuity | Symbolic | Description |
|--------|------------|------------|----------|-------------|
| **Linear** | `InterpolationMethod::Linear` | C0 | Yes | Piecewise linear. Fast, simple, but has gradient discontinuities at grid points. |
| **Hermite** | `InterpolationMethod::Hermite` | C1 | Numeric only | Catmull-Rom cubic spline. Smooth gradients, good for animation and trajectories. |
| **BSpline** | `InterpolationMethod::BSpline` | C2 | Yes | Cubic B-spline. Smoothest option, ideal for optimization. Requires >= 4 points per dimension. |
| **Nearest** | `InterpolationMethod::Nearest` | None | Numeric only | Nearest neighbor. Fast lookup, non-differentiable. |

### Symbolic Table Values

`interpn()` also accepts a `metis::SymbolicVector` of table values, keeping the lookup table coefficients inside the symbolic graph so they can be optimized directly:

```cpp
auto [table_values, table_values_mx] = metis::sym_vec_pair("table", 4);

metis::NumericMatrix xi(1, 2);
xi << 0.25, 0.75;

auto result = metis::interpn(points, table_values, xi,
                             metis::InterpolationMethod::Linear);
```

This parameterized-table path currently supports `Linear` and `BSpline`.
`Hermite` and `Nearest` remain numeric-only for table values.

### Type Aliases (Recommended)

```cpp
// Numeric
metis::NumericVector  // = Eigen::VectorXd
metis::NumericMatrix  // = Eigen::MatrixXd
metis::NumericScalar  // = double

// Symbolic
metis::SymbolicVector // = Eigen::Matrix<casadi::MX, Dynamic, 1>
metis::SymbolicMatrix // = Eigen::Matrix<casadi::MX, Dynamic, Dynamic>
metis::SymbolicScalar // = casadi::MX
```

## Usage Patterns

### Choosing a Method

```text
Use Case                              Recommended Method
---------------------------------------------------------
Fast lookup, low accuracy             Nearest
General purpose, good balance         Linear
Smooth trajectories, C1 gradients     Hermite
Optimization problems, C2 smooth      BSpline
Symbolic differentiation              Linear or BSpline
```

### Smoothness Comparison

```text
                Grid Point
                    |
                    v
    +---------------*---------------+
    |               |               |
C0  |    /----------*----------\    |  Linear: Gradient jumps at knots
    |   /           |           \   |
    +---------------*---------------+
    |            /--*--\            |
C1  |          /    |    \          |  Hermite: Smooth gradient, curvature jump
    |         /     |     \         |
    +---------------*---------------+
    |           ,---*---,           |
C2  |          ,    |    ,          |  BSpline: Smooth gradient AND curvature
    |         ,     |     ,         |
    +-------------------------------+
```

### Data Layout: Fortran Order

Grid values must be flattened in **Fortran order** (column-major), where the first dimension varies fastest.

For a 2x3 grid with coordinates `x = [0, 1]` and `y = [0, 1, 2]`:

```text
Grid Layout (visual):          Flattening Order:
    y=0   y=1   y=2
x=0  a     b     c            [a, d, b, e, c, f]
x=1  d     e     f             ^  ^
                               +--+-- x varies fastest
```

```cpp
// 2D grid: x = [0, 1], y = [0, 1, 2]
metis::NumericVector x(2), y(3);
x << 0, 1;
y << 0, 1, 2;

// Values: z(x,y) = x + y
// Fortran order: iterate x first for each y
metis::NumericVector values(6);
values << 0,   // (0,0) = 0+0
          1,   // (1,0) = 1+0
          1,   // (0,1) = 0+1
          2,   // (1,1) = 1+1
          2,   // (0,2) = 0+2
          3;   // (1,2) = 1+2
```

### Boundary Handling

When `fill_value` is not provided, out-of-bounds queries are **clamped** to the grid boundary:

```cpp
xi << -1.0, 0.5;  // x = -1 is outside [0, 1]

auto result = metis::interpn<double>(points, values, xi);
// Effectively queries at (0.0, 0.5) - clamped to boundary
```

Use `fill_value` to return a specific value for out-of-bounds queries:

```cpp
auto result = metis::interpn<double>(
    points, values, xi,
    metis::InterpolationMethod::Linear,
    std::optional<double>(-999.0)  // Fill value
);
// Returns -999.0 for any query outside grid bounds
```

### Linear Extrapolation

For optimization problems, clamping produces **zero gradients** outside bounds, which can stall solvers. The `ExtrapolationConfig` class provides linear extrapolation with optional safety bounds:

```cpp
metis::NumericVector x(4), y(4);
x << 0, 1, 2, 3;
y << 0, 10, 40, 90;

metis::Interpolator interp(x, y,
    metis::InterpolationMethod::BSpline,
    metis::ExtrapolationConfig::linear(0.0, 200.0));

double val = interp(4.0);  // Extrapolates linearly from boundary slope
                            // then clamps result to [0, 200]
```

| Factory Method | Behavior |
|----------------|----------|
| `ExtrapolationConfig::clamp()` | Clamp queries to grid bounds (default, safe) |
| `ExtrapolationConfig::linear()` | Linear extrapolation, unbounded |
| `ExtrapolationConfig::linear(lower, upper)` | Linear extrapolation with output bounds |

Linear extrapolation works in symbolic mode with full AD support:

```cpp
auto x_sym = metis::sym("x");
auto y_sym = interp(x_sym);

auto dy_dx = metis::jacobian(y_sym, x_sym);
```

N-D extrapolation is fully supported. For each dimension that falls outside bounds, the extrapolation adds a correction term:

```text
result = interp(clamped_query) + sum( slope_d * (query_d - boundary_d) )
```

```cpp
metis::Interpolator interp2d(points, values,
    metis::InterpolationMethod::Linear,
    metis::ExtrapolationConfig::linear(-10.0, 100.0));

metis::NumericVector query(2);
query << 3.0, 3.0;
double result = interp2d(query);
```

### Loading Data from File

```cpp
Eigen::MatrixXd data = load_csv("aero_coeffs.csv");

metis::NumericVector mach = data.col(0).head(n_mach);
metis::NumericVector alpha = data.col(1).head(n_alpha);
metis::NumericVector CL = data.col(2);

std::vector<metis::NumericVector> points = {mach, alpha};

auto cl = metis::interpn<double>(points, CL, query_pts);
```

### Surrogate Model in Optimization

```cpp
template <typename Scalar>
Scalar drag_model(const metis::MetisVector<Scalar>& state) {
    metis::MetisMatrix<Scalar> query(1, 2);
    query(0, 0) = state(0);  // Mach
    query(0, 1) = state(1);  // Angle of attack

    auto cd = metis::interpn<Scalar>(
        aero_grid, cd_values, query,
        metis::InterpolationMethod::BSpline
    );
    return cd(0);
}
```

### Symbolic Mode

For optimization problems, use symbolic interpolation to embed lookups in your objective:

```cpp
metis::SymbolicMatrix xi(1, 2);
xi(0, 0) = metis::sym("x");
xi(0, 1) = metis::sym("y");

auto z_sym = metis::interpn<metis::SymbolicScalar>(
    points, values, xi,
    metis::InterpolationMethod::BSpline
);

auto f = metis::Function("lookup", {xi(0,0), xi(0,1)}, {z_sym(0)});
```

| Method | Symbolic Mode | Reason |
|--------|---------------|--------|
| Linear | Yes | Uses CasADi `interpolant` |
| Hermite | No | Requires interval selection at runtime; throws and recommends BSpline |
| BSpline | Yes | Uses CasADi `interpolant` |
| Nearest | No | Requires rounding (non-smooth) |

### Scattered Data Interpolation

For unstructured point cloud data (non-gridded), use `ScatteredInterpolator`. It fits Radial Basis Functions (RBF) to scattered points, then resamples onto a grid for fast symbolic-compatible queries.

```cpp
#include <metis/math/ScatteredInterpolator.hpp>

metis::NumericMatrix points(20, 2);  // 20 test points, 2D input
metis::NumericVector values(20);
// ... fill in data ...

metis::ScatteredInterpolator interp(points, values);

metis::NumericVector query(2);
query << 0.6, 5.0;
double result = interp(query);
```

| Feature | Gridded (`Interpolator`) | Scattered (`ScatteredInterpolator`) |
|---------|--------------------------|-------------------------------------|
| Data Structure | Regular axis-aligned grid | Arbitrary point cloud |
| Performance | Very fast (tensorially separable) | Slower (RBF solve at construction) |
| Use Case | Uniformly sampled data | Wind tunnel points, CFD meshes |
| Symbolic Mode | Native | Via gridded resampling |

RBF Kernels:

| Kernel | Enum Value | Description |
|--------|------------|-------------|
| **Thin Plate Spline** | `RBFKernel::ThinPlateSpline` | r^2 log(r) - smooth, good default |
| **Multiquadric** | `RBFKernel::Multiquadric` | sqrt(1 + (er)^2) - adjustable shape |
| **Gaussian** | `RBFKernel::Gaussian` | exp(-(er)^2) - localized influence |
| **Linear** | `RBFKernel::Linear` | r - simple, stable for few points |
| **Cubic** | `RBFKernel::Cubic` | r^3 - smooth |

Check fit quality and use in symbolic mode:

```cpp
std::cout << "RMS error: " << interp.reconstruction_error() << "\n";

auto sym_x = metis::sym("x");
auto result = interp(sym_x);
auto grad = metis::jacobian(result, sym_x);
```

## Advanced Usage

### High-Dimensional Interpolation

Metis supports interpolation in arbitrary dimensions using **tensor product** extension:

```cpp
metis::NumericVector pts(3);
pts << 0.0, 1.0, 2.0;

std::vector<metis::NumericVector> points(5, pts);  // 5 dimensions

// Values: 3^5 = 243 grid points
metis::NumericVector values(243);
// ... fill values ...

metis::NumericMatrix xi(1, 5);
xi << 0.5, 0.5, 0.5, 0.5, 0.5;

auto result = metis::interpn<double>(points, values, xi);
```

| Dimensions | Grid Size | Points | Memory |
|------------|-----------|--------|--------|
| 2D | 10x10 | 100 | ~1 KB |
| 3D | 10x10x10 | 1,000 | ~8 KB |
| 4D | 10x10x10x10 | 10,000 | ~80 KB |
| 5D | 10x10x10x10x10 | 100,000 | ~800 KB |

> **Note**: High-dimensional grids grow exponentially. Consider sparse grids or surrogate models for >5D.

### Implementation Details

Both numeric and symbolic interpolation use **CasADi** as the underlying engine:

```text
+-----------------------------------------------------+
|                   metis::interpn                     |
+-----------------------------------------------------+
|                                                      |
|  Scalar = double        Scalar = casadi::MX          |
|  ----------------       ---------------------        |
|        |                       |                     |
|        v                       v                     |
|  +-------------------------------------------+      |
|  |         casadi::interpolant()              |      |
|  |  Creates CasADi Function for grid lookup   |      |
|  +-------------------------------------------+      |
|        |                       |                     |
|        v                       v                     |
|  casadi::DM call         casadi::MX call             |
|  (numeric result)        (symbolic expression)       |
|                                                      |
+-----------------------------------------------------+
```

The Hermite method uses a **custom Catmull-Rom implementation** because CasADi doesn't provide a native C1 interpolant:

```cpp
// Catmull-Rom slope estimation at grid point i:
// m[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])

// Cubic Hermite basis functions:
// h00(t) = 2t^3 - 3t^2 + 1
// h10(t) = t^3 - 2t^2 + t
// h01(t) = -2t^3 + 3t^2
// h11(t) = t^3 - t^2

// Interpolated value:
// p(t) = h00*y0 + h10*m0*h + h01*y1 + h11*m1*h
```

This is intentionally **numeric-only** in Metis: symbolic Hermite queries throw an
`InterpolationError` because interval selection requires runtime comparisons that
cannot be traced cleanly into a symbolic graph. For symbolic optimization workflows,
use `InterpolationMethod::BSpline`.

## Diagnostics & Troubleshooting

The interpolation functions throw `metis::InterpolationError` for invalid inputs:

```cpp
try {
    auto result = metis::interpn<double>(points, values, xi);
} catch (const metis::InterpolationError& e) {
    std::cerr << "Interpolation failed: " << e.what() << std::endl;
}
```

| Error | Cause | Solution |
|-------|-------|----------|
| "points cannot be empty" | Empty grid | Provide at least 2 points per dimension |
| "points must be sorted" | Unsorted grid axis | Sort grid coordinates ascending |
| "values_flat size mismatch" | Wrong value count | Ensure `prod(dims)` values in Fortran order |
| "Hermite/Catmull-Rom is numeric-only" | Using Hermite with MX | Use Linear or BSpline for symbolic |
| "BSpline: Need more data points" | Grid < 4 points | Use >= 4 points per dimension for BSpline |

Scattered interpolation best practices:

1. **Grid Resolution**: Higher resolution leads to better accuracy but more memory. Start with 20-30.
2. **Kernel Choice**: ThinPlateSpline is a good default. Use Linear for very few points.
3. **Check Error**: Always check `reconstruction_error()` to validate fit quality.
4. **Extrapolation**: Queries outside convex hull are clamped to boundary.

## See Also

- [Symbolic Computing Guide](symbolic_computing.md) - Working with CasADi symbolic types
- [Numeric Computing Guide](numeric_computing.md) - Metis type system overview
- [Optimization Guide](optimization.md) - Using interpolation as surrogate models
- [`include/metis/math/Interpolate.hpp`](../../include/metis/math/Interpolate.hpp) - Full API implementation
- [`include/metis/math/ScatteredInterpolator.hpp`](../../include/metis/math/ScatteredInterpolator.hpp) - Scattered interpolation API
- [`examples/interpolation/nd_interpolation_demo.cpp`](../../examples/interpolation/nd_interpolation_demo.cpp) - N-D interpolation example
- [`examples/interpolation/scattered_interpolation_demo.cpp`](../../examples/interpolation/scattered_interpolation_demo.cpp) - Scattered data example
