# Symbolic Computing

Metis provides a powerful symbolic computing layer built on top of CasADi, abstracted to feel like native C++ with Eigen integration. This allows you to compute derivatives, generate code, and optimize systems using the same code you write for simulation. Symbolic mode works by building a computational graph instead of executing immediately, enabling automatic differentiation, sensitivity analysis, and matrix-free second-order products.

## Quick Start

```cpp
#include <metis/metis.hpp>

// Create a symbolic variable
auto x = metis::sym("x");

// Build an expression (creates a computation graph)
auto f = x * x;

// Compute the Jacobian symbolically
auto J = metis::jacobian({f}, {x});

// Compile into a callable function
metis::Function fn({x}, {f, J});

// Evaluate numerically
auto res = fn(3.0);
std::cout << "f(3) = " << res[0] << ", f'(3) = " << res[1] << std::endl;
```

## Core API

*   **`metis::SymbolicScalar`**: Alias for `casadi::MX`. Represents a symbolic value or expression.
*   **`metis::SymbolicMatrix`**: Alias for `Eigen::Matrix<casadi::MX, -1, -1>`. Allows you to use familiar Eigen syntax (block operations, coeff access) on symbolic variables.
*   **`metis::sym(name)`**: Create a scalar symbolic variable.
*   **`metis::sym(name, n)`**: Create a column vector symbolic variable (`n x 1`).
*   **`metis::sym(name, r, c)`**: Create a matrix symbolic variable (`r x c`).
*   **`metis::Function({inputs}, {outputs})`**: Compile symbolic expressions into a callable function.
*   **`metis::jacobian({outputs}, {inputs})`**: Compute the Jacobian of outputs with respect to inputs.

## Usage Patterns

### Creating Variables

Use the `metis::sym` helper to create symbolic variables cleanly.

```cpp
// Scalar variable "x"
auto x = metis::sym("x");

// Column vector "v" (3x1)
auto v = metis::sym("v", 3);

// Matrix "A" (2x2)
auto A = metis::sym("A", 2, 2);
```

### Building Expressions

You can use standard arithmetic operators (`+`, `-`, `*`, `/`) and Metis math functions. These build a computational graph instead of executing immediately.

```cpp
auto y = metis::sin(x) + metis::pow(x, 2.0);
auto z = A * v; // Matrix multiplication
```

### Defining Functions (`metis::Function`)

To evaluate expressions numerically, you must wrap them in a `metis::Function`. This wrapper handles type conversion between Eigen and CasADi.

```cpp
// Define f(x, v) -> y
metis::Function f({x, v}, {y});

// Evaluate with numeric input (doubles/Eigen)
// Returns std::vector<Eigen::MatrixXd>
auto result = f(1.5, Eigen::Vector3d::Zero());
std::cout << result[0] << std::endl;
```

**Features:**
*   **Automatic Naming**: You don't need to provide a string name; one is generated automatically.
*   **Eigen Integration**: Inputs and outputs are converted to/from Eigen types seamlessly.

### Automatic Differentiation (`metis::jacobian`)

Compute derivatives effortlessly. Metis handles the tedious task of concatenating variables for CasADi.

```cpp
// Compute Jacobian J = dy/dx
auto J = metis::jacobian({y}, {x});

// Compute Jacobian w.r.t multiple variables: J = dy/d[x, v]
auto J_full = metis::jacobian({y}, {x, v});
```

## Advanced Usage

### Sensitivity Regime Switching

For compiled `metis::Function` objects, Metis can choose between forward and adjoint
Jacobian construction automatically based on parameter count, output count, and optional
trajectory hints.

```cpp
metis::Function f("f", {x}, {y});

auto rec = metis::select_sensitivity_regime(
    f,
    0,      // output block
    0,      // input block
    400,    // horizon length hint
    true    // stiff trajectory hint
);

auto J_fun = metis::sensitivity_jacobian(f, 0, 0, 400, true);
auto J = J_fun.eval(x_val);
```

Use `rec.integrator_options()` when you want the same recommendation expressed as
CasADi/SUNDIALS options (`nfwd` or `nadj`, plus checkpoint settings for long-horizon adjoints).

### Matrix-Free Second-Order Products

For large optimization problems, you often want `H * v` without ever forming the dense Hessian.
Metis exposes matrix-free Hessian-vector products for both plain scalar expressions and
Lagrangians:

```cpp
auto x = metis::sym("x", 3);
auto v = metis::sym("v", 3);
auto lam = metis::sym("lam", 2);

casadi::MX x0 = x(0);
casadi::MX x1 = x(1);
casadi::MX x2 = x(2);

auto objective = x0 * x0 + x1 * x2 + metis::sin(x2);
auto constraints = casadi::MX::vertcat({x0 + x1, x1 * x2});

auto hvp = metis::hessian_vector_product(objective, x, v);
auto lag_hvp =
    metis::lagrangian_hessian_vector_product(objective, constraints, x, lam, v);
```

For compiled `metis::Function` objects, the wrappers return another `metis::Function`:

```cpp
metis::Function model("model", {x}, {objective});
metis::Function hvp_fun = metis::hessian_vector_product(model, 0, 0);

auto hv = hvp_fun.eval(x_val, v_val);   // original inputs..., then direction v
```

The Lagrangian variant appends the multiplier block first and the direction block last:

```cpp
metis::Function nlp_terms("nlp_terms", {x}, {objective, constraints});
metis::Function lag_hvp_fun =
    metis::lagrangian_hessian_vector_product(nlp_terms, 0, 1, 0);

auto hv = lag_hvp_fun.eval(x_val, lam_val, v_val);
```

These products use CasADi's forward-over-reverse AD internally, so the dense Hessian is never
constructed as an intermediate.

## See Also

- [Numeric Computing Guide](numeric_computing.md) - The numeric counterpart to symbolic mode
- [Math Functions Guide](math_functions.md) - All available `metis::` math functions
- [Optimization Guide](optimization.md) - Using symbolic expressions in optimization
- [`include/metis/core/Function.hpp`](../../include/metis/core/Function.hpp) - `metis::Function` implementation
- [`include/metis/math/AutoDiff.hpp`](../../include/metis/math/AutoDiff.hpp) - Jacobian and Hessian-vector product API
- [`examples/intro/energy_intro.cpp`](../../examples/intro/energy_intro.cpp) - Introductory symbolic example
