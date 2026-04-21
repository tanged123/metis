# Math Functions

Metis provides a comprehensive set of math functions that work seamlessly with both numeric (`double`) and symbolic (`SymbolicScalar`) types. When called with a `metis::SymbolicScalar` argument, C++ automatically finds the correct overload via ADL (Argument-Dependent Lookup), so you can write `sin(x)` or `metis::sin(x)` interchangeably. These functions are the foundation for writing generic template code that compiles to both fast numeric execution and symbolic graph construction.

## Quick Start

```cpp
#include <metis/metis.hpp>

// Works with both double and SymbolicScalar
template <typename Scalar>
Scalar my_function(const Scalar& x, const Scalar& y) {
    return metis::sin(x) + metis::pow(y, 2.0) + metis::exp(-x * y);
}

// Numeric usage
double result = my_function(1.0, 2.0);

// Symbolic usage
auto x = metis::sym("x");
auto y = metis::sym("y");
auto expr = my_function(x, y);
```

## Core API

### Trigonometric
- `metis::sin(x)`, `metis::cos(x)`, `metis::tan(x)`
- `metis::asin(x)`, `metis::acos(x)`, `metis::atan(x)`, `metis::atan2(y, x)`

### Exponential & Logarithmic
- `metis::exp(x)`, `metis::log(x)`, `metis::log10(x)`
- `metis::pow(base, exp)`, `metis::sqrt(x)`

### Hyperbolic
- `metis::sinh(x)`, `metis::cosh(x)`, `metis::tanh(x)`

### Utility
- `metis::abs(x)`, `metis::fabs(x)`
- `metis::floor(x)`, `metis::ceil(x)`
- `metis::fmin(a, b)`, `metis::fmax(a, b)`

### Control Flow
- `metis::where(condition, if_true, if_false)` - Branch-free conditional that works in both numeric and symbolic mode

## Usage Patterns

### ADL (Argument-Dependent Lookup)

When you call a math function with a `metis::SymbolicScalar` argument, C++ automatically finds the correct function via ADL:

```cpp
metis::SymbolicScalar x = metis::sym("x");
auto y = sin(x);  // Finds metis::sin via ADL
auto z = pow(x, 2);  // Finds metis::pow via ADL
```

For mixed types or explicit calls:
```cpp
auto y = metis::sin(x);  // Always works
auto z = metis::pow(x, 2.0);  // Mixed symbolic/numeric
```

### Branch-Free Conditionals with `metis::where`

Use `metis::where` instead of `if/else` to ensure your code works in symbolic mode:

```cpp
template <typename Scalar>
Scalar safe_sqrt(const Scalar& x) {
    return metis::where(x > 0.0, metis::sqrt(x), Scalar(0.0));
}
```

In numeric mode, this compiles to efficient branchless select instructions (where possible). In symbolic mode, it builds a valid conditional node in the computation graph.

### Convenience Header

For cleaner code, use the convenience header:

```cpp
#include <metis/using.hpp>

// Now sym, sin, cos, pow, where are in scope
auto x = sym("x");
auto y = sin(x) + pow(x, 2);
```

## See Also

- [Numeric Computing Guide](numeric_computing.md) - Writing generic template code
- [Symbolic Computing Guide](symbolic_computing.md) - Building symbolic expressions
- [`include/metis/math/Trig.hpp`](../../include/metis/math/Trig.hpp) - Trigonometric function implementations
- [`include/metis/math/Arithmetic.hpp`](../../include/metis/math/Arithmetic.hpp) - Arithmetic function implementations
- [`include/metis/math/Logic.hpp`](../../include/metis/math/Logic.hpp) - `where` and branching logic
- [`examples/math/branching_logic.cpp`](../../examples/math/branching_logic.cpp) - Branching logic examples
