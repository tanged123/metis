# Numeric Computing

Metis is designed to be "Template-First": you write your physics and math logic using generic templates (`T`), and compile them into highly optimized machine code using the Numeric Backend (standard `double` and `Eigen`). Numeric mode provides zero-overhead math calls and Eigen's optimized expression templates with vectorization (AVX, SSE). This guide covers writing generic code, numeric execution, linear algebra, and performance tips for the numeric path.

## Quick Start

```cpp
#include <metis/metis.hpp>

// Write a generic function using metis:: math
template <typename Scalar>
Scalar my_physics(const Scalar& x) {
    return metis::sin(x) + metis::pow(x, 2.0);
}

// Instantiate with double for numeric execution
double result = my_physics(10.0);
std::cout << "Result: " << result << std::endl;
```

## Core API

*   **`metis::NumericScalar`**: Alias for `double`.
*   **`metis::NumericMatrix`**: Alias for `Eigen::MatrixXd`.
*   **`metis::MetisScalar` Concept**: A C++20 concept that matches both `numeric` and `symbolic` scalar types. Use this for templating your functions.
*   `metis::solve(A, b)` -> Uses the default backend policy (`ColPivHouseholderQR` for dense numeric matrices, CasADi default for symbolic, `SparseLU` for sparse numeric input).
*   `metis::solve(A, b, policy)` -> Select dense direct, sparse direct, or iterative Krylov backends explicitly with `metis::LinearSolvePolicy`.
*   `metis::inv(A)` -> Uses `inverse()`.
*   `metis::det(A)` -> Uses `determinant()`.

## Usage Patterns

### Writing Generic Code

To support both numeric and symbolic modes, write your functions as templates:

```cpp
template <typename Scalar>
Scalar my_physics(const Scalar& x) {
    // Use metis:: math namespace, NOT std::
    return metis::sin(x) + metis::pow(x, 2.0);
}
```

### Numeric Execution

When you instantiate your template with `double` (or `metis::NumericScalar`), Metis compiles down to direct C++ math calls and Eigen operations.

*   **Zero Overhead**: `metis::sin` directly calls `std::sin`.
*   **Eigen Speed**: Matrix operations use Eigen's optimized expression templates and vectorization (AVX, SSE).

```cpp
// Instantiation
double result = my_physics(10.0);
```

### Linear Algebra

Metis provides wrappers for common linear algebra operations that work for both backends. In numeric mode, these delegate directly to efficient Eigen Decompositions.

```cpp
metis::NumericMatrix A = /* ... */;
metis::NumericVector b = /* ... */;

// Default solver (ColPivHouseholderQR for dense)
auto x = metis::solve(A, b);

// Explicit policy selection
auto x2 = metis::solve(A, b, metis::LinearSolvePolicy::SparseDirect);
```

## Advanced Usage

*   **Avoid `auto` types in return signatures** of public APIs if you can use `Scalar` or `MetisMatrix<Scalar>`.
*   **Use `metis::where` instead of `if/else`**: This ensures your code works in symbolic mode and compiles to efficient branchless select instructions (where possible) in numeric mode.
*   **Pass by Reference**: `const Scalar&` avoids copying expensive types in symbolic mode, and is negligible for doubles.

## See Also

- [Symbolic Computing Guide](symbolic_computing.md) - The symbolic counterpart to numeric mode
- [Math Functions Guide](math_functions.md) - All available `metis::` math functions
- [`examples/intro/numeric_intro.cpp`](../../examples/intro/numeric_intro.cpp) - Benchmark of numeric operations
- [`include/metis/core/MetisTypes.hpp`](../../include/metis/core/MetisTypes.hpp) - Type alias definitions
