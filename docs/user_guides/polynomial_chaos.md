# Polynomial Chaos

Metis includes a PCE (Polynomial Chaos Expansion) layer in `PolynomialChaos.hpp` for spectral uncertainty quantification. It provides Askey-scheme basis polynomials, multidimensional basis construction, structured stochastic quadrature rules and sparse grids, coefficient recovery from projection or regression samples, and symbolic mean and variance propagation. The key benefit is that fitted PCE coefficients can remain symbolic `casadi::MX` expressions, so statistical moments can be differentiated with respect to design variables without nesting Monte Carlo inside an optimizer. This works in both numeric and symbolic modes.

## Quick Start

```cpp
#include <metis/metis.hpp>

// Build a 1D Legendre basis of degree <= 2
metis::PolynomialChaosBasis basis({metis::legendre_dimension()}, 2);

// Get quadrature rule for projection
auto rule = metis::stochastic_quadrature_rule(metis::legendre_dimension(), 3);

// Evaluate your model at quadrature nodes
metis::NumericVector values(rule.nodes.size());
for (Eigen::Index i = 0; i < rule.nodes.size(); ++i) {
    values(i) = std::sin(rule.nodes(i));  // Example: sin(xi)
}

// Recover PCE coefficients via projection
metis::NumericVector coeffs = metis::pce_projection_coefficients(basis, rule, values);

// Extract mean and variance
double mean = metis::pce_mean(coeffs);
double variance = metis::pce_variance(basis, coeffs);
```

## Core API

*   **`metis::hermite_dimension()`**: Hermite basis for standard normal variables.
*   **`metis::legendre_dimension()`**: Legendre basis for uniform variables on `[-1, 1]`.
*   **`metis::jacobi_dimension(alpha, beta)`**: Jacobi basis for beta-family variables on `[-1, 1]`.
*   **`metis::laguerre_dimension()`**: Laguerre basis for exponential/gamma-family variables on `[0, inf)`.
*   **`metis::pce_polynomial(dim, order, x)`**: Evaluate a univariate basis function (normalized by default).
*   **`metis::pce_squared_norm(dim, order, normalized)`**: Get the squared norm of a basis function.
*   **`metis::PolynomialChaosBasis(dims, degree, opts)`**: Build a multidimensional basis with total-order or tensor-product truncation.
*   **`metis::pce_projection_coefficients(basis, rule, values)`**: Recover coefficients via weighted projection.
*   **`metis::pce_regression_coefficients(basis, samples, values, ridge)`**: Recover coefficients via regression.
*   **`metis::pce_mean(coeffs)`**: Extract the mean from PCE coefficients.
*   **`metis::pce_variance(basis, coeffs)`**: Extract the variance from PCE coefficients.

## Usage Patterns

### Supported Families

Use one `PolynomialChaosDimension` per uncertain input:

```cpp
auto gaussian = metis::hermite_dimension();
auto uniform = metis::legendre_dimension();
auto beta_like = metis::jacobi_dimension(1.0, 2.0);
auto exponential = metis::laguerre_dimension();  // alpha = 0
```

The canonical variable domains are:

- Hermite: standard normal variable
- Legendre: uniform variable on `[-1, 1]`
- Jacobi: mapped beta-family variable on `[-1, 1]`
- Laguerre: exponential / gamma-family variable on `[0, inf)`

Evaluate one univariate basis function with:

```cpp
double psi2 = metis::pce_polynomial(uniform, 2, 0.25);
double raw_norm = metis::pce_squared_norm(uniform, 2, false);
```

By default `pce_polynomial(...)` returns the **normalized** basis function. Pass `normalized = false` if you want the classical raw polynomial instead.

### Building a Multidimensional Basis

`PolynomialChaosBasis` builds the multi-index set and caches each term's squared norm:

```cpp
metis::PolynomialChaosBasis basis(
    {metis::legendre_dimension(), metis::hermite_dimension()},
    2);
```

This constructs a **total-order** basis of degree `<= 2`. To switch to tensor-product truncation:

```cpp
metis::PolynomialChaosBasis basis(
    {metis::legendre_dimension(), metis::hermite_dimension()},
    2,
    metis::PolynomialChaosBasisOptions{
        .truncation = metis::PolynomialChaosTruncation::TensorProduct,
        .normalized = true
    });
```

Inspect the generated terms and evaluate:

```cpp
for (const auto &term : basis.terms()) {
    // term.multi_index, term.squared_norm
}

metis::NumericVector xi(2);
xi << 0.25, -0.4;
metis::NumericVector psi = basis.evaluate(xi);
```

### Sample Matrix Layout

For projection and regression, Metis expects the sample matrix as:

- rows: sample points
- columns: stochastic dimensions

```cpp
// For one uncertain variable:
metis::NumericVector nodes = metis::lgl_nodes(5);
metis::NumericMatrix samples(nodes.size(), 1);
samples.col(0) = nodes;

// For d uncertain variables and N sample points, samples should be N x d.
```

### Structured Quadrature Rules

For probability-measure quadrature, prefer `Quadrature.hpp` over manually assembling nodes and weights:

```cpp
auto rule =
    metis::stochastic_quadrature_rule(metis::legendre_dimension(), 3);
```

The returned weights already integrate against the probability measure, so for Legendre/uniform variables they sum to `1` directly. You do not need the extra `0.5` scaling that raw `lgl_weights(...)` require.

For bounded-support families, nested refinement is available:

```cpp
auto coarse =
    metis::stochastic_quadrature_level(
        metis::legendre_dimension(),
        2,
        metis::StochasticQuadratureRule::ClenshawCurtis);
```

For higher-dimensional problems:

```cpp
auto sparse_grid = metis::smolyak_sparse_grid(
    {metis::legendre_dimension(), metis::hermite_dimension()},
    3);
```

See [Stochastic Quadrature Guide](stochastic_quadrature.md) for the full rule set and sparse-grid details.

### Projection Coefficients

Use weighted projection when you already have quadrature-like nodes and weights:

```cpp
metis::PolynomialChaosBasis basis({metis::legendre_dimension()}, 2);

auto rule =
    metis::stochastic_quadrature_rule(metis::legendre_dimension(), 3);

metis::NumericVector values(rule.nodes.size());
for (Eigen::Index i = 0; i < rule.nodes.size(); ++i) {
    values(i) = 1.2 * metis::pce_polynomial(metis::legendre_dimension(), 0, rule.nodes(i))
              + 0.5 * metis::pce_polynomial(metis::legendre_dimension(), 1, rule.nodes(i))
              - 0.7 * metis::pce_polynomial(metis::legendre_dimension(), 2, rule.nodes(i));
}

metis::NumericVector coeffs =
    metis::pce_projection_coefficients(basis, rule, values);
```

If you already have your own `samples` matrix and `weights` vector, the original overloads still work:

```cpp
metis::NumericVector coeffs =
    metis::pce_projection_coefficients(basis, samples, weights, values);
```

### Regression Coefficients

Use regression when you have collocation samples without matching quadrature weights:

```cpp
// continued from "Projection Coefficients" above (reuses basis, values)

metis::NumericVector nodes(6);
nodes << -1.0, -0.6, -0.2, 0.2, 0.6, 1.0;

metis::NumericMatrix samples(nodes.size(), 1);
samples.col(0) = nodes;

metis::NumericVector coeffs =
    metis::pce_regression_coefficients(basis, samples, values, 0.0);
```

The final `ridge` argument is an optional Tikhonov regularization parameter. When `ridge = 0`, Metis rejects rank-deficient design matrices instead of silently returning a bad fit.

### Symbolic Moments

The main Metis-specific payoff is that the sampled response can stay symbolic:

```cpp
// continued from "Regression Coefficients" above (reuses basis, nodes, samples)

auto a = metis::sym("a");
metis::SymbolicVector sample_values(nodes.size());

for (Eigen::Index i = 0; i < nodes.size(); ++i) {
    sample_values(i) = (1.0 + a) * metis::pce_polynomial(metis::legendre_dimension(), 0, nodes(i))
                     + 0.5 * metis::pce_polynomial(metis::legendre_dimension(), 1, nodes(i))
                     + (0.25 * a) * metis::pce_polynomial(metis::legendre_dimension(), 2, nodes(i));
}

auto coeffs = metis::pce_regression_coefficients(basis, samples, sample_values, 0.0);
auto mean = metis::pce_mean(coeffs);
auto variance = metis::pce_variance(basis, coeffs);
```

Because `coeffs`, `mean`, and `variance` are symbolic expressions, you can differentiate them directly:

```cpp
auto dmean_da = metis::jacobian(mean, a);
auto dvariance_da = metis::jacobian(variance, a);
```

With the default normalized basis:

- `pce_mean(coeffs)` returns the constant coefficient
- `pce_variance(basis, coeffs)` returns the sum of squared non-constant coefficients

If you build the basis with `.normalized = false`, Metis uses each term's stored squared norm instead.

## See Also

- [Stochastic Quadrature Guide](stochastic_quadrature.md) - Full rule set and sparse-grid details
- [Symbolic Computing Guide](symbolic_computing.md) - Symbolic expressions and differentiation
- [`examples/math/polynomial_chaos_demo.cpp`](../../examples/math/polynomial_chaos_demo.cpp) - Full PCE demo
- [`include/metis/math/PolynomialChaos.hpp`](../../include/metis/math/PolynomialChaos.hpp) - PCE API implementation
- [`include/metis/math/Quadrature.hpp`](../../include/metis/math/Quadrature.hpp) - Quadrature rule API
