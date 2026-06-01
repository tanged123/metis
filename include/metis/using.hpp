#pragma once

/**
 * @file using.hpp
 * @brief Convenience header bringing common Metis symbols into scope
 *
 * Include this header for cleaner code without explicit metis:: prefixes.
 *
 * @code
 * #include <metis/using.hpp>
 *
 * auto x = sym("x");
 * auto y = sin(x) + pow(x, 2);
 * auto z = where(x > 0, x, -x);
 * @endcode
 *
 * @warning This header pollutes the global namespace. Use with care in
 * library code - prefer explicit metis:: prefixes there.
 */

#include "metis.hpp"

// Bring common Metis symbols into the global namespace
using metis::sym;
using metis::sym_vec;
using metis::sym_vec_pair;
using metis::sym_vector;

// Type aliases
using metis::DenseLinearSolver;
using metis::IterativeKrylovSolver;
using metis::IterativePreconditioner;
using metis::LinearSolveBackend;
using metis::LinearSolvePolicy;
using metis::NumericMatrix;
using metis::NumericScalar;
using metis::NumericVector;
using metis::PolynomialChaosBasis;
using metis::PolynomialChaosBasisOptions;
using metis::PolynomialChaosDimension;
using metis::PolynomialChaosFamily;
using metis::PolynomialChaosTerm;
using metis::PolynomialChaosTruncation;
using metis::SmolyakQuadratureOptions;
using metis::SparseDirectLinearSolver;
using metis::StochasticQuadratureGrid;
using metis::StochasticQuadratureRule;
using metis::StructuralDiagnosticsOptions;
using metis::StructuralDiagnosticsReport;
using metis::StructuralSensitivityOptions;
using metis::StructuralSensitivityReport;
using metis::SymbolicMatrix;
using metis::SymbolicScalar;
using metis::SymbolicVector;
using metis::UnivariateQuadratureRule;

// Conversion helpers
using metis::as_mx;
using metis::as_vector;
using metis::to_eigen;
using metis::to_mx;

// Structural diagnostics
using metis::analyze_structural_diagnostics;
using metis::analyze_structural_identifiability;
using metis::analyze_structural_observability;

// Math functions
using metis::abs;
using metis::acos;
using metis::asin;
using metis::atan;
using metis::atan2;
using metis::ceil;
using metis::cos;
using metis::cosh;
using metis::exp;
using metis::floor;
using metis::log;
using metis::log10;
using metis::pow;
using metis::sin;
using metis::sinh;
using metis::sqrt;
using metis::tan;
using metis::tanh;

// Control flow
using metis::where;

// Calculus
using metis::gradient;
using metis::hermite_dimension;
using metis::hessian;
using metis::hessian_vector_product;
using metis::jacobi_dimension;
using metis::jacobian;
using metis::lagrangian_hessian_vector_product;
using metis::laguerre_dimension;
using metis::legendre_dimension;
using metis::pce_mean;
using metis::pce_polynomial;
using metis::pce_projection_coefficients;
using metis::pce_regression_coefficients;
using metis::pce_squared_norm;
using metis::pce_variance;
using metis::smolyak_sparse_grid;
using metis::stochastic_quadrature_level;
using metis::stochastic_quadrature_rule;
using metis::tensor_product_quadrature;

// Spacing
using metis::linspace;
