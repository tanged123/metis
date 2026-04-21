/**
 * @file polynomial_chaos_demo.cpp
 * @brief Demonstrate polynomial chaos basis construction, projection, and regression.
 *
 * This example shows:
 * 1. Multidimensional basis construction and multi-index ordering.
 * 2. Built-in stochastic quadrature rules for projection and sparse grids.
 * 3. Symbolic mean/variance gradients through fitted PCE coefficients.
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <string>
#include <vector>

using namespace metis;

namespace {

NumericMatrix make_sample_matrix(const NumericVector &samples) {
    NumericMatrix out(samples.size(), 1);
    out.col(0) = samples;
    return out;
}

void print_multi_index(const std::vector<int> &multi_index) {
    std::cout << "[";
    for (std::size_t i = 0; i < multi_index.size(); ++i) {
        if (i != 0u) {
            std::cout << ", ";
        }
        std::cout << multi_index[i];
    }
    std::cout << "]";
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Polynomial Chaos Demo ===\n\n";

    {
        std::cout << "Case 1: total-order basis construction in two stochastic dimensions\n";

        PolynomialChaosBasis basis({legendre_dimension(), hermite_dimension()}, 2);
        std::cout << "  dimension   = " << basis.dimension() << "\n";
        std::cout << "  order       = " << basis.order() << "\n";
        std::cout << "  term count  = " << basis.size() << "\n";
        std::cout << "  normalized  = " << (basis.normalized() ? "true" : "false") << "\n";
        std::cout << "  terms:\n";
        for (std::size_t i = 0; i < basis.terms().size(); ++i) {
            std::cout << "    psi[" << i << "] multi-index = ";
            print_multi_index(basis.terms()[i].multi_index);
            std::cout << ", squared norm = " << basis.terms()[i].squared_norm << "\n";
        }

        NumericVector point(2);
        point << 0.25, -0.4;
        NumericVector basis_values = basis.evaluate(point);
        std::cout << "  basis values at xi = [0.25, -0.4]^T:\n"
                  << basis_values.transpose() << "\n\n";
    }

    {
        std::cout << "Case 2: built-in stochastic quadrature for projection and sparse grids\n";

        PolynomialChaosBasis basis({legendre_dimension()}, 2);

        const UnivariateQuadratureRule projection_rule =
            stochastic_quadrature_rule(legendre_dimension(), 3);
        auto make_samples = [&](const NumericVector &nodes, const SymbolicScalar &a) {
            SymbolicVector values(nodes.size());
            for (Eigen::Index i = 0; i < nodes.size(); ++i) {
                const double xi = nodes(i);
                values(i) = (1.0 + a) * pce_polynomial(legendre_dimension(), 0, xi) +
                            0.5 * pce_polynomial(legendre_dimension(), 1, xi) +
                            (0.25 * a) * pce_polynomial(legendre_dimension(), 2, xi);
            }
            return values;
        };

        SymbolicScalar a = sym("a");

        SymbolicVector coeffs_projection = pce_projection_coefficients(
            basis, projection_rule, make_samples(projection_rule.nodes, a));
        SymbolicScalar mean = pce_mean(coeffs_projection);
        SymbolicScalar variance = pce_variance(basis, coeffs_projection);

        const StochasticQuadratureGrid sparse_grid =
            smolyak_sparse_grid({legendre_dimension(), hermite_dimension()}, 3);
        SymbolicScalar sparse_moment = 0.0;
        for (Eigen::Index i = 0; i < sparse_grid.samples.rows(); ++i) {
            const double x = sparse_grid.samples(i, 0);
            const double y = sparse_grid.samples(i, 1);
            sparse_moment += sparse_grid.weights(i) * (x * x + a * y * y);
        }

        Function summary("pce_quadrature_summary", {a},
                         {to_mx(coeffs_projection), mean, variance, sparse_moment});

        const auto outputs = summary(2.0);
        const NumericMatrix coeffs_projection_val = outputs[0];

        std::cout << "  Legendre Gauss rule nodes:\n" << projection_rule.nodes.transpose() << "\n";
        std::cout << "  Legendre Gauss rule weights:\n"
                  << projection_rule.weights.transpose() << "\n";
        std::cout << "  projection coefficients at a = 2:\n"
                  << coeffs_projection_val.transpose() << "\n";
        std::cout << "  mean(a=2)      = " << outputs[1](0, 0) << "\n";
        std::cout << "  variance(a=2)  = " << outputs[2](0, 0) << "\n";
        std::cout << "  sparse-grid E[x^2 + a y^2] at a=2 = " << outputs[3](0, 0) << "\n";
        std::cout << "  Smolyak sparse grid points = " << sparse_grid.samples.rows()
                  << " versus dense 5x3 tensor = 15\n\n";
    }

    {
        std::cout << "Case 3: regression with symbolic sample values and moment gradients\n";

        PolynomialChaosBasis basis({legendre_dimension()}, 2);

        NumericVector regression_nodes(6);
        regression_nodes << -1.0, -0.6, -0.2, 0.2, 0.6, 1.0;
        const NumericMatrix regression_samples = make_sample_matrix(regression_nodes);

        SymbolicScalar a = sym("a");
        SymbolicVector sample_values(regression_nodes.size());
        for (Eigen::Index i = 0; i < regression_nodes.size(); ++i) {
            const double xi = regression_nodes(i);
            sample_values(i) = (1.0 + a) * pce_polynomial(legendre_dimension(), 0, xi) +
                               0.5 * pce_polynomial(legendre_dimension(), 1, xi) +
                               (0.25 * a) * pce_polynomial(legendre_dimension(), 2, xi);
        }

        SymbolicVector coeffs_regression =
            pce_regression_coefficients(basis, regression_samples, sample_values, 0.0);
        SymbolicScalar mean = pce_mean(coeffs_regression);
        SymbolicScalar variance = pce_variance(basis, coeffs_regression);

        Function summary(
            "pce_regression_summary", {a},
            {to_mx(coeffs_regression), mean, variance, jacobian(mean, a), jacobian(variance, a)});

        const auto outputs = summary(2.0);
        const NumericMatrix coeffs_regression_val = outputs[0];

        std::cout << "  regression coefficients at a = 2:\n"
                  << coeffs_regression_val.transpose() << "\n";
        std::cout << "  mean(a=2)      = " << outputs[1](0, 0) << "\n";
        std::cout << "  variance(a=2)  = " << outputs[2](0, 0) << "\n";
        std::cout << "  dmean/da       = " << outputs[3](0, 0) << "\n";
        std::cout << "  dvariance/da   = " << outputs[4](0, 0) << "\n\n";
    }

    std::cout << "Takeaway:\n";
    std::cout << "  - PolynomialChaosBasis builds the multi-index set and evaluates the basis\n";
    std::cout << "  - stochastic_quadrature_rule(...), tensor_product_quadrature(...), and\n";
    std::cout << "    smolyak_sparse_grid(...) provide structured probability-measure nodes\n";
    std::cout << "  - pce_projection_coefficients(...) and pce_regression_coefficients(...)\n";
    std::cout << "    recover coefficients from those quadrature samples or plain collocation\n";
    std::cout << "  - because the fitted coefficients remain symbolic, pce_mean(...) and\n";
    std::cout << "    pce_variance(...) can be differentiated directly with CasADi/Metis\n";

    return 0;
}
