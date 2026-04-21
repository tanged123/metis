#pragma once
/**
 * @file OrthogonalPolynomials.hpp
 * @brief Orthogonal polynomial evaluation, spectral nodes, weights, and differentiation matrices
 * @see PolynomialChaos.hpp, Quadrature.hpp
 */

#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include <cmath>
#include <limits>
#include <utility>

namespace metis {

std::pair<double, double> legendre_poly(int n, double x);

namespace detail {

/// Tolerance for zero-length integration intervals in Birkhoff matrix
constexpr double polynomial_root_tolerance = 1e-15;

inline std::pair<NumericVector, NumericVector> gauss_legendre_rule(int n) {
    if (n < 1) {
        throw InvalidArgument("gauss_legendre_rule: n must be >= 1");
    }

    NumericVector nodes(n);
    NumericVector weights(n);

    const int m = (n + 1) / 2;
    const double pi = std::acos(-1.0);
    const double tol = 64.0 * std::numeric_limits<double>::epsilon();

    for (int i = 0; i < m; ++i) {
        double z = std::cos(pi * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));

        for (int iter = 0; iter < 100; ++iter) {
            const auto [pn, dpn] = legendre_poly(n, z);
            const double delta = pn / dpn;
            z -= delta;
            if (std::abs(delta) < tol) {
                break;
            }
        }

        const auto [pn, dpn] = legendre_poly(n, z);
        (void)pn;
        const double w = 2.0 / ((1.0 - z * z) * dpn * dpn);

        nodes(i) = -z;
        nodes(n - 1 - i) = z;
        weights(i) = w;
        weights(n - 1 - i) = w;
    }

    return {nodes, weights};
}

} // namespace detail

/**
 * @brief Gauss-Legendre nodes and weights on [-1, 1]
 * @param n Number of quadrature points
 * @return Pair of (nodes, weights)
 */
inline std::pair<NumericVector, NumericVector> gauss_legendre_rule(int n) {
    return detail::gauss_legendre_rule(n);
}

/**
 * @brief Evaluate Legendre polynomial P_n(x) and derivative P'_n(x)
 * @param n Polynomial degree
 * @param x Evaluation point
 * @return Pair of (P_n(x), P'_n(x))
 */
inline std::pair<double, double> legendre_poly(int n, double x) {
    if (n < 0) {
        throw InvalidArgument("legendre_poly: n must be >= 0");
    }

    if (n == 0) {
        return {1.0, 0.0};
    }
    if (n == 1) {
        return {x, 1.0};
    }

    double p_nm1 = 1.0;
    double p_n = x;
    for (int k = 1; k < n; ++k) {
        const double p_np1 =
            ((2.0 * static_cast<double>(k) + 1.0) * x * p_n - static_cast<double>(k) * p_nm1) /
            (static_cast<double>(k) + 1.0);
        p_nm1 = p_n;
        p_n = p_np1;
    }

    const double nn = static_cast<double>(n);
    const double endpoint_tol = 64.0 * std::numeric_limits<double>::epsilon();
    const double one_minus_abs_x = std::abs(1.0 - std::abs(x));

    double dp_n = 0.0;
    if (one_minus_abs_x < endpoint_tol) {
        const double endpoint = 0.5 * nn * (nn + 1.0);
        if (x >= 0.0) {
            dp_n = endpoint;
        } else {
            const double sign = ((n + 1) % 2 == 0) ? 1.0 : -1.0; // (-1)^(n+1)
            dp_n = sign * endpoint;
        }
    } else {
        dp_n = nn * (x * p_n - p_nm1) / (x * x - 1.0);
    }

    return {p_n, dp_n};
}

/**
 * @brief Evaluate P_n(x) at each x in vector
 * @param n Polynomial degree
 * @param x Evaluation points
 * @return Vector of P_n values
 */
inline NumericVector legendre_poly_vec(int n, const NumericVector &x) {
    NumericVector out(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        out(i) = legendre_poly(n, x(i)).first;
    }
    return out;
}

/**
 * @brief Compute Legendre-Gauss-Lobatto nodes on [-1, 1]
 * @param N Number of nodes (>= 2)
 * @return Node vector
 */
inline NumericVector lgl_nodes(int N) {
    if (N < 2) {
        throw InvalidArgument("lgl_nodes: N must be >= 2");
    }

    NumericVector nodes(N);
    nodes(0) = -1.0;
    nodes(N - 1) = 1.0;
    if (N == 2) {
        return nodes;
    }

    const int n = N - 1;
    const double pi = std::acos(-1.0);
    const double tol = 32.0 * std::numeric_limits<double>::epsilon();

    for (int j = 1; j < N - 1; ++j) {
        double x = -std::cos(pi * static_cast<double>(j) / static_cast<double>(N - 1));

        for (int iter = 0; iter < 100; ++iter) {
            const auto [pn, dpn] = legendre_poly(n, x);
            const double denom = 1.0 - x * x;
            const double ddpn = (2.0 * x * dpn - static_cast<double>(n) * (n + 1) * pn) / denom;

            const double delta = dpn / ddpn;
            x -= delta;

            if (std::abs(delta) < tol) {
                break;
            }
        }

        nodes(j) = x;
    }

    return nodes;
}

/**
 * @brief Compute Chebyshev-Gauss-Lobatto nodes on [-1, 1]
 * @param N Number of nodes (>= 2)
 * @return Node vector
 */
inline NumericVector cgl_nodes(int N) {
    if (N < 2) {
        throw InvalidArgument("cgl_nodes: N must be >= 2");
    }

    NumericVector nodes(N);
    const double pi = std::acos(-1.0);
    for (int j = 0; j < N; ++j) {
        nodes(j) = -std::cos(pi * static_cast<double>(j) / static_cast<double>(N - 1));
    }
    return nodes;
}

/**
 * @brief LGL quadrature weights
 * @param N Number of nodes
 * @param nodes LGL node vector
 * @return Weight vector
 */
inline NumericVector lgl_weights(int N, const NumericVector &nodes) {
    if (N < 2) {
        throw InvalidArgument("lgl_weights: N must be >= 2");
    }
    if (nodes.size() != N) {
        throw InvalidArgument("lgl_weights: nodes size mismatch");
    }

    NumericVector w(N);
    const double scale = 2.0 / (static_cast<double>(N) * static_cast<double>(N - 1));
    const int n = N - 1;
    for (int i = 0; i < N; ++i) {
        const double pn = legendre_poly(n, nodes(i)).first;
        w(i) = scale / (pn * pn);
    }
    return w;
}

/**
 * @brief CGL (Clenshaw-Curtis) quadrature weights on [-1, 1]
 * @param N Number of nodes
 * @param nodes CGL node vector
 * @return Weight vector
 */
inline NumericVector cgl_weights(int N, const NumericVector &nodes) {
    if (N < 2) {
        throw InvalidArgument("cgl_weights: N must be >= 2");
    }
    if (nodes.size() != N) {
        throw InvalidArgument("cgl_weights: nodes size mismatch");
    }

    if (N == 2) {
        NumericVector w(2);
        w(0) = 1.0;
        w(1) = 1.0;
        return w;
    }

    const int n = N - 1;
    const double pi = std::acos(-1.0);
    NumericVector theta(N);
    for (int j = 0; j < N; ++j) {
        theta(j) = pi * static_cast<double>(j) / static_cast<double>(n);
    }

    NumericVector w = NumericVector::Zero(N);
    NumericVector v = NumericVector::Ones(N - 2);

    if (n % 2 == 0) {
        const double w0 = 1.0 / (static_cast<double>(n) * static_cast<double>(n) - 1.0);
        w(0) = w0;
        w(N - 1) = w0;

        for (int k = 1; k < n / 2; ++k) {
            const double denom = 4.0 * static_cast<double>(k) * static_cast<double>(k) - 1.0;
            for (int j = 1; j < N - 1; ++j) {
                v(j - 1) -= 2.0 * std::cos(2.0 * static_cast<double>(k) * theta(j)) / denom;
            }
        }

        const double denom = static_cast<double>(n) * static_cast<double>(n) - 1.0;
        for (int j = 1; j < N - 1; ++j) {
            v(j - 1) -= std::cos(static_cast<double>(n) * theta(j)) / denom;
        }
    } else {
        const double w0 = 1.0 / (static_cast<double>(n) * static_cast<double>(n));
        w(0) = w0;
        w(N - 1) = w0;

        for (int k = 1; k <= (n - 1) / 2; ++k) {
            const double denom = 4.0 * static_cast<double>(k) * static_cast<double>(k) - 1.0;
            for (int j = 1; j < N - 1; ++j) {
                v(j - 1) -= 2.0 * std::cos(2.0 * static_cast<double>(k) * theta(j)) / denom;
            }
        }
    }

    for (int j = 1; j < N - 1; ++j) {
        w(j) = 2.0 * v(j - 1) / static_cast<double>(n);
    }

    return w;
}

/**
 * @brief Compute barycentric interpolation weights for the given nodes
 * @param nodes Interpolation nodes
 * @return Barycentric weight vector
 */
inline NumericVector barycentric_weights(const NumericVector &nodes) {
    const int N = static_cast<int>(nodes.size());
    if (N < 2) {
        throw InvalidArgument("barycentric_weights: need at least 2 nodes");
    }

    NumericVector lambda(N);
    for (int j = 0; j < N; ++j) {
        double prod = 1.0;
        for (int k = 0; k < N; ++k) {
            if (k == j) {
                continue;
            }
            prod *= (nodes(j) - nodes(k));
        }
        lambda(j) = 1.0 / prod;
    }
    return lambda;
}

/**
 * @brief Evaluate the j-th Lagrange basis polynomial at s using barycentric form
 * @param nodes Interpolation nodes
 * @param bary_w Barycentric weights
 * @param j Basis index
 * @param s Evaluation point
 * @return Value of L_j(s)
 */
inline double barycentric_basis_eval(const NumericVector &nodes, const NumericVector &bary_w, int j,
                                     double s) {
    const int N = static_cast<int>(nodes.size());
    if (N < 2) {
        throw InvalidArgument("barycentric_basis_eval: need at least 2 nodes");
    }
    if (bary_w.size() != N) {
        throw InvalidArgument("barycentric_basis_eval: barycentric weight size mismatch");
    }
    if (j < 0 || j >= N) {
        throw InvalidArgument("barycentric_basis_eval: basis index out of range");
    }

    const double tol = 128.0 * std::numeric_limits<double>::epsilon() * (1.0 + std::abs(s));
    for (int k = 0; k < N; ++k) {
        if (std::abs(s - nodes(k)) <= tol) {
            return (k == j) ? 1.0 : 0.0;
        }
    }

    double denom = 0.0;
    for (int k = 0; k < N; ++k) {
        denom += bary_w(k) / (s - nodes(k));
    }
    return (bary_w(j) / (s - nodes(j))) / denom;
}

/**
 * @brief Compute Birkhoff integration matrix B^a for the node set
 *
 * B^a_{ij} = integral_{nodes(0)}^{nodes(i)} ell_j(s) ds
 * where ell_j is the j-th Lagrange basis polynomial.
 *
 * @param nodes Interpolation nodes
 * @return Integration matrix (N x N)
 */
inline NumericMatrix birkhoff_integration_matrix(const NumericVector &nodes) {
    const int N = static_cast<int>(nodes.size());
    if (N < 2) {
        throw InvalidArgument("birkhoff_integration_matrix: need at least 2 nodes");
    }

    const NumericVector bary_w = barycentric_weights(nodes);
    const int n_quad = std::max(8, (N + 1) / 2 + 2);
    const auto [quad_x, quad_w] = detail::gauss_legendre_rule(n_quad);

    NumericMatrix B = NumericMatrix::Zero(N, N);
    const double a = nodes(0);

    for (int i = 0; i < N; ++i) {
        const double b = nodes(i);
        const double half = 0.5 * (b - a);
        const double mid = 0.5 * (b + a);
        if (std::abs(half) < detail::polynomial_root_tolerance) {
            continue; // first row stays zero
        }

        for (int q = 0; q < n_quad; ++q) {
            const double s = mid + half * quad_x(q);
            const double wq = half * quad_w(q);

            int node_match = -1;
            const double tol = 128.0 * std::numeric_limits<double>::epsilon() * (1.0 + std::abs(s));
            for (int k = 0; k < N; ++k) {
                if (std::abs(s - nodes(k)) <= tol) {
                    node_match = k;
                    break;
                }
            }

            if (node_match >= 0) {
                B(i, node_match) += wq;
                continue;
            }

            double denom = 0.0;
            for (int k = 0; k < N; ++k) {
                denom += bary_w(k) / (s - nodes(k));
            }
            for (int j = 0; j < N; ++j) {
                const double ell_j = (bary_w(j) / (s - nodes(j))) / denom;
                B(i, j) += wq * ell_j;
            }
        }
    }

    return B;
}

/**
 * @brief Spectral differentiation matrix using barycentric weights
 * @param nodes Interpolation nodes
 * @return Differentiation matrix (N x N)
 */
inline NumericMatrix spectral_diff_matrix(const NumericVector &nodes) {
    const int N = static_cast<int>(nodes.size());
    if (N < 2) {
        throw InvalidArgument("spectral_diff_matrix: need at least 2 nodes");
    }

    const NumericVector lambda = barycentric_weights(nodes);

    NumericMatrix D = NumericMatrix::Zero(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                continue;
            }
            D(i, j) = (lambda(j) / lambda(i)) / (nodes(i) - nodes(j));
        }
    }

    for (int i = 0; i < N; ++i) {
        D(i, i) = -D.row(i).sum();
    }

    return D;
}

} // namespace metis
