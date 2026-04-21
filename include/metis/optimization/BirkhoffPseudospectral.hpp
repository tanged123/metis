/**
 * @file BirkhoffPseudospectral.hpp
 * @brief Birkhoff pseudospectral transcription for trajectory optimization
 */

#pragma once

#include "TranscriptionBase.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/math/OrthogonalPolynomials.hpp"
#include <tuple>
#include <vector>

namespace metis {

/** @brief Available Birkhoff node distributions */
enum class BirkhoffScheme {
    LGL, ///< Legendre-Gauss-Lobatto nodes
    CGL  ///< Chebyshev-Gauss-Lobatto nodes
};

/** @brief Options for BirkhoffPseudospectral setup */
struct BirkhoffPseudospectralOptions {
    BirkhoffScheme scheme = BirkhoffScheme::LGL;
    int n_nodes = 21; ///< Number of collocation nodes (including endpoints)
};

/**
 * @brief Birkhoff pseudospectral transcription
 *
 * Uses integration-based (Birkhoff) formulation instead of differentiation.
 *
 * @see TranscriptionBase for shared interface
 * @see BirkhoffPseudospectralOptions for configuration
 * @see Pseudospectral for standard pseudospectral variant
 */
class BirkhoffPseudospectral : public TranscriptionBase<BirkhoffPseudospectral> {
    friend class TranscriptionBase<BirkhoffPseudospectral>;

  public:
    /**
     * @brief Construct with a reference to the optimization environment
     * @param opti Opti instance
     */
    explicit BirkhoffPseudospectral(Opti &opti) : TranscriptionBase<BirkhoffPseudospectral>(opti) {}

    /**
     * @brief Set up the Birkhoff problem with fixed final time
     * @param n_states number of state variables
     * @param n_controls number of control variables
     * @param t0 initial time
     * @param tf final time
     * @param opts Birkhoff options
     * @return tuple of (states, controls, time_grid)
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf,
          const BirkhoffPseudospectralOptions &opts = {}) {
        if (opts.n_nodes < 2) {
            throw InvalidArgument("BirkhoffPseudospectral: n_nodes must be >= 2");
        }
        if (n_states < 1) {
            throw InvalidArgument("BirkhoffPseudospectral: n_states must be >= 1");
        }
        if (n_controls < 0) {
            throw InvalidArgument("BirkhoffPseudospectral: n_controls must be >= 0");
        }

        n_states_ = n_states;
        n_controls_ = n_controls;
        n_nodes_ = opts.n_nodes;
        scheme_ = opts.scheme;
        t0_ = t0;
        tf_fixed_ = tf;
        tf_is_variable_ = false;

        NumericVector nodes;
        switch (scheme_) {
        case BirkhoffScheme::LGL:
            nodes = lgl_nodes(n_nodes_);
            break;
        case BirkhoffScheme::CGL:
            nodes = cgl_nodes(n_nodes_);
            break;
        default:
            throw RuntimeError("BirkhoffPseudospectral: unsupported BirkhoffScheme value");
        }

        tau_ = (nodes.array() + 1.0) * 0.5;
        B_ = birkhoff_integration_matrix(nodes);
        bk_weights_ = B_.row(n_nodes_ - 1).transpose();

        states_ = SymbolicMatrix(n_nodes_, n_states_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_states_; ++i) {
                states_(k, i) = opti_.variable(0.0);
            }
        }

        controls_ = SymbolicMatrix(n_nodes_, n_controls_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_controls_; ++i) {
                controls_(k, i) = opti_.variable(0.0);
            }
        }

        V_ = SymbolicMatrix(n_nodes_, n_states_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_states_; ++i) {
                V_(k, i) = opti_.variable(0.0);
            }
        }

        setup_complete_ = true;
        dynamics_set_ = false;
        dynamics_constraints_added_ = false;
        return {states_, controls_, tau_};
    }

    /**
     * @brief Set up the Birkhoff problem with variable final time
     * @param n_states number of state variables
     * @param n_controls number of control variables
     * @param t0 initial time
     * @param tf symbolic final time (decision variable)
     * @param opts Birkhoff options
     * @return tuple of (states, controls, time_grid)
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, const SymbolicScalar &tf,
          const BirkhoffPseudospectralOptions &opts = {}) {
        auto result = setup(n_states, n_controls, t0, 1.0, opts);
        tf_symbolic_ = tf;
        tf_is_variable_ = true;
        return result;
    }

    /** @brief Get the Birkhoff integration matrix
     *  @return n_nodes x n_nodes integration matrix */
    const NumericMatrix &integration_matrix() const { return B_; }
    /** @brief Get the Birkhoff quadrature weights
     *  @return vector of quadrature weights (last row of B) */
    const NumericVector &quadrature_weights() const { return bk_weights_; }
    /** @brief Get the virtual (derivative) decision variables
     *  @return matrix of size (n_nodes x n_states) */
    const SymbolicMatrix &virtual_vars() const { return V_; }

    /**
     * @brief Compute quadrature of an integrand over the time domain
     * @param integrand symbolic vector of values at each node
     * @return symbolic scalar approximation of the definite integral
     */
    SymbolicScalar quadrature(const SymbolicVector &integrand) const {
        if (!setup_complete_) {
            throw RuntimeError("BirkhoffPseudospectral: call setup() before quadrature()");
        }
        if (integrand.size() != n_nodes_) {
            throw InvalidArgument("BirkhoffPseudospectral: integrand size mismatch");
        }

        SymbolicScalar weighted_sum = SymbolicScalar(0.0);
        for (int k = 0; k < n_nodes_; ++k) {
            weighted_sum = weighted_sum + bk_weights_(k) * integrand(k);
        }
        return get_duration() / 2.0 * weighted_sum;
    }

  private:
    BirkhoffScheme scheme_ = BirkhoffScheme::LGL;
    NumericMatrix B_;
    NumericVector bk_weights_;
    SymbolicMatrix V_;

    void add_dynamics_constraints_impl() {
        if (!dynamics_set_) {
            throw RuntimeError(
                "BirkhoffPseudospectral: call set_dynamics() before add_dynamics_constraints()");
        }

        const SymbolicScalar half_dt = get_duration() / 2.0;

        std::vector<SymbolicVector> f(static_cast<std::size_t>(n_nodes_));
        for (int k = 0; k < n_nodes_; ++k) {
            f[static_cast<std::size_t>(k)] =
                dynamics_(get_state_at_node(k), get_control_at_node(k), get_time_at_node(k));
        }

        for (int s = 0; s < n_states_; ++s) {
            // Pointwise nonlinear dynamics: V_i = (dt/2) * f_i(X_i, U_i, t_i)
            for (int i = 0; i < n_nodes_; ++i) {
                opti_.subject_to(V_(i, s) == half_dt * f[static_cast<std::size_t>(i)](s));
            }

            // Linear state recovery for interior nodes.
            for (int i = 1; i < n_nodes_ - 1; ++i) {
                SymbolicScalar Bv_i = SymbolicScalar(0.0);
                for (int j = 0; j < n_nodes_; ++j) {
                    Bv_i = Bv_i + B_(i, j) * V_(j, s);
                }
                opti_.subject_to(states_(i, s) == states_(0, s) + Bv_i);
            }

            // Right boundary grid-equivalency.
            SymbolicScalar wv = SymbolicScalar(0.0);
            for (int j = 0; j < n_nodes_; ++j) {
                wv = wv + bk_weights_(j) * V_(j, s);
            }
            opti_.subject_to(states_(n_nodes_ - 1, s) == states_(0, s) + wv);
        }
    }
};

} // namespace metis
