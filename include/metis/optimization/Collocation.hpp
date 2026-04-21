/**
 * @file Collocation.hpp
 * @brief Direct Collocation for Trajectory Optimization
 */

#pragma once

#include "TranscriptionBase.hpp"
#include "metis/math/Spacing.hpp"
#include <tuple>

namespace metis {

/**
 * @brief Collocation scheme for dynamics discretization
 */
enum class CollocationScheme {
    Trapezoidal,   ///< 2nd order, implicit midpoint rule
    HermiteSimpson ///< 4th order, uses midpoint interpolation
};

/**
 * @brief Options for DirectCollocation setup
 *
 * @see DirectCollocation for usage
 */
struct CollocationOptions {
    CollocationScheme scheme = CollocationScheme::Trapezoidal;
    int n_nodes = 21; ///< Number of discretization nodes (including endpoints)
};

/**
 * @brief Direct collocation transcription
 *
 * @see TranscriptionBase for shared interface
 * @see CollocationOptions for configuration
 */
class DirectCollocation : public TranscriptionBase<DirectCollocation> {
    friend class TranscriptionBase<DirectCollocation>;

  public:
    /**
     * @brief Construct with a reference to the optimization environment
     * @param opti Opti instance
     */
    explicit DirectCollocation(Opti &opti) : TranscriptionBase<DirectCollocation>(opti) {}

    /**
     * @brief Set up the collocation problem with fixed final time
     * @param n_states number of state variables
     * @param n_controls number of control variables
     * @param t0 initial time
     * @param tf final time
     * @param opts collocation options
     * @return tuple of (states, controls, time_grid)
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf, const CollocationOptions &opts = {}) {
        if (opts.n_nodes < 2) {
            throw InvalidArgument("DirectCollocation: n_nodes must be >= 2");
        }
        if (n_states < 1) {
            throw InvalidArgument("DirectCollocation: n_states must be >= 1");
        }
        if (n_controls < 0) {
            throw InvalidArgument("DirectCollocation: n_controls must be >= 0");
        }

        n_states_ = n_states;
        n_controls_ = n_controls;
        n_nodes_ = opts.n_nodes;
        scheme_ = opts.scheme;
        t0_ = t0;
        tf_fixed_ = tf;
        tf_is_variable_ = false;

        tau_ = linspace(0.0, 1.0, n_nodes_);

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

        setup_complete_ = true;
        dynamics_set_ = false;
        dynamics_constraints_added_ = false;
        return {states_, controls_, tau_};
    }

    /**
     * @brief Set up the collocation problem with variable final time
     * @param n_states number of state variables
     * @param n_controls number of control variables
     * @param t0 initial time
     * @param tf symbolic final time (decision variable)
     * @param opts collocation options
     * @return tuple of (states, controls, time_grid)
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, const SymbolicScalar &tf,
          const CollocationOptions &opts = {}) {
        auto result = setup(n_states, n_controls, t0, 1.0, opts);
        tf_symbolic_ = tf;
        tf_is_variable_ = true;
        return result;
    }

    /**
     * @brief Composite quadrature for an integrand sampled at collocation nodes
     *
     * Trapezoidal scheme uses composite trapezoidal weights.
     * Hermite-Simpson uses composite Simpson when possible (even number of
     * intervals), otherwise falls back to trapezoidal.
     *
     * @param integrand symbolic vector of integrand values at each node
     * @return symbolic scalar approximation of the definite integral
     */
    SymbolicScalar quadrature(const SymbolicVector &integrand) const {
        if (!setup_complete_) {
            throw RuntimeError("DirectCollocation: call setup() before quadrature()");
        }
        if (integrand.size() != n_nodes_) {
            throw InvalidArgument("DirectCollocation: integrand size mismatch");
        }
        if (n_nodes_ < 2) {
            throw RuntimeError("DirectCollocation: need at least 2 nodes for quadrature");
        }

        const SymbolicScalar h = get_duration() / static_cast<double>(n_nodes_ - 1);
        SymbolicScalar sum = SymbolicScalar(0.0);

        if (scheme_ == CollocationScheme::HermiteSimpson && ((n_nodes_ - 1) % 2 == 0)) {
            sum = integrand(0) + integrand(n_nodes_ - 1);
            for (int k = 1; k < n_nodes_ - 1; ++k) {
                sum = sum + ((k % 2 == 0) ? 2.0 : 4.0) * integrand(k);
            }
            return h / 3.0 * sum;
        }

        sum = 0.5 * (integrand(0) + integrand(n_nodes_ - 1));
        for (int k = 1; k < n_nodes_ - 1; ++k) {
            sum = sum + integrand(k);
        }
        return h * sum;
    }

    /** @brief Get the number of collocation intervals
     *  @return n_nodes - 1 */
    int n_intervals() const { return n_nodes_ - 1; }

  private:
    CollocationScheme scheme_ = CollocationScheme::Trapezoidal;

    void add_dynamics_constraints_impl() {
        if (!dynamics_set_) {
            throw RuntimeError(
                "DirectCollocation: call set_dynamics() before add_dynamics_constraints()");
        }

        SymbolicScalar dt = get_duration() / static_cast<double>(n_nodes_ - 1);

        for (int k = 0; k < n_nodes_ - 1; ++k) {
            SymbolicVector x_k = get_state_at_node(k);
            SymbolicVector x_kp1 = get_state_at_node(k + 1);
            SymbolicVector u_k = get_control_at_node(k);
            SymbolicVector u_kp1 = get_control_at_node(k + 1);

            SymbolicScalar t_k = get_time_at_node(k);
            SymbolicScalar t_kp1 = get_time_at_node(k + 1);

            SymbolicVector f_k = dynamics_(x_k, u_k, t_k);
            SymbolicVector f_kp1 = dynamics_(x_kp1, u_kp1, t_kp1);

            switch (scheme_) {
            case CollocationScheme::Trapezoidal:
                add_trapezoidal_constraints(x_k, x_kp1, f_k, f_kp1, dt);
                break;
            case CollocationScheme::HermiteSimpson:
                add_hermite_simpson_constraints(x_k, x_kp1, u_k, u_kp1, f_k, f_kp1, t_k, t_kp1, dt);
                break;
            default:
                throw RuntimeError("DirectCollocation: unsupported CollocationScheme value");
            }
        }
    }

    void add_trapezoidal_constraints(const SymbolicVector &x_k, const SymbolicVector &x_kp1,
                                     const SymbolicVector &f_k, const SymbolicVector &f_kp1,
                                     const SymbolicScalar &dt) {
        for (int i = 0; i < n_states_; ++i) {
            SymbolicScalar defect = x_kp1(i) - x_k(i) - 0.5 * dt * (f_k(i) + f_kp1(i));
            opti_.subject_to(defect == 0.0);
        }
    }

    void add_hermite_simpson_constraints(const SymbolicVector &x_k, const SymbolicVector &x_kp1,
                                         const SymbolicVector &u_k, const SymbolicVector &u_kp1,
                                         const SymbolicVector &f_k, const SymbolicVector &f_kp1,
                                         const SymbolicScalar &t_k, const SymbolicScalar &t_kp1,
                                         const SymbolicScalar &dt) {
        SymbolicVector x_mid(n_states_);
        for (int i = 0; i < n_states_; ++i) {
            x_mid(i) = 0.5 * (x_k(i) + x_kp1(i)) + dt / 8.0 * (f_k(i) - f_kp1(i));
        }

        SymbolicVector u_mid(n_controls_);
        for (int i = 0; i < n_controls_; ++i) {
            u_mid(i) = 0.5 * (u_k(i) + u_kp1(i));
        }

        SymbolicScalar t_mid = 0.5 * (t_k + t_kp1);
        SymbolicVector f_mid = dynamics_(x_mid, u_mid, t_mid);

        for (int i = 0; i < n_states_; ++i) {
            SymbolicScalar defect =
                x_kp1(i) - x_k(i) - dt / 6.0 * (f_k(i) + 4.0 * f_mid(i) + f_kp1(i));
            opti_.subject_to(defect == 0.0);
        }
    }
};

} // namespace metis
