/**
 * @file MultiShooting.hpp
 * @brief Multiple Shooting for Trajectory Optimization
 */

#pragma once

#include "TranscriptionBase.hpp"
#include "metis/math/Spacing.hpp"
#include <string>
#include <tuple>

namespace metis {

/**
 * @brief Options for MultipleShooting
 *
 * @see MultipleShooting for usage
 */
struct MultiShootingOptions {
    int n_intervals = 20;              ///< Number of shooting intervals
    std::string integrator = "cvodes"; ///< Integrator plugin ("cvodes", "rk", "idas")
    double tol = 1e-8;                 ///< Integrator required tolerance
    bool normalize_time = true; ///< If true, integrates on normalized time and scales ODE by dt
};

/**
 * @brief Multiple shooting transcription
 *
 * @see TranscriptionBase for shared interface
 * @see MultiShootingOptions for configuration
 */
class MultipleShooting : public TranscriptionBase<MultipleShooting> {
    friend class TranscriptionBase<MultipleShooting>;

  public:
    /**
     * @brief Construct with a reference to the optimization environment
     * @param opti Opti instance
     */
    explicit MultipleShooting(Opti &opti) : TranscriptionBase<MultipleShooting>(opti) {}

    /**
     * @brief Set up the shooting problem with fixed final time
     * @param n_states number of state variables
     * @param n_controls number of control variables
     * @param t0 initial time
     * @param tf final time
     * @param opts shooting options
     * @return tuple of (states, controls, time_grid)
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, double tf,
          const MultiShootingOptions &opts = {}) {
        return setup_impl(n_states, n_controls, t0, tf, false, opts);
    }

    /**
     * @brief Set up the shooting problem with variable final time
     * @param n_states number of state variables
     * @param n_controls number of control variables
     * @param t0 initial time
     * @param tf symbolic final time (decision variable)
     * @param opts shooting options
     * @return tuple of (states, controls, time_grid)
     */
    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup(int n_states, int n_controls, double t0, const SymbolicScalar &tf,
          const MultiShootingOptions &opts = {}) {
        tf_symbolic_ = tf;
        return setup_impl(n_states, n_controls, t0, 1.0, true, opts);
    }

    /** @brief Get the number of shooting intervals
     *  @return interval count */
    int n_intervals() const { return n_intervals_; }

  private:
    MultiShootingOptions opts_;
    casadi::Function integrator_;
    int n_intervals_ = 0;

    std::tuple<SymbolicMatrix, SymbolicMatrix, NumericVector>
    setup_impl(int n_states, int n_controls, double t0, double tf_val, bool variable_tf,
               const MultiShootingOptions &opts) {
        if (opts.n_intervals < 1) {
            throw InvalidArgument("MultipleShooting: n_intervals must be >= 1");
        }
        if (n_states < 1) {
            throw InvalidArgument("MultipleShooting: n_states must be >= 1");
        }
        if (n_controls < 0) {
            throw InvalidArgument("MultipleShooting: n_controls must be >= 0");
        }

        n_states_ = n_states;
        n_controls_ = n_controls;
        opts_ = opts;
        n_intervals_ = opts.n_intervals;
        n_nodes_ = n_intervals_ + 1;
        t0_ = t0;

        if (variable_tf) {
            tf_is_variable_ = true;
        } else {
            tf_fixed_ = tf_val;
            tf_is_variable_ = false;
        }

        states_ = SymbolicMatrix(n_nodes_, n_states_);
        for (int k = 0; k < n_nodes_; ++k) {
            for (int i = 0; i < n_states_; ++i) {
                states_(k, i) = opti_.variable(0.0);
            }
        }

        controls_ = SymbolicMatrix(n_intervals_, n_controls_);
        for (int k = 0; k < n_intervals_; ++k) {
            for (int i = 0; i < n_controls_; ++i) {
                controls_(k, i) = opti_.variable(0.0);
            }
        }

        tau_ = linspace(0.0, 1.0, n_nodes_);
        setup_complete_ = true;
        dynamics_set_ = false;
        dynamics_constraints_added_ = false;
        integrator_ = casadi::Function();

        return {states_, controls_, tau_};
    }

    void ensure_integrator() {
        if (!integrator_.is_null()) {
            return;
        }

        SymbolicScalar x_sym = sym("x", n_states_);
        SymbolicScalar p_sym = sym("p", n_controls_ + 1); // p = [u; dt]
        SymbolicScalar t = sym("t");

        SymbolicScalar u_sym;
        if (n_controls_ > 0) {
            u_sym = p_sym(casadi::Slice(0, n_controls_));
        } else {
            u_sym = casadi::MX(0, 1);
        }
        SymbolicScalar dt_sym = p_sym(n_controls_);

        SymbolicVector x_vec = as_vector(x_sym);
        SymbolicVector u_vec = as_vector(u_sym);
        SymbolicVector dxdt = dynamics_(x_vec, u_vec, t);
        SymbolicVector ode_scaled = dxdt * dt_sym;

        casadi::MXDict dae = {{"x", x_sym}, {"p", p_sym}, {"ode", to_mx(ode_scaled)}};

        casadi::Dict intg_opts;
        if (opts_.integrator == "cvodes") {
            intg_opts["abstol"] = opts_.tol;
            intg_opts["reltol"] = opts_.tol;
        }

        integrator_ = casadi::integrator("shooting_intg", opts_.integrator, dae, intg_opts);
    }

    void add_dynamics_constraints_impl() {
        if (!dynamics_set_) {
            throw RuntimeError(
                "MultipleShooting: call set_dynamics() before add_dynamics_constraints()");
        }
        ensure_integrator();

        const SymbolicScalar dt = get_duration() / static_cast<double>(n_intervals_);

        for (int k = 0; k < n_intervals_; ++k) {
            SymbolicVector x_k = get_state_at_node(k);
            SymbolicVector u_k = get_control_at_node(k);
            SymbolicVector x_kp1 = get_state_at_node(k + 1);

            SymbolicScalar p = SymbolicScalar::vertcat({to_mx(u_k), dt});
            casadi::MXDict args = {{"x0", to_mx(x_k)}, {"p", p}};
            casadi::MXDict res = integrator_(args);
            SymbolicScalar x_integrated = res.at("xf");

            opti_.subject_to(to_mx(x_kp1) == x_integrated);
        }
    }
};

} // namespace metis
