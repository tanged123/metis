/**
 * @file TranscriptionBase.hpp
 * @brief Shared CRTP base for trajectory transcription methods
 */

#pragma once

#include "Opti.hpp"
#include "metis/core/MetisError.hpp"
#include "metis/core/MetisTypes.hpp"
#include <functional>
#include <string>
#include <tuple>
#include <utility>

namespace metis {

/**
 * @brief Shared CRTP base for transcription methods
 *
 * @tparam Derived CRTP derived class (e.g. DirectCollocation, Pseudospectral)
 *
 * @see DirectCollocation for direct collocation transcription
 * @see Pseudospectral for pseudospectral transcription
 * @see MultipleShooting for multiple shooting transcription
 * @see BirkhoffPseudospectral for Birkhoff pseudospectral transcription
 */
template <typename Derived> class TranscriptionBase {
  public:
    /**
     * @brief Construct with a reference to the optimization environment
     * @param opti Opti instance to add variables and constraints to
     */
    explicit TranscriptionBase(Opti &opti) : opti_(opti) {}

    /**
     * @brief Pin all initial states to specified values
     * @param x0 state vector at the initial time
     */
    void set_initial_state(const NumericVector &x0) {
        if (!setup_complete_) {
            throw RuntimeError("TranscriptionBase: call setup() before set_initial_state()");
        }
        if (x0.size() != n_states_) {
            throw InvalidArgument("TranscriptionBase: x0 size mismatch");
        }
        for (int i = 0; i < n_states_; ++i) {
            opti_.subject_to(states_(0, i) == x0(i));
        }
    }

    /**
     * @brief Pin a single initial state component
     * @param idx state index
     * @param value fixed value for that state
     */
    void set_initial_state(int idx, double value) {
        if (!setup_complete_) {
            throw RuntimeError("TranscriptionBase: call setup() before set_initial_state()");
        }
        if (idx < 0 || idx >= n_states_) {
            throw InvalidArgument("TranscriptionBase: initial state index out of range");
        }
        opti_.subject_to(states_(0, idx) == value);
    }

    /**
     * @brief Pin all final states to specified values
     * @param xf state vector at the final time
     */
    void set_final_state(const NumericVector &xf) {
        if (!setup_complete_) {
            throw RuntimeError("TranscriptionBase: call setup() before set_final_state()");
        }
        if (xf.size() != n_states_) {
            throw InvalidArgument("TranscriptionBase: xf size mismatch");
        }
        for (int i = 0; i < n_states_; ++i) {
            opti_.subject_to(states_(n_nodes_ - 1, i) == xf(i));
        }
    }

    /**
     * @brief Pin a single final state component
     * @param idx state index
     * @param value fixed value for that state
     */
    void set_final_state(int idx, double value) {
        if (!setup_complete_) {
            throw RuntimeError("TranscriptionBase: call setup() before set_final_state()");
        }
        if (idx < 0 || idx >= n_states_) {
            throw InvalidArgument("TranscriptionBase: final state index out of range");
        }
        opti_.subject_to(states_(n_nodes_ - 1, idx) == value);
    }

    /**
     * @brief Register the dynamics function xdot = f(x, u, t)
     * @tparam Func callable with signature SymbolicVector(const SymbolicVector&, const
     * SymbolicVector&, const SymbolicScalar&)
     * @param dynamics dynamics function
     */
    template <typename Func> void set_dynamics(Func &&dynamics) {
        if (!setup_complete_) {
            throw RuntimeError("TranscriptionBase: call setup() before set_dynamics()");
        }
        if (dynamics_constraints_added_) {
            throw RuntimeError("TranscriptionBase: set_dynamics() cannot be called after "
                               "add_dynamics_constraints(); call setup() again to start fresh");
        }

        dynamics_ = [fn = std::forward<Func>(dynamics)](
                        const SymbolicVector &x, const SymbolicVector &u,
                        const SymbolicScalar &t) -> SymbolicVector { return fn(x, u, t); };
        dynamics_set_ = true;
    }

    /** @brief Get the state decision variable matrix
     *  @return matrix of size (n_nodes x n_states) */
    const SymbolicMatrix &states() const { return states_; }
    /** @brief Get the control decision variable matrix
     *  @return matrix of size (n_nodes x n_controls) or (n_intervals x n_controls) */
    const SymbolicMatrix &controls() const { return controls_; }
    /** @brief Get the normalized time grid
     *  @return vector of node times in [0, 1] */
    const NumericVector &time_grid() const { return tau_; }
    /** @brief Get the number of discretization nodes
     *  @return node count */
    int n_nodes() const { return n_nodes_; }
    /** @brief Get the number of state variables per node
     *  @return state dimension */
    int n_states() const { return n_states_; }
    /** @brief Get the number of control variables per node
     *  @return control dimension */
    int n_controls() const { return n_controls_; }

    /** @brief Add dynamics constraints to the optimization problem */
    void add_dynamics_constraints() {
        if (!setup_complete_) {
            throw RuntimeError("TranscriptionBase: call setup() before add_dynamics_constraints()");
        }
        static_cast<Derived *>(this)->add_dynamics_constraints_impl();
        dynamics_constraints_added_ = true;
    }

    /** @brief Alias for add_dynamics_constraints() */
    void add_defect_constraints() { add_dynamics_constraints(); }
    /** @brief Alias for add_dynamics_constraints() */
    void add_continuity_constraints() { add_dynamics_constraints(); }

  protected:
    Opti &opti_;
    int n_states_ = 0;
    int n_controls_ = 0;
    int n_nodes_ = 0;

    double t0_ = 0.0;
    double tf_fixed_ = 1.0;
    SymbolicScalar tf_symbolic_;
    bool tf_is_variable_ = false;

    bool setup_complete_ = false;
    bool dynamics_set_ = false;
    bool dynamics_constraints_added_ = false;

    NumericVector tau_;
    SymbolicMatrix states_;
    SymbolicMatrix controls_;

    std::function<SymbolicVector(const SymbolicVector &, const SymbolicVector &,
                                 const SymbolicScalar &)>
        dynamics_;

    SymbolicScalar get_duration() const {
        if (tf_is_variable_) {
            return tf_symbolic_ - t0_;
        }
        return SymbolicScalar(tf_fixed_ - t0_);
    }

    SymbolicScalar get_time_at_node(int k) const {
        if (k < 0 || k >= tau_.size()) {
            throw InvalidArgument("TranscriptionBase: node index out of range");
        }
        return t0_ + tau_(k) * get_duration();
    }

    SymbolicVector get_state_at_node(int k) const {
        if (k < 0 || k >= states_.rows()) {
            throw InvalidArgument("TranscriptionBase: state node index out of range");
        }
        SymbolicVector x(n_states_);
        for (int i = 0; i < n_states_; ++i) {
            x(i) = states_(k, i);
        }
        return x;
    }

    SymbolicVector get_control_at_node(int k) const {
        if (k < 0 || k >= controls_.rows()) {
            throw InvalidArgument("TranscriptionBase: control node index out of range");
        }
        SymbolicVector u(n_controls_);
        for (int i = 0; i < n_controls_; ++i) {
            u(i) = controls_(k, i);
        }
        return u;
    }
};

} // namespace metis
