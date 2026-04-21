#pragma once
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <iostream>
#include <metis/core/MetisIO.hpp>
#include <metis/math/Linalg.hpp>
#include <vector>

// Helper to evaluate 0-argument CasADi MX to double
inline double eval_scalar(const casadi::MX &x) { return metis::eval(x); }

// Helper to evaluate CasADi MX to Eigen Matrix
inline Eigen::MatrixXd eval_matrix(const casadi::MX &x) {
    if (x.is_scalar()) {
        double val = metis::eval(x);
        Eigen::MatrixXd mat(1, 1);
        mat(0, 0) = val;
        return mat;
    }
    // For non-scalar MX matrices
    try {
        casadi::Function f("f", std::vector<casadi::MX>{}, std::vector<casadi::MX>{x});
        auto res = f(std::vector<casadi::DM>{});
        casadi::DM res_dm = res[0];

        Eigen::MatrixXd mat(res_dm.size1(), res_dm.size2());
        for (int i = 0; i < res_dm.size1(); ++i) {
            for (int j = 0; j < res_dm.size2(); ++j) {
                mat(i, j) = static_cast<double>(res_dm(i, j));
            }
        }
        return mat;
    } catch (const std::exception &e) {
        std::cerr << "CasADi eval failed: " << e.what() << std::endl;
        throw;
    }
}

// Overload for Eigen matrix of MX
template <typename Derived>
inline Eigen::MatrixXd eval_matrix(const Eigen::MatrixBase<Derived> &x) {
    return metis::eval(x);
}

// Overload for scalar double (passthrough)
inline double eval_scalar(const double &x) { return x; }
