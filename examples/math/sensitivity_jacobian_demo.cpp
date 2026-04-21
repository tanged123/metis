/**
 * @file sensitivity_jacobian_demo.cpp
 * @brief Demonstrate automatic sensitivity regime selection and Jacobian wrappers.
 *
 * This example shows three workflows:
 * 1. A forward-friendly block with few parameters and many outputs.
 * 2. An adjoint-friendly block with many parameters and a scalar output.
 * 3. A long-horizon checkpointed-adjoint recommendation for integrator-backed workloads.
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <string>
#include <vector>

using namespace metis;

namespace {

const char *regime_name(SensitivityRegime regime) {
    switch (regime) {
    case SensitivityRegime::Forward:
        return "forward";
    case SensitivityRegime::Adjoint:
        return "adjoint";
    case SensitivityRegime::CheckpointedAdjoint:
        return "checkpointed adjoint";
    }
    return "unknown";
}

const char *checkpoint_name(CheckpointInterpolation interpolation) {
    switch (interpolation) {
    case CheckpointInterpolation::None:
        return "none";
    case CheckpointInterpolation::Hermite:
        return "hermite";
    case CheckpointInterpolation::Polynomial:
        return "polynomial";
    }
    return "unknown";
}

double max_abs_error(const NumericMatrix &a, const NumericMatrix &b) {
    return (a - b).array().abs().maxCoeff();
}

void print_integrator_options(const SensitivityRecommendation &rec) {
    casadi::Dict opts = rec.integrator_options();

    std::cout << "  integrator options:\n";
    if (opts.count("nfwd") != 0u) {
        std::cout << "    nfwd                = " << int(opts.at("nfwd")) << "\n";
    }
    if (opts.count("nadj") != 0u) {
        std::cout << "    nadj                = " << int(opts.at("nadj")) << "\n";
    }
    if (opts.count("fsens_err_con") != 0u) {
        std::cout << "    fsens_err_con       = "
                  << (bool(opts.at("fsens_err_con")) ? "true" : "false") << "\n";
    }
    if (opts.count("steps_per_checkpoint") != 0u) {
        std::cout << "    steps_per_checkpoint = " << int(opts.at("steps_per_checkpoint")) << "\n";
    }
    if (opts.count("interpolation_type") != 0u) {
        std::cout << "    interpolation_type  = " << std::string(opts.at("interpolation_type"))
                  << "\n";
    }
}

void print_recommendation(const std::string &label, const SensitivityRecommendation &rec) {
    std::cout << label << "\n";
    std::cout << "  regime              = " << regime_name(rec.regime) << "\n";
    std::cout << "  parameter count     = " << rec.parameter_count << "\n";
    std::cout << "  output count        = " << rec.output_count << "\n";
    std::cout << "  direction count     = " << rec.casadi_direction_count() << "\n";
    std::cout << "  horizon hint        = " << rec.horizon_length << "\n";
    std::cout << "  stiff hint          = " << (rec.stiff ? "true" : "false") << "\n";
    std::cout << "  checkpoint interp   = " << checkpoint_name(rec.checkpoint_interpolation)
              << "\n";
    if (rec.uses_checkpointing()) {
        std::cout << "  steps/checkpoint    = " << rec.steps_per_checkpoint << "\n";
    }
    print_integrator_options(rec);
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << "=== Sensitivity Jacobian Workflow Demo ===\n\n";

    auto state = sym("state", 2);
    auto coeffs = sym("coeffs", 12);

    casadi::MX s0 = state(0);
    casadi::MX s1 = state(1);

    std::vector<casadi::MX> measurement_terms;
    measurement_terms.reserve(4);
    measurement_terms.push_back(s0 + 2.0 * s1 + 0.1 * coeffs(0));
    measurement_terms.push_back(s0 * s1);
    measurement_terms.push_back(metis::sin(s0) + coeffs(1));
    measurement_terms.push_back(s0 - s1 + 0.5 * coeffs(2));
    casadi::MX measurements = casadi::MX::vertcat(measurement_terms);

    casadi::MX objective = 0;
    for (int i = 0; i < 12; ++i) {
        casadi::MX ci = coeffs(i);
        objective += (1.0 + 0.1 * static_cast<double>(i)) * ci * ci;
    }
    objective += 0.5 * s0 * coeffs(0) - 0.25 * s1 * coeffs(3);

    Function model("sensitivity_workflow_model", {state, coeffs}, {measurements, objective});

    auto explicit_measurement_jac = metis::jacobian(measurements, state);
    auto explicit_objective_jac = metis::jacobian(objective, coeffs);
    Function reference("sensitivity_workflow_reference", {state, coeffs},
                       {explicit_measurement_jac, explicit_objective_jac});

    NumericVector state_val(2);
    state_val << 0.4, -1.2;

    NumericVector coeff_val(12);
    for (int i = 0; i < coeff_val.size(); ++i) {
        coeff_val(i) = 0.05 * static_cast<double>(i + 1);
    }

    std::cout << "Model blocks:\n";
    std::cout << "  output[0] = 4 measurement residuals\n";
    std::cout << "  output[1] = scalar objective\n";
    std::cout << "  input[0]  = 2-state vector\n";
    std::cout << "  input[1]  = 12 design coefficients\n\n";

    auto forward_rec = select_sensitivity_regime(model, 0, 0);
    auto forward_jac_fun = sensitivity_jacobian(model, 0, 0);
    NumericMatrix forward_jac = forward_jac_fun.eval(state_val, coeff_val);
    NumericMatrix forward_ref = reference(state_val, coeff_val)[0];

    print_recommendation("Case 1: measurement block wrt state", forward_rec);
    std::cout << "  Jacobian from sensitivity_jacobian:\n" << forward_jac << "\n";
    std::cout << "  max abs error vs explicit jacobian = "
              << max_abs_error(forward_jac, forward_ref) << "\n\n";

    auto adjoint_rec = select_sensitivity_regime(model, 1, 1);
    auto adjoint_jac_fun = sensitivity_jacobian(model, 1, 1);
    NumericMatrix adjoint_jac = adjoint_jac_fun.eval(state_val, coeff_val);
    NumericMatrix adjoint_ref = reference(state_val, coeff_val)[1];

    print_recommendation("Case 2: scalar objective wrt coefficients", adjoint_rec);
    std::cout << "  Jacobian from sensitivity_jacobian:\n" << adjoint_jac << "\n";
    std::cout << "  max abs error vs explicit jacobian = "
              << max_abs_error(adjoint_jac, adjoint_ref) << "\n\n";

    auto long_horizon_rec = select_sensitivity_regime(model, 1, 1, 1200, true);
    auto long_horizon_jac_fun = sensitivity_jacobian(model, 1, 1, 1200, true);
    NumericMatrix long_horizon_jac = long_horizon_jac_fun.eval(state_val, coeff_val);

    print_recommendation("Case 3: long-horizon stiff objective wrt coefficients", long_horizon_rec);
    std::cout << "  Jacobian from sensitivity_jacobian:\n" << long_horizon_jac << "\n";
    std::cout << "  max abs error vs explicit jacobian = "
              << max_abs_error(long_horizon_jac, adjoint_ref) << "\n\n";

    std::cout << "Takeaway:\n";
    std::cout << "  - select_sensitivity_regime(...) chooses the derivative regime metadata\n";
    std::cout << "  - sensitivity_jacobian(...) builds the Jacobian function without manual\n";
    std::cout << "    forward/reverse seed plumbing\n";
    std::cout << "  - for plain Function Jacobians, checkpointed adjoint still uses reverse\n";
    std::cout << "    construction today, while integrator_options() carries the extra\n";
    std::cout << "    checkpoint settings for downstream integrator-backed workflows\n";

    return 0;
}
