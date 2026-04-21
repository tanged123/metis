/**
 * @file structural_diagnostics_demo.cpp
 * @brief Demonstrate structural observability and identifiability checks.
 *
 * This example shows three common preflight diagnostics:
 * 1. Detecting an unobservable state from a measurement model.
 * 2. Detecting a structurally coupled parameter group and an unused parameter.
 * 3. Running observability and identifiability checks together on one model.
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <string>
#include <vector>

using namespace metis;

namespace {

const char *property_name(StructuralProperty property) {
    switch (property) {
    case StructuralProperty::Observability:
        return "observability";
    case StructuralProperty::Identifiability:
        return "identifiability";
    }
    return "unknown";
}

void print_indices(const std::string &label, const std::vector<int> &indices) {
    std::cout << "  " << label << " = [";
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (i != 0u) {
            std::cout << ", ";
        }
        std::cout << indices[i];
    }
    std::cout << "]\n";
}

void print_report(const StructuralSensitivityReport &report) {
    std::cout << "  property        = " << property_name(report.property) << "\n";
    std::cout << "  input block      = " << report.input_name << "\n";
    std::cout << "  structural rank  = " << report.structural_rank << " / "
              << report.variable_dimension << "\n";
    std::cout << "  measurement rows = " << report.output_dimension << "\n";
    print_indices("deficient vars", report.deficient_local_indices);
    print_indices("zero sensitivity", report.zero_sensitivity_local_indices);

    if (!report.deficiency_groups.empty()) {
        std::cout << "  deficient groups:\n";
        for (std::size_t i = 0; i < report.deficiency_groups.size(); ++i) {
            const auto &group = report.deficiency_groups[i];
            std::cout << "    group[" << i << "] rank " << group.structural_rank << " for vars ";
            std::cout << "[";
            for (std::size_t j = 0; j < group.input_local_indices.size(); ++j) {
                if (j != 0u) {
                    std::cout << ", ";
                }
                std::cout << group.input_local_indices[j];
            }
            std::cout << "]";
            if (!group.output_rows.empty()) {
                std::cout << " using rows [";
                for (std::size_t j = 0; j < group.output_rows.size(); ++j) {
                    if (j != 0u) {
                        std::cout << ", ";
                    }
                    std::cout << group.output_rows[j];
                }
                std::cout << "]";
            }
            std::cout << "\n";
        }
    }

    if (!report.issues.empty()) {
        std::cout << "  suggestions:\n";
        for (const auto &issue : report.issues) {
            std::cout << "    - " << issue.message << "\n";
        }
    }
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Structural Diagnostics Demo ===\n\n";

    {
        std::cout << "Case 1: observability check finds an unmeasured state\n";

        auto x = sym("x", 3, 1);
        auto y = SymbolicScalar::vertcat({
            x(0) + x(1),
            x(1),
        });

        Function sensor_model("sensor_model", {x}, {y});
        auto report = analyze_structural_observability(sensor_model);
        print_report(report);
        std::cout << "\n";
    }

    {
        std::cout << "Case 2: identifiability check finds a coupled parameter block\n";

        auto x = sym("x");
        auto p = sym("p", 4, 1);
        auto y = SymbolicScalar::vertcat({
            p(0) + p(1) + x,
            p(1) + p(2),
        });

        Function calibration_model("calibration_model", {x, p}, {y});
        auto report = analyze_structural_identifiability(calibration_model, 1);
        print_report(report);
        std::cout << "\n";
    }

    {
        std::cout << "Case 3: combined diagnostics on one measurement model\n";

        auto x = sym("x", 2, 1);
        auto p = sym("p", 2, 1);
        auto y = SymbolicScalar::vertcat({
            x(0),
            x(1) + p(0),
        });

        Function measurement_model("measurement_model", {x, p}, {y});

        StructuralDiagnosticsOptions opts;
        opts.state_input_idx = 0;
        opts.parameter_input_idx = 1;
        auto report = analyze_structural_diagnostics(measurement_model, opts);

        std::cout << "  observability:\n";
        print_report(*report.observability);
        std::cout << "  identifiability:\n";
        print_report(*report.identifiability);
        std::cout << "  any deficiency  = " << (report.has_deficiency() ? "yes" : "no") << "\n\n";
    }

    std::cout << "Takeaway:\n";
    std::cout
        << "  - structural observability checks measurement coverage over state coordinates\n";
    std::cout << "  - structural identifiability checks whether parameters can be separated by the "
                 "chosen outputs\n";
    std::cout << "  - the report distinguishes zero-sensitivity variables from coupled deficient "
                 "groups\n";

    return 0;
}
