/**
 * @file structural_transforms_demo.cpp
 * @brief Demonstrate alias elimination, BLT decomposition, and tearing recommendations.
 *
 * This example walks through three structural-analysis workflows:
 * 1. Alias elimination with a nonlinear residual left in reduced coordinates.
 * 2. BLT decomposition on a system with two scalar blocks and one coupled block.
 * 3. Tearing recommendations on a simple algebraic cycle.
 */

#include <iomanip>
#include <iostream>
#include <metis/metis.hpp>
#include <string>
#include <vector>

using namespace metis;

namespace {

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

void print_blocks(const BLTDecomposition &blt) {
    std::cout << "  block count = " << blt.blocks.size() << "\n";
    for (std::size_t block_idx = 0; block_idx < blt.blocks.size(); ++block_idx) {
        const auto &block = blt.blocks[block_idx];
        std::cout << "  block[" << block_idx << "]\n";
        print_indices("residuals", block.residual_indices);
        print_indices("variables", block.variable_indices);
        print_indices("tear vars", block.tear_variable_indices);
    }
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Structural Transforms Demo ===\n\n";

    {
        std::cout << "Case 1: alias elimination leaves a reduced nonlinear solve\n";

        auto x = sym("x", 3, 1);
        auto p = sym("p");
        SymbolicScalar x0 = x(0);
        SymbolicScalar x1 = x(1);
        SymbolicScalar x2 = x(2);

        auto residual = SymbolicScalar::vertcat({
            x0 - x1,
            x2 - p,
            metis::sin(x0) + x2 - 2.0,
        });

        Function fn("alias_case", {x, p}, {residual});
        auto analysis = structural_analyze(fn);

        std::cout << "  original residuals:\n";
        std::cout << "    r0 = x0 - x1\n";
        std::cout << "    r1 = x2 - p\n";
        std::cout << "    r2 = sin(x0) + x2 - 2\n";
        print_indices("kept vars", analysis.alias_elimination.kept_variable_indices);
        print_indices("eliminated vars", analysis.alias_elimination.eliminated_variable_indices);
        print_indices("kept residuals", analysis.alias_elimination.kept_residual_indices);
        print_indices("eliminated residuals",
                      analysis.alias_elimination.eliminated_residual_indices);
        std::cout << "  substitutions:\n";
        for (const auto &sub : analysis.alias_elimination.substitutions) {
            std::cout << "    x[" << sub.eliminated_variable_index << "] <- " << sub.replacement
                      << "  (from residual " << sub.residual_index << ")\n";
        }

        NumericMatrix x_reduced(1, 1);
        x_reduced(0, 0) = 1.25;
        NumericMatrix reconstructed =
            analysis.alias_elimination.reconstruct_full_input.eval(x_reduced, 0.5);
        NumericMatrix reduced_residual =
            analysis.alias_elimination.reduced_function.eval(x_reduced, 0.5);

        std::cout << "  reconstructed full state at x_reduced = 1.25, p = 0.5:\n";
        std::cout << reconstructed.transpose() << "\n";
        std::cout << "  reduced residual value:\n";
        std::cout << reduced_residual << "\n";
        std::cout << "  reduced BLT summary:\n";
        print_blocks(analysis.blt);
        std::cout << "\n";
    }

    {
        std::cout << "Case 2: BLT separates independent and coupled subsystems\n";

        auto x = sym("x", 4, 1);
        auto residual = SymbolicScalar::vertcat({
            x(0) - 1.0,
            x(1) + x(2),
            x(1) - x(2),
            x(3) - 3.0,
        });

        Function fn("blt_case", {x}, {residual});
        auto blt = block_triangularize(fn);

        std::cout << "  original residuals:\n";
        std::cout << "    r0 = x0 - 1\n";
        std::cout << "    r1 = x1 + x2\n";
        std::cout << "    r2 = x1 - x2\n";
        std::cout << "    r3 = x3 - 3\n";
        print_blocks(blt);
        std::cout << "\n";
    }

    {
        std::cout << "Case 3: tearing recommends an iteration variable for a cycle\n";

        auto x = sym("x", 3, 1);
        auto residual = SymbolicScalar::vertcat({
            x(0) + x(1),
            x(1) + x(2),
            x(2) + x(0),
        });

        Function fn("tearing_case", {x}, {residual});
        auto blt = block_triangularize(fn);

        std::cout << "  original residuals:\n";
        std::cout << "    r0 = x0 + x1\n";
        std::cout << "    r1 = x1 + x2\n";
        std::cout << "    r2 = x2 + x0\n";
        print_blocks(blt);
        std::cout << "\n";
    }

    std::cout << "Takeaway:\n";
    std::cout << "  - alias elimination shrinks directly solvable affine rows first\n";
    std::cout << "  - BLT exposes independent scalar blocks versus coupled blocks\n";
    std::cout << "  - tearing returns a starting-point iteration-variable recommendation\n";

    return 0;
}
