#include <iostream>
#include <metis/metis.hpp>

/**
 * @brief Symbolic graph generation for the README/slide drag example.
 *
 * The same generic `drag()` compiles to native arithmetic in numeric mode and
 * builds a CasADi graph in symbolic mode. The interesting bit is the supersonic
 * drag coefficient: `metis::where(...)` is a branchless *operation* (it becomes
 * a node in the graph), not a C++ `if`/`else` that would pick one path at trace
 * time. We then dump the traced graph in the clean "conceptual" theme.
 */

// Supersonic threshold + drag coefficient (free names used by drag()).
// These must be plain doubles so the numeric instantiation below stays f64.
constexpr double a_sonic = 343.0;  // speed of sound [m/s]
constexpr double cd_super = 0.045; // drag coefficient above Mach 1

// write the physics once, templated on Scalar
template <class Scalar> Scalar drag(Scalar rho, Scalar v, Scalar S, Scalar cd) {
    Scalar cd_eff = metis::where(v > a_sonic,   // an op,
                                 cd_super, cd); // not if/else
    return 0.5 * rho * v * v * S * cd_eff;
}

int main() {
    std::cout << "=== Metis: Symbolic Graph of the Drag Example ===\n\n";

    // ------------------------------------------------------------------
    // NUMERIC: f64, fully inlined. `where` collapses to a ternary.
    // ------------------------------------------------------------------
    double d_sub = drag(1.225, 250.0, 1.0, 0.02); // subsonic   -> uses cd = 0.02
    double d_sup = drag(1.225, 400.0, 1.0, 0.02); // supersonic -> where() overrides to cd_super
    std::cout << "Numeric subsonic   (v=250): D = " << d_sub << " N\n";
    std::cout << "Numeric supersonic (v=400): D = " << d_sup << " N\n\n";

    // ------------------------------------------------------------------
    // SYMBOLIC: the same call builds graph nodes. Named symbols make the
    // graph read like the slide (rho, v, S, cd) instead of i0, i1, ...
    // ------------------------------------------------------------------
    auto rho = metis::sym("rho");
    auto v = metis::sym("v");
    auto S = metis::sym("S");
    auto cd = metis::sym("cd");
    metis::SymbolicScalar g = drag(rho, v, S, cd);
    std::cout << "Symbolic: g = " << g << "\n\n";

    // Wrap with named inputs/outputs so the deep (fully expanded) graph labels
    // its leaves rho/v/S/cd and its terminal node "drag".
    casadi::Function fn("drag", {rho, v, S, cd}, {g}, {"rho", "v", "S", "cd"}, {"drag"});

    // Clean left-to-right "blueprint" theme matching the conceptual slide.
    const auto theme = metis::GraphStyle::conceptual();
    metis::export_graph_deep(fn, "drag_graph_deep", metis::DeepGraphFormat::HTML, "drag", theme);
    metis::export_graph_deep(fn, "drag_graph_deep", metis::DeepGraphFormat::DOT, "drag", theme);
    metis::render_graph("drag_graph_deep.dot", "drag_graph_deep.pdf"); // needs graphviz `dot`
    metis::render_graph("drag_graph_deep.dot", "drag_graph_deep.png");
    std::cout << "Deep graph -> drag_graph_deep.{html,pdf,png} (conceptual theme)\n";

    // ------------------------------------------------------------------
    // The numeric twin: the SAME Function, emitted as native arithmetic.
    //   - portable C kernel (the code it compiles to)
    //   - SSA work-vector panel (the algorithm it runs), pairs with the graph
    // ------------------------------------------------------------------
    metis::export_function_code(fn, "drag_num"); // -> drag_num.c
    metis::export_eval_algorithm(fn, "drag_num_algo", metis::DeepGraphFormat::HTML, theme,
                                 "drag - numeric twin");
    metis::export_eval_algorithm(fn, "drag_num_algo", metis::DeepGraphFormat::DOT, theme,
                                 "drag - numeric twin");
    metis::render_graph("drag_num_algo.dot", "drag_num_algo.png");
    std::cout << "Numeric twin -> drag_num.c + drag_num_algo.{html,png}\n";
    std::cout << "View:  xdg-open drag_graph_deep.pdf  |  xdg-open drag_num_algo.png\n";

    return 0;
}
