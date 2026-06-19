# =============================================================================
# Intro Examples
# =============================================================================
add_executable(numeric_intro intro/numeric_intro.cpp)
target_link_libraries(numeric_intro PRIVATE metis)

add_executable(energy_intro intro/energy_intro.cpp)
target_link_libraries(energy_intro PRIVATE metis)

add_executable(print_example intro/print_example.cpp)
target_link_libraries(print_example PRIVATE metis)

add_executable(sparsity_intro intro/sparsity_intro.cpp)
target_link_libraries(sparsity_intro PRIVATE metis)

# =============================================================================
# Math Examples
# =============================================================================
add_executable(branching_logic math/branching_logic.cpp)
target_link_libraries(branching_logic PRIVATE metis)

add_executable(loop_patterns math/loop_patterns.cpp)
target_link_libraries(loop_patterns PRIVATE metis)

add_executable(graph_visualization math/graph_visualization.cpp)
target_link_libraries(graph_visualization PRIVATE metis)

add_executable(deep_graph_demo math/deep_graph_demo.cpp)
target_link_libraries(deep_graph_demo PRIVATE metis)

add_executable(sparse_derivative_pipeline math/sparse_derivative_pipeline.cpp)
target_link_libraries(sparse_derivative_pipeline PRIVATE metis)

add_executable(sensitivity_jacobian_demo math/sensitivity_jacobian_demo.cpp)
target_link_libraries(sensitivity_jacobian_demo PRIVATE metis)

add_executable(structural_transforms_demo math/structural_transforms_demo.cpp)
target_link_libraries(structural_transforms_demo PRIVATE metis)

add_executable(structural_diagnostics_demo math/structural_diagnostics_demo.cpp)
target_link_libraries(structural_diagnostics_demo PRIVATE metis)

add_executable(linear_solve_policy_demo math/linear_solve_policy_demo.cpp)
target_link_libraries(linear_solve_policy_demo PRIVATE metis)

add_executable(polynomial_chaos_demo math/polynomial_chaos_demo.cpp)
target_link_libraries(polynomial_chaos_demo PRIVATE metis)

# =============================================================================
# Interpolation Examples
# =============================================================================
add_executable(nd_interpolation_demo interpolation/nd_interpolation_demo.cpp)
target_link_libraries(nd_interpolation_demo PRIVATE metis)

add_executable(rootfinding_demo interpolation/rootfinding_demo.cpp)
target_link_libraries(rootfinding_demo PRIVATE metis)

add_executable(table_from_file_demo interpolation/table_from_file_demo.cpp)
target_link_libraries(table_from_file_demo PRIVATE metis)

add_executable(scattered_interpolation_demo interpolation/scattered_interpolation_demo.cpp)
target_link_libraries(scattered_interpolation_demo PRIVATE metis)

# =============================================================================
# Simulation Examples
# =============================================================================
add_executable(drag_coefficient simulation/drag_coefficient.cpp)
target_link_libraries(drag_coefficient PRIVATE metis)

add_executable(hybrid_sim simulation/hybrid_sim.cpp)
target_link_libraries(hybrid_sim PRIVATE metis)

add_executable(smooth_trajectory simulation/smooth_trajectory.cpp)
target_link_libraries(smooth_trajectory PRIVATE metis)

add_executable(aircraft_attitudes simulation/aircraft_attitudes.cpp)
target_link_libraries(aircraft_attitudes PRIVATE metis)

add_executable(brachistochrone simulation/brachistochrone.cpp)
target_link_libraries(brachistochrone PRIVATE metis)

add_executable(integration_demo integration_demo.cpp)
target_link_libraries(integration_demo PRIVATE metis)

# =============================================================================
# Optimization Examples (Phase 6)
# =============================================================================
add_executable(rosenbrock optimization/rosenbrock.cpp)
target_link_libraries(rosenbrock PRIVATE metis)

add_executable(brachistochrone_opti optimization/brachistochrone_opti.cpp)
target_link_libraries(brachistochrone_opti PRIVATE metis)

add_executable(beam_deflection optimization/beam_deflection.cpp)
target_link_libraries(beam_deflection PRIVATE metis)

add_executable(drag_optimization optimization/drag_optimization.cpp)
target_link_libraries(drag_optimization PRIVATE metis)

add_executable(drag_graph optimization/drag_graph.cpp)
target_link_libraries(drag_graph PRIVATE metis)

add_executable(parametric_sweep optimization/parametric_sweep.cpp)
target_link_libraries(parametric_sweep PRIVATE metis)

add_executable(transcription_comparison_demo optimization/transcription_comparison_demo.cpp)
target_link_libraries(transcription_comparison_demo PRIVATE metis)
