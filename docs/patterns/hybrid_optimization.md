# Hybrid Optimization Patterns

How to handle simulations with **unknown end-times** or **event detection** in Metis (AutoDiff/Optimization).

## The Problem
Standard AutoDiff requires a **fixed computational graph**. 
- You cannot optimize a loop condition (e.g., `while (y > 0)`). 
- If the number of steps changes during optimization, the graph structure changes, breaking the gradient flow.

## Pattern 1: Hybrid Pre-Step (Mesh Refinement)
**Best for**: Long simulations where you need to find an approximate "number of steps" $N$.

### Concept
1. **Pre-Step (Numeric)**: Run a cheap simulation (no AutoDiff) to find $N$.
2. **Build Graph (Symbolic)**: Construct a graph with exactly $N$ steps.
3. **Optimize**: Compute gradients on this fixed graph. If the trajectory changes significantly, re-run the Pre-Step (outer loop).

```cpp
// 1. Numeric Pre-Step
int steps = find_steps_until_impact_numeric(params);

// 2. Symbolic Build
auto result = simulate_fixed_steps_symbolic(params, steps);

// 3. Optimize
auto grad = metis::jacobian({result}, {params});
```

## Pattern 2: Free-Time Formulation (Time Scaling)
**Best for**: Getting precise event timing (e.g., "Hit target exactly at $t_f$").

### Concept
Fix the number of steps $N$ (e.g., 100), and make **Time** or **Timestep** ($\Delta t$) an optimization variable. 

Problem: Find $T$ such that $y(T) = 0$.

```cpp
// Fix N=100
int N = 100;
auto T_sym = metis::sym("T");

// Timestep scales with T
// dt = T / N; 
auto y_final = simulate_scaled_time(y0, v0, T_sym, N);

// Optimize T to make y_final == 0
auto dy_dT = metis::jacobian({y_final}, {T_sym});
// Use Newton's method on T...
```

## Comparison

| Feature | Pre-Step (Hybrid) | Free-Time (Scaling) |
|---------|-------------------|---------------------|
| **Timestep** | Fixed ($\Delta t = 0.01$) | Variable ($\Delta t = T/N$) |
| **Steps** | Variable (determined numerically) | Fixed (constant) |
| **Precision** | Grid-limited ($\pm \Delta t$) | Exact (continuous) |
| **Use Case** | Rough trajectory optimization | Exact event detection |

## Example Code
See `examples/hybrid_sim.cpp` for a complete implementation of both patterns.
