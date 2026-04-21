# Loop Patterns in Metis

## The Core Distinction: Structural vs Dynamic

### Structural Loops (✅ Work in Symbolic Mode)
Loop bounds are **known at graph-building time** (int, const):
```cpp
for (int i = 0; i < 10; i++) {
    // Loop index 'i' is structural
    // Values computed can be symbolic
    result += symbolic_value * i;
}
```

### Dynamic Loops (⚠️ Numeric Mode ONLY)
Loop continuation depends on **runtime values**:
```cpp
while (error > tolerance) {  // ❌ Can't evaluate symbolic condition!
    error = update(error);
}
```

## Common Patterns

### 1. Simple For Loop
```cpp
template <typename Scalar>
Scalar sum_of_squares(int n) {
    Scalar sum = 0.0;
    for (int i = 1; i <= n; ++i) {  // n is structural
        Scalar value = static_cast<Scalar>(i);
        sum += value * value;  // values can be symbolic
    }
    return sum;
}
```

### 2. Nested Loops (Matrix Operations)
```cpp
template <typename Scalar>
Matrix<Scalar> multiply(const Matrix<Scalar>& A, const Matrix<Scalar>& B) {
    int m = A.rows(), n = A.cols(), p = B.cols();
    Matrix<Scalar> C(m, p);
    
    for (int i = 0; i < m; ++i) {          // Structural bounds
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                C(i,j) += A(i,k) * B(k,j); // Symbolic values
            }
        }
    }
    return C;
}
```

### 3. While Loop → Fixed Iterations
```cpp
// ❌ DOESN'T WORK (Symbolic):
while (error > tol) { ... }

// ✅ WORKS (Both modes):
for (int i = 0; i < max_iterations; ++i) {
    // Fixed iterations instead of convergence check
    x = update(x);
}
```

### 4. Break/Continue → Conditional Accumulation
```cpp
// ❌ DOESN'T WORK (Symbolic):
for (int i = 0; i < n; ++i) {
    if (values(i) > threshold) break;  // Dynamic condition!
}

// ✅ WORKS (Both modes):
Scalar result = default_value;
for (int i = 0; i < n; ++i) {
    result = metis::where(values(i) > threshold,
                         values(i),      // Use this value
                         result);        // Keep previous
}
```

## When You Need Dynamic Loops

### Use Case: Time-stepping simulation until convergence

**Option 1: Numeric-Only Simulation** (RECOMMENDED for pure simulation)
```cpp
double simulate_until_steady_state(double x0) {
    // ✅ Use dynamic loops freely in numeric mode!
    double x = x0;
    double error = 1000.0;
    
    while (error > 1e-6) {  // Dynamic - totally fine!
        double x_new = physics_update(x);
        error = std::abs(x_new - x);
        x = x_new;
    }
    
    return x;
}
```

**Option 2: Fixed Iterations** (for symbolic compatibility)
```cpp
template <typename Scalar>
Scalar simulate_fixed_steps(const Scalar& x0, int n_steps) {
    // ✅ Works in both modes
    Scalar x = x0;
    
    for (int i = 0; i < n_steps; ++i) {
        x = physics_update(x);
    }
    
    return x;
}
```

**Option 3: Hybrid Approach** (best of both worlds)
```cpp
// 1. Run numeric simulation to determine iteration count
int n_steps = determine_steps_numerically();

// 2. Use that fixed count for symbolic version
template <typename Scalar>
Scalar simulate_hybrid(const Scalar& x0) {
    return simulate_fixed_steps(x0, n_steps);  // Can now differentiate!
}
```

## Advanced Patterns

### Conditional Logic Inside Loops
```cpp
template <typename Scalar>
Scalar selective_sum(const Vector<Scalar>& values,
                     const Vector<Scalar>& thresholds) {
    Scalar sum = 0.0;
    
    for (int i = 0; i < values.size(); ++i) {
        // Use where() for conditions on values
        sum += metis::where(values(i) > thresholds(i),
                           values(i),
                           Scalar(0.0));
    }
    
    return sum;
}
```

### Element-wise Transformations
```cpp
template <typename Scalar>
Vector<Scalar> apply_function(const Vector<Scalar>& input) {
    int n = input.size();
    Vector<Scalar> output(n);
    
    for (int i = 0; i < n; ++i) {
        output(i) = complex_function(input(i));
    }
    
    return output;
}
```

## Key Takeaways

| Feature | Numeric Mode | Symbolic Mode |
|---------|--------------|---------------|
| `for (int i=0; i<N; i++)` | ✅ Yes | ✅ Yes |
| `while (symbolic > val)` | ✅ Yes | ❌ No |
| `if (symbolic) break` | ✅ Yes | ❌ No |
| Loop values | Any | Symbolic OK |
| Loop bounds | Any | Must be structural |

**When to use which:**
- **Pure simulation/analysis**: Use dynamic loops freely!
- **Optimization/AutoDiff**: Use fixed iterations
- **Both**: Hybrid approach (numeric determines N, symbolic uses fixed N)

## See Also
- `examples/loop_patterns.cpp` - Comprehensive examples
- `examples/numeric_intro.cpp` - Time-stepping example
- `docs/patterns/branching_logic.md` - Conditional patterns
