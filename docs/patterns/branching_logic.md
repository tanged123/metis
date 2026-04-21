# Branching Logic in Metis

## The Problem

Standard C++ control flow (`if/else`, `switch`) doesn't work with symbolic types:

```cpp
// ❌ DOESN'T WORK - condition can't be evaluated at graph-building time
template <typename Scalar>
Scalar bad_example(const Scalar& x) {
    if (x < 0) {  // ERROR with symbolic types!
        return -x;
    }
    return x;
}
```

## The Solution: `metis::where()`

Use `metis::where(condition, true_val, false_val)` for symbolic-compatible branching:

```cpp
// ✅ WORKS - creates computational graph
template <typename Scalar>
Scalar good_example(const Scalar& x) {
    return metis::where(x < 0.0, -x, x);
}
```

## Pattern Guide

### 1. Simple If-Else
```cpp
// C++: if (x < 0) return -x; else return x;
return metis::where(x < 0.0, -x, x);
```

### 2. If-Else-If-Else (Using select - RECOMMENDED)
```cpp
// C++: if (x < -1) return -1; else if (x > 1) return 1; else return x;

// ✅ Clean with select()
return metis::select({x < -1.0, x > 1.0},
                    {Scalar(-1.0), Scalar(1.0)},
                    x);  // default

// ❌ Hard to read with nested where()
return metis::where(x < -1.0, 
                   Scalar(-1.0),
                   metis::where(x > 1.0, Scalar(1.0), x));
```

### 3. Switch-Case Logic (select shines here!)
```cpp
// C++: switch (regime) { case 0: ... case 1: ... }

return metis::select({mach < 0.3, mach < 0.8, mach < 1.2},
                    {Scalar(0.02), Scalar(0.025), Scalar(0.05)},
                    Scalar(0.03));  // default
```

### 4. Complex Multi-Step Logic
When branches need multiple calculation steps, use helper functions:

```cpp
// Helper functions for complex calculations
template <typename Scalar>
Scalar turbulent_flow(const Scalar& re, const Scalar& v) {
    auto cf = 0.074 / metis::pow(re, 0.2);
    auto correction = 1.0 + 0.144 * metis::pow(v / 343.0, 2.0);
    return cf * correction;
}

template <typename Scalar>
Scalar laminar_flow(const Scalar& re) {
    return 1.328 / metis::sqrt(re);
}

// Use in branching logic
template <typename Scalar>
Scalar skin_friction(const Scalar& re, const Scalar& v) {
    return metis::where(re > 5e5,
                       turbulent_flow(re, v),  // Multi-step calculation
                       laminar_flow(re));       // Simpler calculation
}
```

**Important**: Both branches are **always evaluated** in symbolic mode (that's how computational graphs work). The condition selects which result to use.

### 5. Multi-Variable Conditions
```cpp
// Combine multiple conditions
Scalar is_stalled = alpha_abs > alpha_stall;
Scalar low_reynolds = reynolds < 1e5;

return metis::where(is_stalled && low_reynolds,
                   correction_value,
                   nominal_value);
```

## API Reference

### `metis::where(condition, true_val, false_val)`
Basic ternary selection. Use for simple if-else.

### `metis::select(conditions, values, default_value)`
Multi-way selection. Cleaner than nested `where()` for switch-like logic.

- **conditions**: Vector/list of conditions to check (in order)
- **values**: Vector/list of values to return (same size as conditions)
- **default_value**: Value if no condition matches

Returns the value corresponding to the **first true condition**, or default.

## Key Takeaways

1. **Always use `metis::where()`** for conditions involving template scalar types
2. **Nest `where()` calls** for multi-way branching (if-else-if chains)
3. **Compute intermediate flags** to make complex logic readable
4. **Works in both modes**: numeric (evaluates immediately) and symbolic (builds graph)
5. **Enables autodiff**: derivatives flow through all branches correctly

## See Also

- `examples/branching_logic.cpp` - Comprehensive examples
- `include/metis/math/Logic.hpp` - Implementation details
- `docs/user_guides/symbolic_computing.md` - Symbolic mode guide
