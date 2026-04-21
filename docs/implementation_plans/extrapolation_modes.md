# Extrapolation Modes for Interpolator

Add opt-in linear extrapolation with configurable safety bounds to the `Interpolator` class.

## Proposed Changes

### Core Types

#### [MODIFY] [Interpolate.hpp](file:///home/tanged/sources/metis/include/metis/math/Interpolate.hpp)

**1. Add ExtrapolationMode enum (after InterpolationMethod):**
```cpp
enum class ExtrapolationMode {
    Clamp,   ///< Clamp queries to grid bounds (default, current behavior)
    Linear   ///< Linear extrapolation from boundary slope
};
```

**2. Add ExtrapolationConfig struct:**
```cpp
struct ExtrapolationConfig {
    ExtrapolationMode mode = ExtrapolationMode::Clamp;
    
    // Safety bounds (applied to final output, not input)
    std::optional<double> lower_bound = std::nullopt;
    std::optional<double> upper_bound = std::nullopt;
    
    // Factory methods for convenience
    static ExtrapolationConfig clamp() { return {}; }
    
    static ExtrapolationConfig linear(
        std::optional<double> lower = std::nullopt,
        std::optional<double> upper = std::nullopt) {
        return {ExtrapolationMode::Linear, lower, upper};
    }
};
```

**3. Add member variables to Interpolator:**
```cpp
ExtrapolationConfig m_extrap_config;

// Precomputed boundary slopes for linear extrapolation (per dimension)
// For 1D: just left/right slopes
// For N-D: slopes at each boundary hyperplane
std::vector<double> m_boundary_slopes_left;   // d(value)/d(x) at x_min per dim
std::vector<double> m_boundary_slopes_right;  // d(value)/d(x) at x_max per dim
```

**4. Modify constructors to accept optional ExtrapolationConfig:**
```cpp
// Existing signature preserved (backwards compatible):
Interpolator(const std::vector<NumericVector>& points, 
             const NumericVector& values,
             InterpolationMethod method = InterpolationMethod::Linear);

// New overload with extrapolation config:
Interpolator(const std::vector<NumericVector>& points,
             const NumericVector& values,
             InterpolationMethod method,
             ExtrapolationConfig extrap);

// 1D convenience:
Interpolator(const NumericVector& x, const NumericVector& y,
             InterpolationMethod method,
             ExtrapolationConfig extrap);
```

**5. Add private helper for linear extrapolation:**
```cpp
// For 1D numeric:
double apply_linear_extrap_1d(double query, double clamped_result) const;

// For 1D symbolic:
SymbolicScalar apply_linear_extrap_1d_sym(
    const SymbolicScalar& query, 
    const SymbolicScalar& clamped_result) const;

// Apply output bounds (works for both):
template<typename Scalar>
Scalar apply_output_bounds(const Scalar& value) const;
```

---

### Linear Extrapolation Logic

For **1D**, linear extrapolation works as:
```cpp
if (query < x_min) {
    slope = (y[1] - y[0]) / (x[1] - x[0]);
    result = y[0] + slope * (query - x_min);
} else if (query > x_max) {
    slope = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]);
    result = y[n-1] + slope * (query - x_max);
} else {
    result = interpolate_internal(query);
}
result = apply_output_bounds(result);
```

For **N-D**, extrapolation is applied **per-dimension** before interpolation:
- If any coordinate is outside bounds, extrapolation is needed
- Can extrapolate along one dimension while interpolating others
- This maintains consistency with tensor-product interpolation

> [!IMPORTANT]
> N-D linear extrapolation is more complex. For Phase 1, consider **1D-only** linear extrapolation with N-D falling back to clamp.

---

### API Usage Example

```cpp
// Default (backwards compatible): clamp
metis::Interpolator interp1(x, y);

// Explicit clamp:
metis::Interpolator interp2(x, y, 
    metis::InterpolationMethod::Linear,
    metis::ExtrapolationConfig::clamp());

// Linear extrapolation with safety bounds:
metis::Interpolator interp3(x, y,
    metis::InterpolationMethod::BSpline,
    metis::ExtrapolationConfig::linear(-100.0, 1000.0));  // bounds

// Linear extrapolation, unbounded (use with caution):
metis::Interpolator interp4(x, y,
    metis::InterpolationMethod::Linear,
    metis::ExtrapolationConfig::linear());
```

---

## Verification Plan

### Automated Tests

Add tests to [test_interpolate.cpp](file:///home/tanged/sources/metis/tests/math/test_interpolate.cpp):

| Test Name | Description |
|-----------|-------------|
| `ExtrapLinear1D_LeftSide` | Query below x_min, verify linear slope |
| `ExtrapLinear1D_RightSide` | Query above x_max, verify linear slope |
| `ExtrapLinear1D_WithBounds` | Verify output clamping after extrapolation |
| `ExtrapLinear1D_Symbolic` | Verify symbolic linear extrapolation + AD |
| `ExtrapBackwardsCompat` | Default constructor still clamps |
| `ExtrapND_FallbackToClamp` | N-D with Linear mode falls back gracefully |

**Run tests:**
```bash
./scripts/ci.sh
# Or specifically:
./build/tests/test_math --gtest_filter="*Extrap*"
```

### Manual Verification

Update [table_from_file_demo.cpp](file:///home/tanged/sources/metis/examples/interpolation/table_from_file_demo.cpp) to demonstrate:
1. Creating interpolator with linear extrapolation
2. Querying outside bounds  
3. Showing bounded vs unbounded behavior

```bash
./build/examples/table_from_file_demo
```

Verify output shows:
- Extrapolated values outside grid bounds
- Values correctly clamped to safety bounds
- Gradients are non-zero outside bounds (unlike clamp mode)
