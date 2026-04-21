# Implementation Plan: Improved Metis IO & Debugging

**Goal**: Enable elegant printing, getting, and setting of Metis types, specifically focusing on the Symbolic (CasADi) backend compatibility with Eigen's IO.

## The Problem
Eigen's default `operator<<` relies on `Eigen::NumTraits` and `std::numeric_limits` for formatting. `casadi::MX` doesn't fully satisfy these traits in a way compatible with Eigen's IO system, causing build errors when printing `MetisMatrix<casadi::MX>`.

Additionally, retrieving values from symbolic matrices requires verbose `to_mx` or manual loops.

## Proposed Solution

1.  **Specialized `operator<<` for Symbolic Matrices**:
    *   Overload `operator<<` for `Eigen::Matrix<casadi::MX, ...>` globally or within `metis` namespace.
    *   Format the output to look like a standard matrix, printing the symbolic expressions.

2.  **`NumTraits` Specialization for `casadi::MX`**:
    *   Ideally, we should specialize `Eigen::NumTraits<casadi::MX>` to be more robust. This might fix Eigen's native printing and other algorithms.

3.  **IO Utilities**:
    *   `disp(obj)`: A universal display function.
    *   `eval(obj)`: A helper to quickly evaluate constant calculations (without variables) or full expressions.

4.  **Value Access**:
    *   Ensure `operator()` works seamlessly (it does via Eigen).

## Detailed Changes

### 1. `include/metis/core/MetisIO.hpp` (New)
*   Define `Eigen::NumTraits<casadi::MX>` specialization (if missing or incomplete in CasADi integration).
*   Overload `std::ostream& operator<<` for `MetisMatrix<casadi::MX>`.

### 2. `include/metis/metis.hpp`
*   Include `MetisIO.hpp`.

### 3. Verify with `examples/print_example.cpp`
*   Show printing of Scalar, Matrix (Numeric), Matrix (Symbolic).

## Integration Steps
1.  Check if `Eigen/src/Core/IO.h` error is due to missing `digits10` or similar properties in `NumTraits`.
2.  Implement `MetisIO.hpp`.

## Verification
*   Compile and run `examples/print_example.cpp`.
