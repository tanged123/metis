---
trigger: always_on
---

## Workspace Specific Rules (Metis)

### 1. Architectural Constraints (The "Red Line")
*   **Template-First**: All physics models must be templated on a generic `Scalar` type. DO NOT use `std::variant` or runtime polymorphism for core math.
*   **Dual-Backend Compatibility**: Code must compile and run correctly for both:
    *   **Numeric Mode**: `double` / `Eigen::MatrixXd` (Standard execution).
    *   **Symbolic Mode**: `casadi::MX` / `Eigen::Matrix<casadi::MX>` (Graph generation).

### 2. Math & Control Flow
*   **Math Dispatch**: Always use `metis::` namespace for math operations (e.g., `metis::sin`, `metis::pow`) instead of `std::`. This ensures proper dispatch to CasADi or std based on the type.
*   **Branching**:
    *   **NEVER** use standard C++ `if/else` on `Scalar` types (optimization variables).
    *   **ALWAYS** use `metis::where(condition, true_val, false_val)` for branching logic involving scalars.
*   **Loops**:
    *   Standard `for` loops are allowed ONLY if bounds are structural (integers/constants), not optimization variables.

### 3. Coding Style & Standards
*   **Language Standard**: C++20.
*   **Formatting**: Adhere to `treefmt` (clang-format) rules.
*   **Testing**: Write GoogleTest cases for all new functionality. Ensure tests run for both numeric and symbolic backends if applicable.

### 4. Project Structure to Respect
*   `include/metis/core/`: Concepts and Type traits.
*   `include/metis/math/`: Math dispatch logic.
*   `include/metis/linalg/`: Linear algebra extensions.
*   `tests/`: Test suite.

### 4. Read the Overview! 
*   Always make sure to read through the docs before doing anything! Get context from design_overview.md!

## workflow
*   Build: `./scripts/build.sh`
*   Test: `./scripts/test.sh`
*   Dev: `./scripts/dev.sh`