Here is the implementation roadmap for Phase 2: The Math & Numerics Layer, designed for your AI agent.
This roadmap directly maps the AeroSandbox numpy module structure (which you provided) to the Metis C++ architecture. It prioritizes the "Traceability" features essential for the Code Transformations paradigm.
metis-math-roadmap.md
Metis Phase 2: Math & Numerics Implementation Plan
Goal: Implement the "Agnostic Math Stack" that serves as a drop-in replacement for std:: and Eigen.
Constraint: Every function must be a template that works for both double (Numeric Mode) and casadi::MX (Symbolic Mode).
1. Directory Structure Update
Agent Action: Create the following headers in include/metis/math/.
Python Source (ASB)
Metis C++ Header
Responsibility
arithmetic_*.py
Arithmetic.hpp
Basic operators (+, -, *, /) and element-wise wrappers.
trig.py
Trig.hpp
Trigonometry (sin, cos, atan2) with backend dispatch.
calculus.py
Calculus.hpp
Gradients, Jacobians (Autodiff interfaces).
conditionals.py
Logic.hpp
The critical metis::where and sigmoid_blend.
finite_difference.py
DiffOps.hpp
Numerical differentiation (diff, trapz) on arrays.
interpolate.py
Interpolate.hpp
The MetisInterpolator class (1D/2D/3D).
linalg.py
Linalg.hpp
Matrix operations (solve, inv, norm).
rotations.py
Rotations.hpp
3D rotation matrices and coordinate transforms.
spacing.py
Spacing.hpp
Generators (linspace, cosine_spacing).

2. Implementation Specifications
A. Arithmetic & Trig (Arithmetic.hpp, Trig.hpp)
Critical Logic: Use if constexpr or SFINAE to route to std:: or casadi::.
Functions: sin, cos, tan, exp, log, pow, sqrt, abs.
Vectorization: Ensure these functions accept MetisMatrix (Eigen types) and apply element-wise operations automatically (using .array() in Eigen).
B. The Logic Core (Logic.hpp)
Critical Logic: This replaces standard C++ control flow.
metis::where(cond, if_true, if_false):
Numeric: Returns cond ? if_true : if_false.
Symbolic: Returns casadi::if_else(cond, if_true, if_false).
metis::sigmoid_blend(x, val_low, val_high, sharpness):
Implements smooth transition for optimization.
Comparison Operators: Overload (>, <, ==) to return bool (numeric) or casadi::MX (symbolic).
C. Differential Operators (DiffOps.hpp)
Critical Logic: Implement finite difference stencils on vectors.
metis::diff(vector): Returns adjacent differences ($x_{i+1} - x_i$).
metis::trapz(y, x): Trapezoidal integration.
metis::gradient_1d(y, x): Central difference gradient estimation.
D. Interpolation (Interpolate.hpp)
Critical Logic: Wrap stateful interpolation to support both binary search and graph nodes.
Class: MetisInterpolator (Templated class is tricky here; prefer a class that holds data and has a templated operator()).
Backend:
Numeric: std::upper_bound + linear math.
Symbolic: casadi::interpolant (builds a lookup node).
E. Linear Algebra (Linalg.hpp)
Critical Logic: Expose Eigen's power without breaking symbols.
metis::solve(A, b): Solves $Ax=b$.
Numeric: A.colPivHouseholderQr().solve(b)
Symbolic: casadi::solve(A, b)
metis::norm(x): Vector norms.
metis::outer(x, y): Outer product.
F. Rotations (Rotations.hpp)
Critical Logic: standard aerospace 3D math.
metis::rotation_matrix_2d(angle)
metis::rotation_matrix_3d(angle, axis)
Implementation: Build 3x3 Eigen matrices populated with metis::sin and metis::cos.
3. Testing Strategy (The "Dual-Check")
Every single math function needs a corresponding test in tests/ that runs twice.
Example Test Pattern:

C++


// tests/test_math.cpp
template <typename Scalar>
void test_logic_where() {
    Scalar a = 1.0;
    Scalar b = 2.0;
    
    // Test: where(a < b, 10, 20) -> should be 10
    auto res = metis::where(a < b, 10.0, 20.0);
    
    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_EQ(res, 10.0);
    } else {
        // For symbolic, we verify the graph structure or evaluate it
        // (Evaluation requires casting to function, covered in Phase 4)
        EXPECT_FALSE(res.is_empty());
    }
}

TEST(MathTests, Numeric) { test_logic_where<double>(); }
TEST(MathTests, Symbolic) { test_logic_where<casadi::MX>(); }


4. Execution Order for Agent
Scaffold: Create all .hpp files in the include tree.
Core Math: Implement Arithmetic.hpp and Trig.hpp first (dependencies for everything else).
Tests 1: Verify basic math compiles for both backends.
Logic: Implement Logic.hpp (where operator). This is the hardest template challenge.
Tests 2: Verify branching logic.
Linalg/DiffOps: Implement array helpers using Eigen.
Interpolation: Implement the MetisInterpolator class.
Final Polish: Format code with clang-format.
