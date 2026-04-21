#include <cmath>
#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisError.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/AutoDiff.hpp>
#include <metis/math/ScatteredInterpolator.hpp>

// ============================================================================
// 1D Scattered Interpolation Tests
// ============================================================================

TEST(ScatteredInterpolatorTest, Basic1D) {
    // Scattered data from y = x^2
    metis::NumericVector x(10);
    metis::NumericVector y(10);

    // Non-uniform spacing
    x << 0.0, 0.3, 0.7, 1.2, 1.8, 2.5, 3.1, 3.6, 4.2, 5.0;
    for (int i = 0; i < 10; ++i) {
        y(i) = x(i) * x(i);
    }

    metis::ScatteredInterpolator interp(x, y);

    EXPECT_TRUE(interp.valid());
    EXPECT_EQ(interp.dims(), 1);

    // Test at known points
    EXPECT_NEAR(interp(2.0), 4.0, 0.5); // Should be close to x^2 = 4
    EXPECT_NEAR(interp(3.0), 9.0, 0.5); // Should be close to x^2 = 9
}

TEST(ScatteredInterpolatorTest, Basic1D_Linear) {
    // Perfect linear data - should interpolate exactly
    metis::NumericVector x(5);
    metis::NumericVector y(5);

    x << 0.0, 1.0, 2.0, 3.0, 4.0;
    y << 0.0, 2.0, 4.0, 6.0, 8.0; // y = 2x

    metis::ScatteredInterpolator interp(x, y);

    EXPECT_NEAR(interp(1.5), 3.0, 0.1);
    EXPECT_NEAR(interp(2.5), 5.0, 0.1);
}

TEST(ScatteredInterpolatorTest, ReconstructionError) {
    metis::NumericVector x(20);
    metis::NumericVector y(20);

    // Uniform data
    for (int i = 0; i < 20; ++i) {
        x(i) = static_cast<double>(i) * 0.5;
        y(i) = std::sin(x(i));
    }

    metis::ScatteredInterpolator interp(x, y, 100); // High resolution for good fit

    // Reconstruction error should be small for smooth function
    EXPECT_LT(interp.reconstruction_error(), 0.1);
}

// ============================================================================
// 2D Scattered Interpolation Tests
// ============================================================================

TEST(ScatteredInterpolatorTest, Basic2D) {
    // 2D scattered data: z = x + y
    int n = 25;
    metis::NumericMatrix points(n, 2);
    metis::NumericVector values(n);

    int idx = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            double xi = static_cast<double>(i);
            double yi = static_cast<double>(j);
            points(idx, 0) = xi;
            points(idx, 1) = yi;
            values(idx) = xi + yi;
            ++idx;
        }
    }

    // Higher grid resolution for better RBF approximation
    metis::ScatteredInterpolator interp(points, values, 30);

    EXPECT_TRUE(interp.valid());
    EXPECT_EQ(interp.dims(), 2);

    // Test at midpoint (RBF approximation has some error)
    metis::NumericVector query(2);
    query << 2.5, 2.5;
    EXPECT_NEAR(interp(query), 5.0, 1.0); // Wider tolerance for RBF
}

TEST(ScatteredInterpolatorTest, Scattered2D_Random) {
    // Truly scattered (random-ish) 2D data
    int n = 16;
    metis::NumericMatrix points(n, 2);
    metis::NumericVector values(n);

    // Pseudo-random points
    double xs[] = {0.1,  0.9,  0.2,  0.8,  0.3,  0.7,  0.4,  0.6,
                   0.15, 0.85, 0.25, 0.75, 0.35, 0.65, 0.45, 0.55};
    double ys[] = {0.2, 0.8, 0.3,  0.7,  0.4,  0.6,  0.5,  0.5,
                   0.1, 0.9, 0.15, 0.85, 0.25, 0.75, 0.35, 0.65};

    for (int i = 0; i < n; ++i) {
        points(i, 0) = xs[i];
        points(i, 1) = ys[i];
        values(i) = xs[i] * xs[i] + ys[i] * ys[i]; // z = x^2 + y^2
    }

    metis::ScatteredInterpolator interp(points, values, 20);

    // Query at center
    metis::NumericVector query(2);
    query << 0.5, 0.5;
    double expected = 0.5; // 0.25 + 0.25
    EXPECT_NEAR(interp(query), expected, 0.2);
}

// ============================================================================
// 3D Scattered Interpolation Tests
// ============================================================================

TEST(ScatteredInterpolatorTest, Basic3D) {
    // 3D data: w = x + y + z
    int n = 27; // 3x3x3
    metis::NumericMatrix points(n, 3);
    metis::NumericVector values(n);

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                points(idx, 0) = static_cast<double>(i);
                points(idx, 1) = static_cast<double>(j);
                points(idx, 2) = static_cast<double>(k);
                values(idx) = i + j + k;
                ++idx;
            }
        }
    }

    // Higher grid resolution for 3D RBF
    metis::ScatteredInterpolator interp(points, values, 15);

    EXPECT_EQ(interp.dims(), 3);

    metis::NumericVector query(3);
    query << 1.0, 1.0, 1.0;
    EXPECT_NEAR(interp(query), 3.0, 1.0); // Wider tolerance for RBF
}

// ============================================================================
// Kernel Tests
// ============================================================================

TEST(ScatteredInterpolatorTest, DifferentKernels) {
    // Build scattered data as explicit matrix (not reshaped vector)
    metis::NumericMatrix points(10, 1);
    metis::NumericVector values(10);

    for (int i = 0; i < 10; ++i) {
        points(i, 0) = static_cast<double>(i);
        values(i) = std::sin(points(i, 0));
    }

    // Test each kernel type using N-D constructor explicitly
    metis::ScatteredInterpolator tps(points, values, 30, metis::RBFKernel::ThinPlateSpline);
    EXPECT_TRUE(tps.valid());

    metis::ScatteredInterpolator mq(points, values, 30, metis::RBFKernel::Multiquadric);
    EXPECT_TRUE(mq.valid());

    metis::ScatteredInterpolator gauss(points, values, 30, metis::RBFKernel::Gaussian);
    EXPECT_TRUE(gauss.valid());

    metis::ScatteredInterpolator linear(points, values, 30, metis::RBFKernel::Linear);
    EXPECT_TRUE(linear.valid());

    metis::ScatteredInterpolator cubic(points, values, 30, metis::RBFKernel::Cubic);
    EXPECT_TRUE(cubic.valid());
}

// ============================================================================
// Symbolic Compatibility Tests
// ============================================================================

TEST(ScatteredInterpolatorTest, SymbolicEvaluation) {
    // Create simple 1D interpolator
    metis::NumericVector x(10);
    metis::NumericVector y(10);

    for (int i = 0; i < 10; ++i) {
        x(i) = static_cast<double>(i);
        y(i) = x(i) * x(i);
    }

    metis::ScatteredInterpolator interp(x, y, 50);

    // Symbolic query
    metis::SymbolicScalar sym_x = casadi::MX::sym("x");
    metis::SymbolicScalar result = interp(sym_x);

    // Create metis::Function and evaluate
    metis::Function f("test", {sym_x}, {result});

    // Evaluate at x=3
    auto res = f(3.0);
    double value = res[0](0, 0);

    EXPECT_NEAR(value, 9.0, 0.5); // Should be close to 3² = 9
}

TEST(ScatteredInterpolatorTest, SymbolicEvaluation2D) {
    // Create 2D interpolator
    int n = 25;
    metis::NumericMatrix points(n, 2);
    metis::NumericVector values(n);

    int idx = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            points(idx, 0) = static_cast<double>(i);
            points(idx, 1) = static_cast<double>(j);
            values(idx) = points(idx, 0) + points(idx, 1);
            ++idx;
        }
    }

    // Higher resolution for better RBF approximation
    metis::ScatteredInterpolator interp(points, values, 30);

    // Symbolic 2D query using individual symbols
    auto x_sym = metis::sym("x");
    auto y_sym = metis::sym("y");
    metis::SymbolicVector query(2);
    query << x_sym, y_sym;

    metis::SymbolicScalar result = interp(query);

    metis::Function f("test2d", {x_sym, y_sym}, {result});

    // Evaluate at (2, 3)
    auto res = f(2.0, 3.0);
    double value = res[0](0, 0);

    EXPECT_NEAR(value, 5.0, 1.0); // 2 + 3 = 5 (wider tolerance for RBF)
}

TEST(ScatteredInterpolatorTest, SymbolicGradient) {
    // Test that gradients work through the interpolator
    metis::NumericVector x(10);
    metis::NumericVector y(10);

    for (int i = 0; i < 10; ++i) {
        x(i) = static_cast<double>(i);
        y(i) = x(i) * x(i); // y = x²
    }

    metis::ScatteredInterpolator interp(x, y, 100);

    auto sym_x = metis::sym("x");
    metis::SymbolicScalar result = interp(sym_x);

    // Get Jacobian using metis helper
    auto jac = metis::jacobian(result, sym_x);
    metis::Function df("df", {sym_x}, {jac});

    // Evaluate gradient at x=3
    // For y=x², dy/dx = 2x, so at x=3, gradient ≈ 6
    auto grad = df(3.0);
    double grad_value = grad[0](0, 0);

    EXPECT_NEAR(grad_value, 6.0, 1.0); // dy/dx ≈ 2*3 = 6
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(ScatteredInterpolatorTest, ErrorEmptyPoints) {
    metis::NumericMatrix empty(0, 2);
    metis::NumericVector values(0);

    EXPECT_THROW(metis::ScatteredInterpolator(empty, values), metis::InterpolationError);
}

TEST(ScatteredInterpolatorTest, ErrorSizeMismatch) {
    metis::NumericMatrix points(10, 2);
    metis::NumericVector values(5); // Wrong size

    EXPECT_THROW(metis::ScatteredInterpolator(points, values), metis::InterpolationError);
}

TEST(ScatteredInterpolatorTest, ErrorLowResolution) {
    metis::NumericMatrix points(5, 1);
    metis::NumericVector values(5);
    points << 0, 1, 2, 3, 4;
    values << 0, 1, 4, 9, 16;

    // grid_resolution must be >= 2
    EXPECT_THROW(metis::ScatteredInterpolator(points, values, 1), metis::InterpolationError);
}

TEST(ScatteredInterpolatorTest, ErrorUninitializedQuery) {
    metis::ScatteredInterpolator uninit;

    EXPECT_THROW(uninit(1.0), metis::InterpolationError);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ScatteredInterpolatorTest, ConstantFunction) {
    // All values are the same
    metis::NumericVector x(5), y(5);
    x << 0, 1, 2, 3, 4;
    y << 7, 7, 7, 7, 7;

    metis::ScatteredInterpolator interp(x, y);

    EXPECT_NEAR(interp(2.5), 7.0, 0.1);
}

TEST(ScatteredInterpolatorTest, FewPoints) {
    // Test with small number of points - use Linear RBF kernel
    // (Thin plate spline is ill-conditioned for very few points)
    metis::NumericVector x(3), y(3);
    x << 0, 0.5, 1;
    y << 0, 1, 2; // y = 2x

    // Use Linear kernel which is stable for few points
    metis::ScatteredInterpolator interp(x, y, 50, metis::RBFKernel::Linear);
    EXPECT_TRUE(interp.valid());
    EXPECT_NEAR(interp(0.25), 0.5, 0.5); // y ≈ 0.5
}

TEST(ScatteredInterpolatorTest, ExplicitGrid) {
    // Use custom grid specification
    int n = 9;
    metis::NumericMatrix points(n, 2);
    metis::NumericVector values(n);

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            points(idx, 0) = static_cast<double>(i);
            points(idx, 1) = static_cast<double>(j);
            values(idx) = i * j;
            ++idx;
        }
    }

    // Custom grid with higher resolutions
    std::vector<metis::NumericVector> grid(2);
    grid[0] = metis::NumericVector::LinSpaced(25, -0.5, 2.5);
    grid[1] = metis::NumericVector::LinSpaced(25, -0.5, 2.5);

    metis::ScatteredInterpolator interp(points, values, grid);
    EXPECT_TRUE(interp.valid());

    metis::NumericVector query(2);
    query << 1.0, 2.0;
    EXPECT_NEAR(interp(query), 2.0, 1.0); // 1 * 2 = 2 (RBF tolerance)
}

TEST(ScatteredInterpolatorTest, Extrapolation) {
    // Test extrapolation (outside data range)
    // The gridded interpolator clamps by default
    metis::NumericVector x(5), y(5);
    x << 0, 1, 2, 3, 4;
    y << 0, 2, 4, 6, 8; // y = 2x

    metis::ScatteredInterpolator interp(x, y, 50);

    // Query outside data range - should clamp to boundary
    double val_left = interp(-1.0); // Outside left
    double val_right = interp(5.0); // Outside right

    // Clamped: should return values at boundaries (0 and 8)
    EXPECT_NEAR(val_left, 0.0, 1.0);
    EXPECT_NEAR(val_right, 8.0, 1.0);
}
