#include "../utils/TestUtils.hpp"
#include <gtest/gtest.h>
#include <metis/core/Function.hpp>
#include <metis/core/MetisError.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/AutoDiff.hpp>
#include <metis/math/Interpolate.hpp>
#include <string>

namespace {

void expect_hermite_symbolic_error(const metis::InterpolationError &err) {
    const std::string message = err.what();
    EXPECT_NE(message.find("Hermite/Catmull-Rom"), std::string::npos);
    EXPECT_NE(message.find("runtime comparisons"), std::string::npos);
    EXPECT_NE(message.find("BSpline"), std::string::npos);
}

void expect_symbolic_table_values_error(const metis::InterpolationError &err) {
    const std::string message = err.what();
    EXPECT_NE(message.find("symbolic table values"), std::string::npos);
    EXPECT_NE(message.find("BSpline"), std::string::npos);
}

} // namespace

// ============================================================================
// Interpolator Tests (1D Interpolation Class)
// ============================================================================

template <typename Scalar> void test_interp1d() {
    // x = [0, 1, 2]
    // y = [0, 10, 0]
    metis::NumericVector x(3);
    x << 0.0, 1.0, 2.0;
    metis::NumericVector y(3);
    y << 0.0, 10.0, 0.0;

    metis::Interpolator interp(x, y); // Default: Linear

    Scalar query_mid = 0.5; // Expect 5.0
    auto res_mid = interp(query_mid);

    Scalar query_right = 1.5; // Expect 5.0
    auto res_right = interp(query_right);

    // Query at boundary (clamped)
    Scalar query_bound = 2.0;
    auto res_bound = interp(query_bound);

    if constexpr (std::is_same_v<Scalar, double>) {
        EXPECT_DOUBLE_EQ(res_mid, 5.0);
        EXPECT_DOUBLE_EQ(res_right, 5.0);
        EXPECT_DOUBLE_EQ(res_bound, 0.0); // Value at x=2
    } else {
        EXPECT_DOUBLE_EQ(metis::eval(res_mid), 5.0);
        EXPECT_DOUBLE_EQ(metis::eval(res_right), 5.0);
        EXPECT_DOUBLE_EQ(metis::eval(res_bound), 0.0);
    }
}

TEST(InterpolatorTests, Numeric) { test_interp1d<double>(); }

TEST(InterpolatorTests, Symbolic) { test_interp1d<metis::SymbolicScalar>(); }

TEST(InterpolatorTests, CoverageErrorChecks) {
    metis::NumericVector x(3);
    x << 0, 1, 2;
    metis::NumericVector y(2);
    y << 0, 1;

    // Mismatched size
    EXPECT_THROW(metis::Interpolator(x, y), metis::InterpolationError);

    // Size < 2
    metis::NumericVector x1(1);
    x1 << 0;
    metis::NumericVector y1(1);
    y1 << 0;
    EXPECT_THROW(metis::Interpolator(x1, y1), metis::InterpolationError);

    // Unsorted
    metis::NumericVector xu(3);
    xu << 0, 2, 1;
    metis::NumericVector yu(3);
    yu << 0, 0, 0;
    EXPECT_THROW(metis::Interpolator(xu, yu), metis::InterpolationError);

    // Uninitialized use
    metis::Interpolator empty;
    EXPECT_THROW(empty(1.0), metis::InterpolationError);

    // Uninitialized matrix use
    metis::NumericMatrix q(1, 1);
    q << 1.0;
    EXPECT_THROW(empty(q), metis::InterpolationError);
}

TEST(InterpolatorTests, BoundsClamping) {
    metis::NumericVector x(3);
    x << 0, 1, 2;
    metis::NumericVector y(3);
    y << 0, 10, 20;
    metis::Interpolator interp(x, y);

    // Query outside bounds - should clamp
    EXPECT_DOUBLE_EQ(interp(-1.0), 0.0); // Clamps to x=0, y=0
    EXPECT_DOUBLE_EQ(interp(5.0), 20.0); // Clamps to x=2, y=20
}

TEST(InterpolatorTests, HermiteMethod) {
    // Test Hermite (C1) interpolation
    metis::NumericVector x(4);
    x << 0, 1, 2, 3;
    metis::NumericVector y(4);
    y << 0, 1, 4, 9; // y = x^2

    metis::Interpolator interp(x, y, metis::InterpolationMethod::Hermite);
    EXPECT_EQ(interp.method(), metis::InterpolationMethod::Hermite);

    // Should produce smooth interpolation
    double result = interp(1.5);
    EXPECT_GT(result, 1.0); // Between y(1)=1 and y(2)=4
    EXPECT_LT(result, 4.0);
}

TEST(InterpolatorTests, BSplineMethod) {
    // Test BSpline (C2) interpolation
    metis::NumericVector x(4);
    x << 0, 1, 2, 3;
    metis::NumericVector y(4);
    y << 1, 1, 1, 1; // Constant function

    metis::Interpolator interp(x, y, metis::InterpolationMethod::BSpline);
    EXPECT_EQ(interp.method(), metis::InterpolationMethod::BSpline);

    // Constant should interpolate exactly
    EXPECT_NEAR(interp(0.5), 1.0, 1e-10);
    EXPECT_NEAR(interp(1.5), 1.0, 1e-10);
    EXPECT_NEAR(interp(2.5), 1.0, 1e-10);
}

TEST(InterpolatorTests, BSplineRequires4Points) {
    // BSpline should fail with < 4 points
    metis::NumericVector x(3);
    x << 0, 1, 2;
    metis::NumericVector y(3);
    y << 0, 1, 2;

    EXPECT_THROW(metis::Interpolator(x, y, metis::InterpolationMethod::BSpline),
                 metis::InterpolationError);
}

TEST(InterpolatorTests, NearestMethod) {
    // Test Nearest neighbor
    metis::NumericVector x(3);
    x << 0, 1, 2;
    metis::NumericVector y(3);
    y << 0, 10, 20;

    metis::Interpolator interp(x, y, metis::InterpolationMethod::Nearest);

    // Nearest to x=0
    EXPECT_DOUBLE_EQ(interp(0.4), 0.0);
    // Nearest to x=1
    EXPECT_DOUBLE_EQ(interp(0.6), 10.0);
    EXPECT_DOUBLE_EQ(interp(1.4), 10.0);
    // Nearest to x=2
    EXPECT_DOUBLE_EQ(interp(1.6), 20.0);
}

TEST(InterpolatorTests, HermiteSymbolicNotSupported) {
    metis::NumericVector x(4);
    x << 0, 1, 2, 3;
    metis::NumericVector y(4);
    y << 0, 1, 4, 9;

    metis::Interpolator interp(x, y, metis::InterpolationMethod::Hermite);

    // Symbolic should throw with guidance toward BSpline
    metis::SymbolicScalar query = metis::sym("q");
    try {
        static_cast<void>(interp(query));
        FAIL() << "Expected InterpolationError for symbolic Hermite query";
    } catch (const metis::InterpolationError &err) {
        expect_hermite_symbolic_error(err);
    }
}

TEST(InterpolatorTests, HermiteSymbolicBatchNotSupported) {
    metis::NumericVector x(4);
    x << 0, 1, 2, 3;
    metis::NumericVector y(4);
    y << 0, 1, 4, 9;

    metis::Interpolator interp(x, y, metis::InterpolationMethod::Hermite);

    metis::SymbolicMatrix queries(2, 1);
    queries(0, 0) = metis::sym("q0");
    queries(1, 0) = metis::sym("q1");

    try {
        static_cast<void>(interp(queries));
        FAIL() << "Expected InterpolationError for symbolic Hermite batch query";
    } catch (const metis::InterpolationError &err) {
        expect_hermite_symbolic_error(err);
    }
}

TEST(InterpolatorTests, BSplineSymbolic) {
    // BSpline should work with symbolic
    metis::NumericVector x(4);
    x << 0, 1, 2, 3;
    metis::NumericVector y(4);
    y << 1, 1, 1, 1;

    metis::Interpolator interp(x, y, metis::InterpolationMethod::BSpline);

    metis::SymbolicScalar query = casadi::MX(1.5);
    auto result = interp(query);

    EXPECT_NEAR(eval_scalar(result), 1.0, 1e-9);
}

TEST(InterpolatorTests, VectorizedQuery) {
    // Test vectorized queries - pass as matrix for batch evaluation
    metis::NumericVector x(3);
    x << 0, 1, 2;
    metis::NumericVector y(3);
    y << 0, 10, 20;

    metis::Interpolator interp(x, y);

    // Create as matrix (Nx1) for batch query
    metis::NumericMatrix queries(3, 1);
    queries << 0.5, 1.0, 1.5;

    auto results = interp(queries);

    EXPECT_NEAR(results(0), 5.0, 1e-10);
    EXPECT_NEAR(results(1), 10.0, 1e-10);
    EXPECT_NEAR(results(2), 15.0, 1e-10);
}

// ============================================================================
// N-Dimensional Interpolation Tests (interpn)
// ============================================================================

TEST(InterpnTests, Numeric2DLinear) {
    // 2D grid: x = [0, 1], y = [0, 1]
    // Values: z(x, y) = x + y
    // z(0,0)=0, z(1,0)=1, z(0,1)=1, z(1,1)=2
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // Values in Fortran order: (0,0), (1,0), (0,1), (1,1) -> 0, 1, 1, 2
    metis::NumericVector values(4);
    values << 0.0, 1.0, 1.0, 2.0;

    // Query at (0.5, 0.5) - should get 1.0
    metis::NumericMatrix xi(1, 2);
    xi << 0.5, 0.5;

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Linear);

    EXPECT_NEAR(result(0), 1.0, 1e-10);
}

TEST(InterpnTests, Numeric2DMultiplePoints) {
    // 3x3 grid: x = [0, 1, 2], y = [0, 1, 2]
    // Values: z(x, y) = x * y
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    metis::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // Values in Fortran order:
    // (0,0)=0, (1,0)=0, (2,0)=0, (0,1)=0, (1,1)=1, (2,1)=2, (0,2)=0, (1,2)=2, (2,2)=4
    metis::NumericVector values(9);
    values << 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 4.0;

    // Multiple query points
    metis::NumericMatrix xi(3, 2);
    xi << 0.5, 0.5, // Should give ~0.25
        1.0, 1.0,   // Should give 1.0 exactly
        1.5, 1.5;   // Should give ~2.25

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Linear);

    EXPECT_NEAR(result(0), 0.25, 1e-10);
    EXPECT_NEAR(result(1), 1.0, 1e-10);
    EXPECT_NEAR(result(2), 2.25, 1e-10);
}

TEST(InterpnTests, Numeric2DBSpline) {
    // Test BSpline method - needs at least 4 points per dimension for cubic
    // 4x4 grid with constant function for easy verification
    metis::NumericVector x_pts(4);
    x_pts << 0.0, 1.0, 2.0, 3.0;
    metis::NumericVector y_pts(4);
    y_pts << 0.0, 1.0, 2.0, 3.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // Values: z = 1 (constant for easy verification)
    metis::NumericVector values(16);
    values.setConstant(1.0);

    metis::NumericMatrix xi(1, 2);
    xi << 1.5, 1.5;

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::BSpline);

    // For constant function, bspline should also give 1.0
    EXPECT_NEAR(result(0), 1.0, 1e-10);
}

TEST(InterpnTests, NumericFillValue) {
    // Test out-of-bounds with fill_value
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};
    metis::NumericVector values(4);
    values << 0.0, 1.0, 1.0, 2.0;

    // Query outside bounds
    metis::NumericMatrix xi(2, 2);
    xi << 0.5, 0.5, // In bounds
        2.0, 0.5;   // Out of bounds in x

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Linear,
                                         std::optional<double>(-999.0));

    EXPECT_NEAR(result(0), 1.0, 1e-10);    // In bounds
    EXPECT_NEAR(result(1), -999.0, 1e-10); // Fill value
}

TEST(InterpnTests, NumericExtrapolation) {
    // Without fill_value, should clamp to bounds
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};
    metis::NumericVector values(4);
    values << 0.0, 1.0, 1.0, 2.0;

    // Query outside bounds - should clamp
    metis::NumericMatrix xi(1, 2);
    xi << 2.0, 0.5; // x=2 should clamp to x=1

    auto result = metis::interpn<double>(points, values, xi);

    // At (1.0, 0.5): interpolate between z(1,0)=1 and z(1,1)=2 -> 1.5
    EXPECT_NEAR(result(0), 1.5, 1e-10);
}

TEST(InterpnTests, Symbolic2DLinear) {
    // 2D grid symbolic test
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // z(x,y) = x + y
    metis::NumericVector values(4);
    values << 0.0, 1.0, 1.0, 2.0;

    // Symbolic query - use fixed numeric values for simplicity
    // Query at (0.5, 0.5) which should give 1.0
    Eigen::Matrix<metis::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 2);
    xi(0, 0) = casadi::MX(0.5);
    xi(0, 1) = casadi::MX(0.5);

    auto result = metis::interpn<metis::SymbolicScalar>(points, values, xi);

    // Evaluate the symbolic result (no variables, just constants)
    EXPECT_NEAR(eval_scalar(result(0)), 1.0, 1e-9);
}

TEST(InterpnTests, SymbolicValues2DLinearWeights) {
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;
    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    auto [values_vec, values_mx] = metis::sym_vec_pair("table_values", 4);
    metis::SymbolicScalar qx = metis::sym("qx");
    metis::SymbolicScalar qy = metis::sym("qy");
    metis::SymbolicMatrix xi(1, 2);
    xi(0, 0) = qx;
    xi(0, 1) = qy;

    auto result = metis::interpn(points, values_vec, xi, metis::InterpolationMethod::Linear);
    auto jac = casadi::MX::jacobian(metis::to_mx(result), values_mx);
    casadi::Function eval_fn("eval_parametric_linear", {qx, qy, values_mx},
                             {metis::to_mx(result), jac});

    auto outputs = eval_fn(std::vector<casadi::DM>{
        casadi::DM(0.25), casadi::DM(0.75), casadi::DM(std::vector<double>{0.0, 1.0, 2.0, 3.0})});
    std::vector<double> jacobian_values = std::vector<double>(outputs[1]);

    EXPECT_NEAR(double(outputs[0]), 1.75, 1e-9);
    ASSERT_EQ(jacobian_values.size(), 4u);
    EXPECT_NEAR(jacobian_values[0], 0.1875, 1e-9);
    EXPECT_NEAR(jacobian_values[1], 0.0625, 1e-9);
    EXPECT_NEAR(jacobian_values[2], 0.5625, 1e-9);
    EXPECT_NEAR(jacobian_values[3], 0.1875, 1e-9);
}

TEST(InterpnTests, SymbolicValuesInterpnLinear2D) {
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;
    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    metis::SymbolicScalar v00 = metis::sym("v00");
    metis::SymbolicScalar v10 = metis::sym("v10");
    metis::SymbolicScalar v01 = metis::sym("v01");
    metis::SymbolicScalar v11 = metis::sym("v11");

    // Flatten in Fortran (column-major) order: (0,0), (1,0), (0,1), (1,1)
    metis::SymbolicVector values(4);
    values(0) = v00;
    values(1) = v10;
    values(2) = v01;
    values(3) = v11;

    metis::MetisMatrix<metis::SymbolicScalar> xi(1, 2);
    xi(0, 0) = metis::SymbolicScalar(0.25);
    xi(0, 1) = metis::SymbolicScalar(0.75);

    auto result = metis::interpn<metis::SymbolicScalar>(points, values, xi,
                                                        metis::InterpolationMethod::Linear);
    casadi::Function eval_fn("eval_parametric_linear_2d", {v00, v10, v01, v11},
                             {metis::to_mx(result)});
    auto outputs = eval_fn(std::vector<casadi::DM>{casadi::DM(0.0), casadi::DM(1.0),
                                                   casadi::DM(2.0), casadi::DM(3.0)});

    EXPECT_NEAR(double(outputs[0]), 1.75, 1e-9);
}

TEST(InterpnTests, SymbolicValuesBSplineConstant) {
    metis::NumericVector x_pts(4);
    x_pts << 0.0, 1.0, 2.0, 3.0;
    metis::NumericVector y_pts(4);
    y_pts << 0.0, 1.0, 2.0, 3.0;
    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    metis::SymbolicScalar c = metis::sym("c");
    metis::SymbolicVector values(16);
    for (int i = 0; i < values.size(); ++i) {
        values(i) = c;
    }

    metis::NumericMatrix xi(1, 2);
    xi << 1.5, 1.5;

    auto result = metis::interpn(points, values, xi, metis::InterpolationMethod::BSpline);
    auto jac = casadi::MX::jacobian(metis::to_mx(result), c);
    casadi::Function eval_fn("eval_parametric_bspline", {c}, {metis::to_mx(result), jac});
    auto outputs = eval_fn(std::vector<casadi::DM>{casadi::DM(3.25)});

    EXPECT_NEAR(double(outputs[0]), 3.25, 1e-9);
    EXPECT_NEAR(double(outputs[1]), 1.0, 1e-9);
}

TEST(InterpnTests, SymbolicValuesHermiteNotSupported) {
    metis::NumericVector x_pts(4);
    x_pts << 0.0, 1.0, 2.0, 3.0;
    std::vector<metis::NumericVector> points = {x_pts};
    auto values = metis::sym_vec("table_values_1d", 4);

    metis::NumericMatrix xi(1, 1);
    xi << 0.5;

    try {
        static_cast<void>(metis::interpn(points, values, xi, metis::InterpolationMethod::Hermite));
        FAIL() << "Expected InterpolationError for Hermite with symbolic table values";
    } catch (const metis::InterpolationError &err) {
        expect_symbolic_table_values_error(err);
    }
}

TEST(InterpnTests, Numeric3D) {
    // 3D interpolation: 2x2x2 grid
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;
    metis::NumericVector z_pts(2);
    z_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts, z_pts};

    // Values: f(x,y,z) = x + y + z
    // Fortran order: iterate x fastest, then y, then z
    // (0,0,0)=0, (1,0,0)=1, (0,1,0)=1, (1,1,0)=2, (0,0,1)=1, (1,0,1)=2, (0,1,1)=2, (1,1,1)=3
    metis::NumericVector values(8);
    values << 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0;

    // Query at center (0.5, 0.5, 0.5) -> should give 1.5
    metis::NumericMatrix xi(1, 3);
    xi << 0.5, 0.5, 0.5;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 1.5, 1e-10);
}

TEST(InterpnTests, Numeric4D) {
    // 4D interpolation: 2^4 = 16 grid points
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {pts, pts, pts, pts};

    // Values: f(x1, x2, x3, x4) = x1 + x2 + x3 + x4
    // 2^4 = 16 values in Fortran order
    metis::NumericVector values(16);
    int idx = 0;
    for (int i4 = 0; i4 < 2; ++i4) {
        for (int i3 = 0; i3 < 2; ++i3) {
            for (int i2 = 0; i2 < 2; ++i2) {
                for (int i1 = 0; i1 < 2; ++i1) {
                    values(idx++) = i1 + i2 + i3 + i4;
                }
            }
        }
    }

    // Query at center (0.5, 0.5, 0.5, 0.5) -> should give 2.0
    metis::NumericMatrix xi(1, 4);
    xi << 0.5, 0.5, 0.5, 0.5;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 2.0, 1e-10);
}

TEST(InterpnTests, Numeric5D) {
    // 5D interpolation: 2^5 = 32 grid points
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {pts, pts, pts, pts, pts};

    // Values: f = sum of all coordinates
    metis::NumericVector values(32);
    int idx = 0;
    for (int i5 = 0; i5 < 2; ++i5) {
        for (int i4 = 0; i4 < 2; ++i4) {
            for (int i3 = 0; i3 < 2; ++i3) {
                for (int i2 = 0; i2 < 2; ++i2) {
                    for (int i1 = 0; i1 < 2; ++i1) {
                        values(idx++) = i1 + i2 + i3 + i4 + i5;
                    }
                }
            }
        }
    }

    // Query at center -> should give 2.5
    metis::NumericMatrix xi(1, 5);
    xi << 0.5, 0.5, 0.5, 0.5, 0.5;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 2.5, 1e-10);
}

TEST(InterpnTests, Numeric6D) {
    // 6D interpolation: 2^6 = 64 grid points
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {pts, pts, pts, pts, pts, pts};

    // Values: f = sum of all coordinates
    metis::NumericVector values(64);
    int idx = 0;
    for (int i6 = 0; i6 < 2; ++i6) {
        for (int i5 = 0; i5 < 2; ++i5) {
            for (int i4 = 0; i4 < 2; ++i4) {
                for (int i3 = 0; i3 < 2; ++i3) {
                    for (int i2 = 0; i2 < 2; ++i2) {
                        for (int i1 = 0; i1 < 2; ++i1) {
                            values(idx++) = i1 + i2 + i3 + i4 + i5 + i6;
                        }
                    }
                }
            }
        }
    }

    // Query at center -> should give 3.0
    metis::NumericMatrix xi(1, 6);
    xi << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 3.0, 1e-10);
}

TEST(InterpnTests, Numeric7D) {
    // 7D interpolation: 2^7 = 128 grid points
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {pts, pts, pts, pts, pts, pts, pts};

    // Values: f = sum of all coordinates
    metis::NumericVector values(128);
    int idx = 0;
    for (int i7 = 0; i7 < 2; ++i7) {
        for (int i6 = 0; i6 < 2; ++i6) {
            for (int i5 = 0; i5 < 2; ++i5) {
                for (int i4 = 0; i4 < 2; ++i4) {
                    for (int i3 = 0; i3 < 2; ++i3) {
                        for (int i2 = 0; i2 < 2; ++i2) {
                            for (int i1 = 0; i1 < 2; ++i1) {
                                values(idx++) = i1 + i2 + i3 + i4 + i5 + i6 + i7;
                            }
                        }
                    }
                }
            }
        }
    }

    // Query at center -> should give 3.5
    metis::NumericMatrix xi(1, 7);
    xi << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 3.5, 1e-10);
}

TEST(InterpnTests, ErrorEmptyPoints) {
    std::vector<metis::NumericVector> points;
    metis::NumericVector values(1);
    values << 1.0;
    metis::NumericMatrix xi(1, 1);
    xi << 0.5;

    EXPECT_THROW(metis::interpn<double>(points, values, xi), metis::InterpolationError);
}

TEST(InterpnTests, ErrorUnsortedPoints) {
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 2.0, 1.0; // Not sorted!
    std::vector<metis::NumericVector> points = {x_pts};
    metis::NumericVector values(3);
    values << 1.0, 2.0, 3.0;
    metis::NumericMatrix xi(1, 1);
    xi << 0.5;

    EXPECT_THROW(metis::interpn<double>(points, values, xi), metis::InterpolationError);
}

TEST(InterpnTests, ErrorValuesSizeMismatch) {
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;
    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // Wrong size: should be 4, not 3
    metis::NumericVector values(3);
    values << 1.0, 2.0, 3.0;
    metis::NumericMatrix xi(1, 2);
    xi << 0.5, 0.5;

    EXPECT_THROW(metis::interpn<double>(points, values, xi), metis::InterpolationError);
}

// ============================================================================
// Edge and Corner Cases
// ============================================================================

TEST(InterpnTests, QueryAtGridPoints2D) {
    // Query exactly at grid points should return exact values
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    metis::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // z = x + 2*y
    // Fortran order: iterate x fastest
    metis::NumericVector values(9);
    values << 0, 1, 2, // y=0: (0,0)=0, (1,0)=1, (2,0)=2
        2, 3, 4,       // y=1: (0,1)=2, (1,1)=3, (2,1)=4
        4, 5, 6;       // y=2: (0,2)=4, (1,2)=5, (2,2)=6

    // Query all grid points
    metis::NumericMatrix xi(9, 2);
    xi << 0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2;

    auto result = metis::interpn<double>(points, values, xi);

    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(result(i), values(i), 1e-10) << "Mismatch at grid point " << i;
    }
}

TEST(InterpnTests, QueryAtEdges2D) {
    // Query along edges (one coordinate at boundary)
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    metis::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // z = x + y (simple to verify)
    metis::NumericVector values(9);
    values << 0, 1, 2, // y=0
        1, 2, 3,       // y=1
        2, 3, 4;       // y=2

    // Query along bottom edge (y=0), top edge (y=2), left edge (x=0), right edge (x=2)
    metis::NumericMatrix xi(8, 2);
    xi << 0.5, 0.0, // bottom edge
        1.5, 0.0,   // bottom edge
        0.5, 2.0,   // top edge
        1.5, 2.0,   // top edge
        0.0, 0.5,   // left edge
        0.0, 1.5,   // left edge
        2.0, 0.5,   // right edge
        2.0, 1.5;   // right edge

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 0.5, 1e-10); // (0.5, 0) -> 0.5
    EXPECT_NEAR(result(1), 1.5, 1e-10); // (1.5, 0) -> 1.5
    EXPECT_NEAR(result(2), 2.5, 1e-10); // (0.5, 2) -> 2.5
    EXPECT_NEAR(result(3), 3.5, 1e-10); // (1.5, 2) -> 3.5
    EXPECT_NEAR(result(4), 0.5, 1e-10); // (0, 0.5) -> 0.5
    EXPECT_NEAR(result(5), 1.5, 1e-10); // (0, 1.5) -> 1.5
    EXPECT_NEAR(result(6), 2.5, 1e-10); // (2, 0.5) -> 2.5
    EXPECT_NEAR(result(7), 3.5, 1e-10); // (2, 1.5) -> 3.5
}

TEST(InterpnTests, QueryAtCorners2D) {
    // Query exactly at all four corners
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    metis::NumericVector values(4);
    values << 1.0, 2.0, 3.0, 4.0; // corners: (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4

    metis::NumericMatrix xi(4, 2);
    xi << 0, 0, 1, 0, 0, 1, 1, 1;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 1.0, 1e-10);
    EXPECT_NEAR(result(1), 2.0, 1e-10);
    EXPECT_NEAR(result(2), 3.0, 1e-10);
    EXPECT_NEAR(result(3), 4.0, 1e-10);
}

// ============================================================================
// Extrapolation Tests
// ============================================================================

TEST(InterpnTests, ExtrapolationAllDirections2D) {
    // Test extrapolation (clamping) in all directions
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // z = x + y
    metis::NumericVector values(4);
    values << 0, 1, 1, 2; // (0,0)=0, (1,0)=1, (0,1)=1, (1,1)=2

    // Query outside in all directions (will be clamped)
    metis::NumericMatrix xi(8, 2);
    xi << -1.0, 0.5, // left of grid
        2.0, 0.5,    // right of grid
        0.5, -1.0,   // below grid
        0.5, 2.0,    // above grid
        -1.0, -1.0,  // bottom-left corner
        2.0, -1.0,   // bottom-right corner
        -1.0, 2.0,   // top-left corner
        2.0, 2.0;    // top-right corner

    auto result = metis::interpn<double>(points, values, xi);

    // All should clamp to boundary values
    EXPECT_NEAR(result(0), 0.5, 1e-10); // clamps to (0, 0.5)
    EXPECT_NEAR(result(1), 1.5, 1e-10); // clamps to (1, 0.5)
    EXPECT_NEAR(result(2), 0.5, 1e-10); // clamps to (0.5, 0)
    EXPECT_NEAR(result(3), 1.5, 1e-10); // clamps to (0.5, 1)
    EXPECT_NEAR(result(4), 0.0, 1e-10); // clamps to (0, 0)
    EXPECT_NEAR(result(5), 1.0, 1e-10); // clamps to (1, 0)
    EXPECT_NEAR(result(6), 1.0, 1e-10); // clamps to (0, 1)
    EXPECT_NEAR(result(7), 2.0, 1e-10); // clamps to (1, 1)
}

TEST(InterpnTests, FillValueAllDirections2D) {
    // Test fill_value in all out-of-bounds directions
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};
    metis::NumericVector values(4);
    values << 0, 1, 1, 2;

    // Mix of in-bounds and out-of-bounds
    metis::NumericMatrix xi(5, 2);
    xi << 0.5, 0.5, // in bounds
        -0.5, 0.5,  // out left
        1.5, 0.5,   // out right
        0.5, -0.5,  // out bottom
        0.5, 1.5;   // out top

    double fill = -999.0;
    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Linear,
                                         std::optional<double>(fill));

    EXPECT_NEAR(result(0), 1.0, 1e-10);  // in bounds
    EXPECT_NEAR(result(1), fill, 1e-10); // out of bounds
    EXPECT_NEAR(result(2), fill, 1e-10); // out of bounds
    EXPECT_NEAR(result(3), fill, 1e-10); // out of bounds
    EXPECT_NEAR(result(4), fill, 1e-10); // out of bounds
}

// ============================================================================
// Non-Uniform Grid Tests
// ============================================================================

TEST(InterpnTests, NonUniformGrid2D) {
    // Non-uniformly spaced grid
    metis::NumericVector x_pts(4);
    x_pts << 0.0, 0.1, 0.5, 1.0; // Clustered near 0
    metis::NumericVector y_pts(3);
    y_pts << 0.0, 0.8, 1.0; // Clustered near 1

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // z = x * y
    metis::NumericVector values(12);
    int idx = 0;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 4; ++i) {
            values(idx++) = x_pts(i) * y_pts(j);
        }
    }

    // Query at various points
    metis::NumericMatrix xi(3, 2);
    xi << 0.05, 0.4, // in first x-cell
        0.75, 0.9,   // in last x-cell, second y-cell
        0.25, 0.5;   // between cells

    auto result = metis::interpn<double>(points, values, xi);

    // Expected: linear interpolation of x*y
    EXPECT_NEAR(result(0), 0.05 * 0.4, 0.05); // Approximate
    EXPECT_NEAR(result(1), 0.75 * 0.9, 0.05);
    EXPECT_NEAR(result(2), 0.25 * 0.5, 0.05);
}

TEST(InterpnTests, NonUniformGrid3D) {
    // Non-uniform 3D grid
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 0.2, 1.0;
    metis::NumericVector y_pts(3);
    y_pts << 0.0, 0.5, 1.0;
    metis::NumericVector z_pts(2);
    z_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts, z_pts};

    // z = x + y + z_coord
    metis::NumericVector values(18); // 3*3*2 = 18
    int idx = 0;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                values(idx++) = x_pts(i) + y_pts(j) + z_pts(k);
            }
        }
    }

    // Query
    metis::NumericMatrix xi(1, 3);
    xi << 0.1, 0.25, 0.5;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 0.1 + 0.25 + 0.5, 0.05);
}

// ============================================================================
// High-Dimensional Edge Cases
// ============================================================================

TEST(InterpnTests, HighDimEdgeQuery5D) {
    // Query at edge of 5D hypercube
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points(5, pts);

    // f = sum of coordinates
    metis::NumericVector values(32);
    for (int i = 0; i < 32; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 5; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query at edge: (0.5, 0.5, 0.5, 0.5, 0) - last dim at boundary
    metis::NumericMatrix xi(1, 5);
    xi << 0.5, 0.5, 0.5, 0.5, 0.0;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 2.0, 1e-10); // 0.5*4 + 0 = 2.0
}

TEST(InterpnTests, HighDimCornerQuery6D) {
    // Query at corner of 6D hypercube
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points(6, pts);

    // f = sum of coordinates
    metis::NumericVector values(64);
    for (int i = 0; i < 64; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 6; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query at all-ones corner
    metis::NumericMatrix xi(1, 6);
    xi << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 6.0, 1e-10); // sum of all 1s
}

TEST(InterpnTests, HighDimExtrapolation4D) {
    // Test extrapolation in 4D
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points(4, pts);

    // f = x1 + x2 + x3 + x4
    metis::NumericVector values(16);
    for (int i = 0; i < 16; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 4; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query outside grid (should clamp)
    metis::NumericMatrix xi(1, 4);
    xi << 2.0, 2.0, 2.0, 2.0; // All out of bounds

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 4.0, 1e-10); // clamps to (1,1,1,1)
}

// ============================================================================
// Multiple Query Points Batch Test
// ============================================================================

TEST(InterpnTests, BatchQuery100Points3D) {
    // Test with many query points
    metis::NumericVector pts(3);
    pts << 0.0, 0.5, 1.0;

    std::vector<metis::NumericVector> points(3, pts);

    // f = x + y + z
    metis::NumericVector values(27); // 3^3
    int idx = 0;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                values(idx++) = pts(i) + pts(j) + pts(k);
            }
        }
    }

    // 100 random-ish query points
    metis::NumericMatrix xi(100, 3);
    for (int i = 0; i < 100; ++i) {
        double t = static_cast<double>(i) / 99.0;
        xi(i, 0) = t;
        xi(i, 1) = t * 0.8;
        xi(i, 2) = t * 0.6;
    }

    auto result = metis::interpn<double>(points, values, xi);

    // Verify a subset
    for (int i = 0; i < 100; i += 10) {
        double expected = xi(i, 0) + xi(i, 1) + xi(i, 2);
        EXPECT_NEAR(result(i), expected, 1e-10) << "Mismatch at query " << i;
    }
}

TEST(InterpnTests, TransposedXiInput) {
    // Test that transposed xi input works (n_dims x n_points instead of n_points x n_dims)
    metis::NumericVector x_pts(2);
    x_pts << 0.0, 1.0;
    metis::NumericVector y_pts(2);
    y_pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};
    metis::NumericVector values(4);
    values << 0, 1, 1, 2; // z = x + y

    // xi as (n_dims, n_points) = (2, 3)
    metis::NumericMatrix xi(2, 3);
    xi << 0.5, 0.0, 1.0, // x values
        0.5, 0.0, 1.0;   // y values

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 1.0, 1e-10); // (0.5, 0.5)
    EXPECT_NEAR(result(1), 0.0, 1e-10); // (0, 0)
    EXPECT_NEAR(result(2), 2.0, 1e-10); // (1, 1)
}

// ============================================================================
// Symbolic High-Dimensional Tests
// ============================================================================

TEST(InterpnTests, Symbolic3DLinear) {
    // 3D symbolic interpolation
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points(3, pts);

    // f = x + y + z
    metis::NumericVector values(8);
    values << 0, 1, 1, 2, 1, 2, 2, 3; // Fortran order

    // Symbolic query at fixed point
    Eigen::Matrix<metis::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 3);
    xi(0, 0) = casadi::MX(0.5);
    xi(0, 1) = casadi::MX(0.5);
    xi(0, 2) = casadi::MX(0.5);

    auto result = metis::interpn<metis::SymbolicScalar>(points, values, xi);

    // Should give 1.5
    EXPECT_NEAR(eval_scalar(result(0)), 1.5, 1e-9);
}

TEST(InterpnTests, Symbolic4DLinear) {
    // 4D symbolic interpolation
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points(4, pts);

    // f = sum of coordinates
    metis::NumericVector values(16);
    for (int i = 0; i < 16; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 4; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Symbolic query
    Eigen::Matrix<metis::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 4);
    xi(0, 0) = casadi::MX(0.25);
    xi(0, 1) = casadi::MX(0.25);
    xi(0, 2) = casadi::MX(0.25);
    xi(0, 3) = casadi::MX(0.25);

    auto result = metis::interpn<metis::SymbolicScalar>(points, values, xi);

    // Should give 1.0 (0.25 * 4)
    EXPECT_NEAR(eval_scalar(result(0)), 1.0, 1e-9);
}

TEST(InterpnTests, Symbolic5DCorner) {
    // 5D symbolic at corner
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points(5, pts);

    // f = sum of coordinates
    metis::NumericVector values(32);
    for (int i = 0; i < 32; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 5; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query at (1,1,1,1,1) corner
    Eigen::Matrix<metis::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 5);
    for (int d = 0; d < 5; ++d) {
        xi(0, d) = casadi::MX(1.0);
    }

    auto result = metis::interpn<metis::SymbolicScalar>(points, values, xi);

    EXPECT_NEAR(eval_scalar(result(0)), 5.0, 1e-9);
}

TEST(InterpnTests, SymbolicBSpline4D) {
    // 4D B-spline symbolic interpolation (requires 4+ points per dim)
    metis::NumericVector pts(4);
    pts << 0.0, 1.0, 2.0, 3.0;

    std::vector<metis::NumericVector> points(4, pts);

    // f = 1 (constant for easy verification)
    metis::NumericVector values(256); // 4^4 = 256
    values.setConstant(1.0);

    // Symbolic query at center
    Eigen::Matrix<metis::SymbolicScalar, Eigen::Dynamic, Eigen::Dynamic> xi(1, 4);
    for (int d = 0; d < 4; ++d) {
        xi(0, d) = casadi::MX(1.5);
    }

    auto result = metis::interpn<metis::SymbolicScalar>(points, values, xi,
                                                        metis::InterpolationMethod::BSpline);

    EXPECT_NEAR(eval_scalar(result(0)), 1.0, 1e-9);
}

// ============================================================================
// Tests Using Metis Types (API Best Practice Demonstration)
// ============================================================================

TEST(InterpnTests, MetisTypesNumeric3D) {
    // Demonstrate using metis::NumericVector instead of metis::NumericVector
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    metis::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;
    metis::NumericVector z_pts(2);
    z_pts << 0.0, 1.0;

    // Use vector of metis::NumericVector
    std::vector<metis::NumericVector> points = {x_pts, y_pts, z_pts};

    // Values using metis::NumericVector
    metis::NumericVector values(18); // 3*3*2 = 18
    int idx = 0;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                values(idx++) = x_pts(i) + y_pts(j) + z_pts(k);
            }
        }
    }

    // Query using metis::NumericMatrix
    metis::NumericMatrix xi(2, 3);
    xi << 0.5, 0.5, 0.5, 1.0, 1.0, 0.5;

    auto result = metis::interpn<metis::NumericScalar>(points, values, xi);

    EXPECT_NEAR(result(0), 1.5, 1e-10); // 0.5 + 0.5 + 0.5
    EXPECT_NEAR(result(1), 2.5, 1e-10); // 1.0 + 1.0 + 0.5
}

TEST(InterpnTests, MetisTypesSymbolic4D) {
    // Demonstrate using metis::SymbolicMatrix for queries
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points(4, pts);

    // f = sum of coordinates
    metis::NumericVector values(16);
    for (int i = 0; i < 16; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 4; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Query using metis::SymbolicMatrix
    metis::SymbolicMatrix xi(1, 4);
    for (int d = 0; d < 4; ++d) {
        xi(0, d) = casadi::MX(0.5);
    }

    auto result = metis::interpn<metis::SymbolicScalar>(points, values, xi);

    // Should give 2.0 (0.5 * 4)
    EXPECT_NEAR(eval_scalar(result(0)), 2.0, 1e-9);
}

TEST(InterpnTests, MetisTypesTemplated) {
    // Template test using MetisVector<T> and MetisMatrix<T>
    metis::MetisVector<double> pts(3);
    pts << 0.0, 0.5, 1.0;

    std::vector<metis::MetisVector<double>> points(2, pts);

    // f = x + y
    metis::MetisVector<double> values(9);
    int idx = 0;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            values(idx++) = pts(i) + pts(j);
        }
    }

    // Query
    metis::MetisMatrix<double> xi(3, 2);
    xi << 0.25, 0.25, 0.5, 0.5, 0.75, 0.75;

    auto result = metis::interpn<double>(points, values, xi);

    EXPECT_NEAR(result(0), 0.5, 1e-10); // 0.25 + 0.25
    EXPECT_NEAR(result(1), 1.0, 1e-10); // 0.5 + 0.5
    EXPECT_NEAR(result(2), 1.5, 1e-10); // 0.75 + 0.75
}

TEST(InterpnTests, MetisTypesHighDim6D) {
    // 6D test with Metis types
    metis::NumericVector pts(2);
    pts << 0.0, 1.0;

    std::vector<metis::NumericVector> points(6, pts);

    // f = sum of coordinates
    metis::NumericVector values(64); // 2^6 = 64
    for (int i = 0; i < 64; ++i) {
        int sum = 0;
        int temp = i;
        for (int d = 0; d < 6; ++d) {
            sum += (temp & 1);
            temp >>= 1;
        }
        values(i) = sum;
    }

    // Edge query: (0.5, 0.5, 0.5, 0.5, 0.5, 0)
    metis::NumericMatrix xi(1, 6);
    xi << 0.5, 0.5, 0.5, 0.5, 0.5, 0.0;

    auto result = metis::interpn<metis::NumericScalar>(points, values, xi);

    EXPECT_NEAR(result(0), 2.5, 1e-10); // 0.5*5 + 0
}

// ============================================================================
// Hermite (C1 Catmull-Rom) Interpolation Tests
// ============================================================================

TEST(InterpnTests, Hermite1D) {
    // 1D Hermite interpolation (via 1D grid)
    metis::NumericVector x_pts(4);
    x_pts << 0.0, 1.0, 2.0, 3.0;

    std::vector<metis::NumericVector> points = {x_pts};

    // y = x^2 (quadratic function to test smoothness)
    metis::NumericVector values(4);
    values << 0.0, 1.0, 4.0, 9.0;

    // Query at interior points
    metis::NumericMatrix xi(3, 1);
    xi << 0.5, 1.5, 2.5;

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Hermite);

    // Hermite should produce smooth interpolation
    // At x=0.5, linear would give 0.5, Hermite should be closer to 0.25
    EXPECT_GT(result(0), 0.0);
    EXPECT_LT(result(0), 1.0);

    // At x=1.5, expect between 1.0 and 4.0, closer to 2.25
    EXPECT_GT(result(1), 1.0);
    EXPECT_LT(result(1), 4.0);

    // At x=2.5, expect between 4.0 and 9.0, closer to 6.25
    EXPECT_GT(result(2), 4.0);
    EXPECT_LT(result(2), 9.0);
}

TEST(InterpnTests, Hermite2DLinearFunction) {
    // 2D Hermite should exactly interpolate linear functions
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    metis::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // z = x + y (linear function)
    metis::NumericVector values(9);
    values << 0, 1, 2, 1, 2, 3, 2, 3, 4; // Fortran order

    metis::NumericMatrix xi(4, 2);
    xi << 0.5, 0.5, 1.0, 1.0, 0.25, 0.75, 1.5, 0.5;

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Hermite);

    // Linear functions should be exactly interpolated
    EXPECT_NEAR(result(0), 1.0, 1e-10); // 0.5 + 0.5
    EXPECT_NEAR(result(1), 2.0, 1e-10); // 1.0 + 1.0
    EXPECT_NEAR(result(2), 1.0, 1e-10); // 0.25 + 0.75
    EXPECT_NEAR(result(3), 2.0, 1e-10); // 1.5 + 0.5
}

TEST(InterpnTests, Hermite3DSmoothness) {
    // Test 3D Hermite interpolation
    metis::NumericVector pts(3);
    pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points(3, pts);

    // f = x + y + z
    metis::NumericVector values(27);
    int idx = 0;
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                values(idx++) = pts(i) + pts(j) + pts(k);
            }
        }
    }

    // Query at center
    metis::NumericMatrix xi(1, 3);
    xi << 1.0, 1.0, 1.0;

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Hermite);

    EXPECT_NEAR(result(0), 3.0, 1e-10); // 1+1+1
}

TEST(InterpnTests, HermiteVsLinearSmoother) {
    // Hermite should produce smoother results than linear for non-linear data
    metis::NumericVector x_pts(5);
    x_pts << 0.0, 1.0, 2.0, 3.0, 4.0;

    std::vector<metis::NumericVector> points = {x_pts};

    // Sinusoidal function (highly non-linear)
    metis::NumericVector values(5);
    values << 0.0, 0.84147, 0.9093, 0.14112, -0.7568; // sin(0), sin(1), sin(2), sin(3), sin(4)

    // Query between grid points
    metis::NumericMatrix xi(1, 1);
    xi << 1.5;

    auto result_linear =
        metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Linear);
    auto result_hermite =
        metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Hermite);

    // sin(1.5) ≈ 0.9975
    double true_val = 0.9974949866;

    // Both should be reasonable, but we mainly check they're different
    EXPECT_NE(result_linear(0), result_hermite(0));

    // Hermite may or may not be closer depending on the function
    EXPECT_GT(result_hermite(0), 0.5); // Should be positive
    EXPECT_LT(result_hermite(0), 1.5); // Should be reasonable
}

TEST(InterpnTests, HermiteHighDim4D) {
    // 4D Hermite interpolation
    metis::NumericVector pts(3);
    pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points(4, pts);

    // f = sum of coordinates
    metis::NumericVector values(81); // 3^4
    int idx = 0;
    for (int i4 = 0; i4 < 3; ++i4) {
        for (int i3 = 0; i3 < 3; ++i3) {
            for (int i2 = 0; i2 < 3; ++i2) {
                for (int i1 = 0; i1 < 3; ++i1) {
                    values(idx++) = pts(i1) + pts(i2) + pts(i3) + pts(i4);
                }
            }
        }
    }

    // Query at center
    metis::NumericMatrix xi(1, 4);
    xi << 1.0, 1.0, 1.0, 1.0;

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Hermite);

    EXPECT_NEAR(result(0), 4.0, 1e-10);
}

TEST(InterpnTests, HermiteEdgesAndCorners) {
    // Hermite at grid edges and corners
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;
    metis::NumericVector y_pts(3);
    y_pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};

    // Constant function (should interpolate exactly)
    metis::NumericVector values(9);
    values.setConstant(5.0);

    // Query at edge and corner
    metis::NumericMatrix xi(4, 2);
    xi << 0.0, 0.0, // corner
        2.0, 2.0,   // corner
        0.0, 1.0,   // edge
        1.0, 0.0;   // edge

    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Hermite);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(result(i), 5.0, 1e-10) << "Failed at query " << i;
    }
}

TEST(InterpnTests, HermiteFillValue) {
    // Hermite with fill_value for out-of-bounds
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points = {x_pts};

    metis::NumericVector values(3);
    values << 1.0, 2.0, 3.0;

    // In-bounds and out-of-bounds queries
    metis::NumericMatrix xi(3, 1);
    xi << 0.5, -1.0, 3.0;

    double fill = -999.0;
    auto result = metis::interpn<double>(points, values, xi, metis::InterpolationMethod::Hermite,
                                         std::optional<double>(fill));

    EXPECT_GT(result(0), 1.0); // In bounds
    EXPECT_LT(result(0), 2.0);
    EXPECT_NEAR(result(1), fill, 1e-10); // Out of bounds
    EXPECT_NEAR(result(2), fill, 1e-10); // Out of bounds
}

TEST(InterpnTests, HermiteSymbolicNotSupported) {
    // Hermite should throw for symbolic types
    metis::NumericVector x_pts(3);
    x_pts << 0.0, 1.0, 2.0;

    std::vector<metis::NumericVector> points = {x_pts};
    metis::NumericVector values(3);
    values << 1.0, 2.0, 3.0;

    metis::SymbolicMatrix xi(1, 1);
    xi(0, 0) = casadi::MX(0.5);

    try {
        static_cast<void>(metis::interpn<metis::SymbolicScalar>(
            points, values, xi, metis::InterpolationMethod::Hermite));
        FAIL() << "Expected InterpolationError for symbolic Hermite interpn query";
    } catch (const metis::InterpolationError &err) {
        expect_hermite_symbolic_error(err);
    }
}

// ============================================================================
// Extrapolation Configuration Tests
// ============================================================================

TEST(ExtrapConfigTests, LinearExtrap1D_LeftSide) {
    // Test linear extrapolation below x_min
    // y = x^2 sampled at [0, 1, 2, 3, 4]
    metis::NumericVector x(5), y(5);
    x << 0, 1, 2, 3, 4;
    y << 0, 1, 4, 9, 16;

    // Left slope at x=0: (y[1] - y[0]) / (x[1] - x[0]) = 1
    metis::Interpolator interp(x, y, metis::InterpolationMethod::BSpline,
                               metis::ExtrapolationConfig::linear());

    // Query at x = -2: expect y[0] + slope * (x - x_min) = 0 + 1 * (-2 - 0) = -2
    double result = interp(-2.0);
    EXPECT_NEAR(result, -2.0, 1e-10);

    // Query at x = -0.5
    double result2 = interp(-0.5);
    EXPECT_NEAR(result2, -0.5, 1e-10);
}

TEST(ExtrapConfigTests, LinearExtrap1D_RightSide) {
    // Test linear extrapolation above x_max
    // y = x^2 sampled at [0, 1, 2, 3, 4]
    metis::NumericVector x(5), y(5);
    x << 0, 1, 2, 3, 4;
    y << 0, 1, 4, 9, 16;

    // Right slope at x=4: (y[4] - y[3]) / (x[4] - x[3]) = (16-9)/(4-3) = 7
    metis::Interpolator interp(x, y, metis::InterpolationMethod::Linear,
                               metis::ExtrapolationConfig::linear());

    // Query at x = 5: expect y[4] + slope * (x - x_max) = 16 + 7 * (5 - 4) = 23
    double result = interp(5.0);
    EXPECT_NEAR(result, 23.0, 1e-10);

    // Query at x = 6: expect 16 + 7 * 2 = 30
    double result2 = interp(6.0);
    EXPECT_NEAR(result2, 30.0, 1e-10);
}

TEST(ExtrapConfigTests, LinearExtrap1D_WithBounds) {
    // Test that output bounds are applied after extrapolation
    metis::NumericVector x(3), y(3);
    x << 0, 1, 2;
    y << 10, 20, 30; // slope = 10

    // Linear extrapolation with bounds [0, 50]
    metis::Interpolator interp(x, y, metis::InterpolationMethod::Linear,
                               metis::ExtrapolationConfig::linear(0.0, 50.0));

    // Query at x = -2: expect 10 + 10*(-2-0) = -10, but clamped to 0
    double result_left = interp(-2.0);
    EXPECT_NEAR(result_left, 0.0, 1e-10);

    // Query at x = 5: expect 30 + 10*(5-2) = 60, but clamped to 50
    double result_right = interp(5.0);
    EXPECT_NEAR(result_right, 50.0, 1e-10);

    // In-bounds query should not be affected by bounds (20 is within [0,50])
    double result_mid = interp(1.0);
    EXPECT_NEAR(result_mid, 20.0, 1e-10);
}

TEST(ExtrapConfigTests, LinearExtrap1D_Symbolic) {
    // Test symbolic linear extrapolation with automatic differentiation
    metis::NumericVector x(4), y(4);
    x << 0, 1, 2, 3;
    y << 0, 2, 4, 6; // y = 2x (linear function)

    metis::Interpolator interp(x, y, metis::InterpolationMethod::Linear,
                               metis::ExtrapolationConfig::linear());

    // Symbolic query
    auto x_sym = metis::sym("x");
    auto y_sym = interp(x_sym);

    // Compute derivative
    auto dy_dx = metis::jacobian(y_sym, x_sym);

    metis::Function f("extrap_test", {x_sym}, {y_sym, dy_dx});

    // Query at x = -1 (extrapolation left)
    // Left slope = (y[1] - y[0]) / (x[1] - x[0]) = 2
    auto res = f(-1.0);
    EXPECT_NEAR(res[0](0, 0), -2.0, 1e-9); // y = 0 + 2 * (-1) = -2
    EXPECT_NEAR(res[1](0, 0), 2.0, 1e-9);  // dy/dx = slope = 2

    // Query at x = 5 (extrapolation right)
    auto res2 = f(5.0);
    EXPECT_NEAR(res2[0](0, 0), 10.0, 1e-9); // y = 6 + 2 * (5-3) = 10
    EXPECT_NEAR(res2[1](0, 0), 2.0, 1e-9);  // dy/dx = slope = 2

    // Query in-bounds (x = 1.5)
    auto res3 = f(1.5);
    EXPECT_NEAR(res3[0](0, 0), 3.0, 1e-9); // y = 2 * 1.5 = 3
}

TEST(ExtrapConfigTests, BackwardsCompatibility_DefaultClamp) {
    // Verify that default constructor still uses clamping
    metis::NumericVector x(3), y(3);
    x << 0, 1, 2;
    y << 0, 10, 0;

    metis::Interpolator interp(x, y); // No extrapolation config = clamp

    // Query outside bounds should clamp
    double result_left = interp(-1.0); // Should clamp to x=0, y=0
    double result_right = interp(3.0); // Should clamp to x=2, y=0

    EXPECT_NEAR(result_left, 0.0, 1e-10);
    EXPECT_NEAR(result_right, 0.0, 1e-10);
}

TEST(ExtrapConfigTests, ExplicitClamp) {
    // Test explicit ExtrapolationConfig::clamp()
    metis::NumericVector x(3), y(3);
    x << 0, 1, 2;
    y << 5, 10, 15;

    metis::Interpolator interp(x, y, metis::InterpolationMethod::Linear,
                               metis::ExtrapolationConfig::clamp());

    // Query outside bounds should clamp
    double result_left = interp(-1.0); // Clamps to x=0, y=5
    double result_right = interp(3.0); // Clamps to x=2, y=15

    EXPECT_NEAR(result_left, 5.0, 1e-10);
    EXPECT_NEAR(result_right, 15.0, 1e-10);
}

TEST(ExtrapConfigTests, LinearExtrap2D_SingleDimOutOfBounds) {
    // Test 2D extrapolation when only one dimension is out of bounds
    // z = x + y grid
    metis::NumericVector x_pts(3), y_pts(3);
    x_pts << 0, 1, 2;
    y_pts << 0, 1, 2;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};
    // z = x + y in Fortran order
    metis::NumericVector values(9);
    values << 0, 1, 2, 1, 2, 3, 2, 3, 4; // z[i,j] = x[i] + y[j]

    metis::Interpolator interp(points, values, metis::InterpolationMethod::Linear,
                               metis::ExtrapolationConfig::linear());

    // Query with x out of bounds, y in bounds
    metis::NumericVector query1(2);
    query1 << 3.0, 1.0; // x=3 (out), y=1 (in)
    double result1 = interp(query1);
    // Clamped interp at (2, 1) = 3, plus slope_x * (3-2) = 1 * 1 = 1
    // Expected: 3 + 1 = 4
    EXPECT_NEAR(result1, 4.0, 0.1); // Allow some tolerance for FD slopes

    // Query with x in bounds, y out of bounds
    metis::NumericVector query2(2);
    query2 << 1.0, 3.0; // x=1 (in), y=3 (out)
    double result2 = interp(query2);
    // Clamped interp at (1, 2) = 3, plus slope_y * (3-2) = 1 * 1 = 1
    // Expected: 3 + 1 = 4
    EXPECT_NEAR(result2, 4.0, 0.1);
}

TEST(ExtrapConfigTests, LinearExtrap2D_BothDimsOutOfBounds) {
    // Test 2D extrapolation at a corner (both dims out of bounds)
    metis::NumericVector x_pts(3), y_pts(3);
    x_pts << 0, 1, 2;
    y_pts << 0, 1, 2;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};
    // z = x + y
    metis::NumericVector values(9);
    values << 0, 1, 2, 1, 2, 3, 2, 3, 4;

    metis::Interpolator interp(points, values, metis::InterpolationMethod::Linear,
                               metis::ExtrapolationConfig::linear());

    // Query at corner: x=3, y=3 (both out)
    metis::NumericVector query(2);
    query << 3.0, 3.0;
    double result = interp(query);
    // Clamped at (2,2) = 4, plus slope_x * 1 + slope_y * 1 = 1 + 1 = 2
    // Expected: 4 + 2 = 6
    EXPECT_NEAR(result, 6.0, 0.2);
}

TEST(ExtrapConfigTests, LinearExtrap2D_WithBounds) {
    // Test 2D extrapolation with output bounds
    metis::NumericVector x_pts(3), y_pts(3);
    x_pts << 0, 1, 2;
    y_pts << 0, 1, 2;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};
    metis::NumericVector values(9);
    values << 0, 1, 2, 1, 2, 3, 2, 3, 4;

    // Extrapolate with bounds [0, 5]
    metis::Interpolator interp(points, values, metis::InterpolationMethod::Linear,
                               metis::ExtrapolationConfig::linear(0.0, 5.0));

    // Query at corner that would exceed bounds
    metis::NumericVector query(2);
    query << 4.0, 4.0; // Would extrapolate to ~8, but clamped to 5
    double result = interp(query);
    EXPECT_NEAR(result, 5.0, 0.1); // Should be clamped to upper bound
}

TEST(ExtrapConfigTests, LinearExtrap2D_Symbolic) {
    // Test 2D symbolic extrapolation with AD
    metis::NumericVector x_pts(3), y_pts(3);
    x_pts << 0, 1, 2;
    y_pts << 0, 1, 2;

    std::vector<metis::NumericVector> points = {x_pts, y_pts};
    metis::NumericVector values(9);
    values << 0, 1, 2, 1, 2, 3, 2, 3, 4; // z = x + y

    metis::Interpolator interp(points, values, metis::InterpolationMethod::Linear,
                               metis::ExtrapolationConfig::linear());

    // Symbolic query
    auto x_sym = metis::sym("x");
    auto y_sym = metis::sym("y");
    metis::SymbolicVector query_sym(2);
    query_sym << x_sym, y_sym;

    auto z_sym = interp(query_sym);

    metis::Function f("extrap2d", {x_sym, y_sym}, {z_sym});

    // Query outside bounds
    auto res = f(3.0, 3.0);
    EXPECT_NEAR(res[0](0, 0), 6.0, 0.2); // Same as numeric test
}
