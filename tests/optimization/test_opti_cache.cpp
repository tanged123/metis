
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Arithmetic.hpp>
#include <metis/optimization/Opti.hpp>
#include <metis/optimization/OptiSol.hpp>

class OptiCacheTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Ensure clean state
        std::remove(filename.c_str());
    }

    void TearDown() override {
        // Clean up
        std::remove(filename.c_str());
    }

    std::string filename = "test_opti_cache.json";
};

TEST_F(OptiCacheTest, SaveAndLoad) {
    metis::Opti opti;

    auto x = opti.variable(1.0);
    auto y = opti.variable(2.0);

    // Minimize (x-1)^2 + (y-2)^2 -> x=1, y=2
    opti.minimize(metis::pow(x - 1, 2) + metis::pow(y - 2, 2));

    auto sol = opti.solve({.verbose = false});

    // Save
    std::map<std::string, metis::SymbolicScalar> vars;
    vars["x"] = x;
    vars["y"] = y;
    sol.save(filename, vars);

    // Verify file exists
    std::ifstream f(filename);
    ASSERT_TRUE(f.good());
    f.close();

    // Load
    auto data = metis::OptiSol::load(filename);

    ASSERT_EQ(data.count("x"), 1);
    ASSERT_EQ(data.count("y"), 1);

    EXPECT_NEAR(data["x"][0], 1.0, 1e-6);
    EXPECT_NEAR(data["y"][0], 2.0, 1e-6);
}

TEST_F(OptiCacheTest, VectorVariable) {
    metis::Opti opti;
    int N = 5;
    auto v = opti.variable(N, 0.0);

    // Minimize (v[i] - i)^2
    metis::SymbolicScalar obj = 0;
    for (int i = 0; i < N; ++i) {
        obj = obj + metis::pow(v(i) - i, 2);
    }
    opti.minimize(obj);

    auto sol = opti.solve({.verbose = false});

    // Save
    std::map<std::string, metis::SymbolicVector> vars;
    vars["v"] = v;
    sol.save(filename, vars);

    // Verify file exists
    std::ifstream f(filename);
    ASSERT_TRUE(f.good());
    f.close();

    // Load
    auto data = metis::OptiSol::load(filename);

    ASSERT_EQ(data.count("v"), 1);
    const auto &vec = data["v"];
    ASSERT_EQ(vec.size(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(vec[i], static_cast<double>(i), 1e-5);
    }
}

TEST_F(OptiCacheTest, WarmStartConvergence) {
    // 1. Solve 'cold' problem to get baseline and solution
    metis::Opti opti_cold;
    auto x = opti_cold.variable(0.0); // Bad initial guess
    auto y = opti_cold.variable(0.0);
    // Rosenbrock: (1-x)^2 + 100(y-x^2)^2 with optimum at (1,1)
    opti_cold.minimize(metis::pow(1 - x, 2) + 100 * metis::pow(y - metis::pow(x, 2), 2));
    auto sol_cold = opti_cold.solve({.verbose = false});

    int iter_cold = sol_cold.num_iterations().value_or(-1);

    // Save solution
    std::map<std::string, metis::SymbolicScalar> vars;
    vars["x"] = x;
    vars["y"] = y;
    sol_cold.save(filename, vars);

    // 2. Solve 'warm' problem loading from cache
    auto data = metis::OptiSol::load(filename);

    metis::Opti opti_warm;
    // Initialize with loaded values
    double init_x = data.count("x") ? data["x"][0] : 0.0;
    double init_y = data.count("y") ? data["y"][0] : 0.0;

    auto x2 = opti_warm.variable(init_x);
    auto y2 = opti_warm.variable(init_y);

    opti_warm.minimize(metis::pow(1 - x2, 2) + 100 * metis::pow(y2 - metis::pow(x2, 2), 2));
    auto sol_warm = opti_warm.solve({.verbose = false});

    int iter_warm = sol_warm.num_iterations().value_or(-1);

    // Verify warm start was faster (should be 0 or very few iterations)
    EXPECT_LT(iter_warm, iter_cold);
    EXPECT_LT(iter_warm, 5); // Should be very fast
}
