#include "metis/core/MetisConcepts.hpp"
#include "metis/core/MetisTypes.hpp"
#include <gtest/gtest.h>

// Generic test logic
template <typename Scalar> void test_scalar_properties() {
    Scalar a = 5.0;
    Scalar b = 2.0;
    // In symbolic mode, this builds a graph. In numeric, it computes.
    auto c = a + b;

    // Simple check to ensure c is of the correct type and constructible
    static_assert(metis::MetisScalar<Scalar>, "Must be a MetisScalar");
}

TEST(CoreTests, NumericMode) {
    test_scalar_properties<metis::NumericScalar>();

    // Value check for numeric
    metis::NumericScalar a = 5.0;
    metis::NumericScalar b = 2.0;
    EXPECT_DOUBLE_EQ(a + b, 7.0);
}

TEST(CoreTests, SymbolicMode) {
    test_scalar_properties<metis::SymbolicScalar>();

    // Structural check for symbolic
    metis::SymbolicScalar a = 5.0;
    metis::SymbolicScalar b = 2.0;
    metis::SymbolicScalar c = a + b;
    // We can't easily check the value without a Function, but we can check it instantiated
    EXPECT_FALSE(c.is_empty());
}
