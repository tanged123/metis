/// @file test_using_header.cpp
/// @brief Compile-guard for <metis/using.hpp>.
///
/// The convenience header pulls metis symbols into the global namespace via
/// using-declarations. Nothing else in the build includes it, so a stale
/// using-declaration (referencing a symbol that no longer exists in metis::)
/// would otherwise go unnoticed until a downstream user hit it. This test
/// exists purely to force the header through the compiler and exercise a few
/// of the re-exported symbols.

#include <metis/using.hpp>

#include <gtest/gtest.h>

TEST(UsingHeader, MathSymbolsResolveInGlobalNamespace) {
    // Symbolic path: confirm the re-exported math/control-flow helpers bind.
    auto x = sym("x");
    auto expr = sin(x) + pow(x, 2.0) + where(x > 0.0, x, -x);
    EXPECT_FALSE(expr.is_empty());
}

TEST(UsingHeader, ConversionHelpersResolve) {
    auto v = sym_vector("v", 3);
    auto mx = as_mx(v);
    EXPECT_EQ(mx.size1(), 3);

    NumericVector n(2);
    n << 1.0, 2.0;
    auto eigen_back = to_eigen(to_mx(n));
    EXPECT_EQ(eigen_back.rows(), 2);
}
