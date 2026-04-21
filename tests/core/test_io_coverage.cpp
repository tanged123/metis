#include "../utils/TestUtils.hpp"
#include <fstream>
#include <gtest/gtest.h>
#include <metis/core/MetisError.hpp>
#include <metis/core/MetisIO.hpp>
#include <metis/core/MetisTypes.hpp>
#include <metis/math/Trig.hpp>

// ======================================================================
// MetisIO.hpp Coverage
// ======================================================================

TEST(MetisIOCoverage, ExportGraphError) {
    auto x = metis::sym("x");
    // Invalid path should throw
    // Assuming /root or /invalid is not writable
    EXPECT_THROW(metis::export_graph_dot(x, "/invalid/path/graph"), metis::RuntimeError);
}

TEST(MetisIOCoverage, VisualizeGraphError) {
    auto x = metis::sym("x");
    // Should catch exception and return false
    bool success = metis::visualize_graph(x, "/invalid/path/graph");
    EXPECT_FALSE(success);
}

TEST(MetisIOCoverage, RenderGraphFailure) {
    // If dot is missing, this returns false
    // If dot exists but file is bad, returns false
    // We just want to ensure it doesn't crash and returns false
    bool success = metis::render_graph("nonexistent.dot", "out.pdf");
    EXPECT_FALSE(success);
}

TEST(MetisIOCoverage, DetailHelpers) {
    // Test escape_dot_label with special chars
    // We can't access detail:: directly usually, but it's in header.
    // If it's private/implementation detail, we might skip.
    // But it's in metis::detail, which is accessible.

    std::string complex_label = "a\"b\\c\nd<e>{f}";
    std::string escaped = metis::detail::escape_dot_label(complex_label);

    EXPECT_NE(escaped.find("\\\""), std::string::npos);
    EXPECT_NE(escaped.find("\\\\"), std::string::npos);
    EXPECT_NE(escaped.find("\\n"), std::string::npos);
    EXPECT_NE(escaped.find("&lt;"), std::string::npos);

    // Test truncation
    std::string long_label(100, 'a');
    std::string truncated = metis::detail::escape_dot_label(long_label);
    EXPECT_TRUE(truncated.find("...") != std::string::npos);
    EXPECT_LT(truncated.size(), 100);
}

TEST(MetisIOCoverage, GetOpName) {
    // Test get_op_name for various types
    using namespace metis::detail;

    auto x = metis::sym("x");
    EXPECT_EQ(get_op_name(x), "x");

    auto c = casadi::MX(42.0);
    EXPECT_EQ(get_op_name(c), "42");

    auto expr = metis::sin(x);
    EXPECT_EQ(get_op_name(expr), "sin");

    // Binary op truncation
    // Create a deeply nested expression to force long string representation
    auto long_expr = x;
    for (int i = 0; i < 10; ++i)
        long_expr = long_expr + long_expr;

    // get_op_name truncates if > 30 chars
    std::string name = get_op_name(long_expr);
    // Just verify it runs safely
    EXPECT_FALSE(name.empty());
}
