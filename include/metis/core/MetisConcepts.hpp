/// @file MetisConcepts.hpp
/// @brief C++20 concepts constraining valid Metis scalar types
#pragma once
#include <casadi/casadi.hpp>
#include <concepts>

namespace metis {
/**
 * @brief Concept for valid Metis scalars
 * @tparam T Type to check (must be floating-point or casadi::MX)
 */
template <typename T>
concept MetisScalar = std::floating_point<T> || std::same_as<T, casadi::MX>;
} // namespace metis
