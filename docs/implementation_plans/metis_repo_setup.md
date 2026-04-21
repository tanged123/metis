metis-repo-setup.md
Metis Repository Setup Instructions
Project: Metis (Traceable C++ Numerical Framework)
Goal: Initialize a robust, modern C++20 repository with Nix flake management, CMake build system, and GitHub Actions CI.
Philosophy: Reproducibility first. The development environment (Nix) must match CI exactly.
1. Directory Structure Definition
Agent Action: Create the following directory hierarchy.

Plaintext


metis/
├── .github/
│   └── workflows/
│       ├── ci.yml          # Main Nix-based CI workflow
│       └── format.yml      # Clang-format check
├── cmake/                  # Custom CMake modules (if needed)
├── examples/               # User-facing demo code
│   └── basic_optimization.cpp
├── include/
│   └── metis/
│       ├── core/           # Type aliases & Concepts
│       │   ├── MetisTypes.hpp
│       │   └── MetisConcepts.hpp
│       ├── math/           # Math dispatch layer
│       │   └── MetisMath.hpp
│       ├── linalg/         # Eigen integration
│       │   └── MetisMatrix.hpp
│       └── metis.hpp       # Main include file
├── tests/                  # GoogleTest suite
│   ├── CMakeLists.txt
│   └── test_core.cpp
├── .clang-format           # Style guide (LLVM style recommended)
├── .gitignore              # Standard C++ gitignore
├── CMakeLists.txt          # Root build configuration
├── flake.nix               # Nix dependency definition
└── README.md               # Project documentation


2. Nix Flake Configuration (flake.nix)
Agent Action: Create flake.nix to lock the development environment.
Requirements:
Inputs: nixpkgs (use nixpkgs-unstable for latest C++ tools).
Packages:
casadi: The symbolic backend (ensure IPOPT support is enabled).
eigen: The numeric linear algebra backend.
cmake, ninja, pkg-config: Build tools.
clang_16 (or newer): Compiler supporting C++20 Concepts.
gtest: Testing framework.
DevShell: Must expose CC and CXX environment variables pointing to Clang.
Snippet Reference:

Nix


{
  description = "Metis: Traceable C++ Framework";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        stdenv = pkgs.clang16Stdenv; # Enforce Clang 16+
      in {
        devShells.default = pkgs.mkShell.override { inherit stdenv; } {
          packages = with pkgs; [
            cmake ninja pkg-config
            eigen
            casadi # Verify this includes IPOPT in nixpkgs
            gtest
            clang-tools # for clang-format/tidy
          ];
          
          # Force CMake to find Nix packages
          shellHook = ''
            export CMAKE_PREFIX_PATH=${pkgs.eigen}:${pkgs.casadi}:${pkgs.gtest}
          '';
        };
      }
    );
}


3. CMake Configuration (CMakeLists.txt)
Agent Action: Create the root build script.
Requirements:
Standard: Require CMAKE_CXX_STANDARD 20.
Target: Define metis as an INTERFACE library (header-only).
Dependencies: Use find_package for Eigen3 and CasADi.
Note: Nix provides these config files.
Flags: Enable -Wall -Wextra -Werror to enforce quality.
Snippet Reference:

CMake


cmake_minimum_required(VERSION 3.20)
project(metis VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(Eigen3 3.4 REQUIRED)
find_package(casadi REQUIRED)

# --- Main Library ---
add_library(metis INTERFACE)
target_include_directories(metis INTERFACE include)

# Link Dependencies
target_link_libraries(metis INTERFACE 
    Eigen3::Eigen 
    casadi
)

# --- Testing ---
enable_testing()
add_subdirectory(tests)


4. The Foundation Code (Phase 1)
Agent Action: Populate include/metis/core/ to establish the "Metis" identity.
A. MetisConcepts.hpp
Define the MetisScalar concept.

C++


#pragma once
#include <concepts>
#include <casadi/casadi.hpp>

namespace metis {
    template <typename T>
    concept MetisScalar = std::floating_point<T> || std::same_as<T, casadi::MX>;
}


B. MetisTypes.hpp
Define the backend aliases.

C++


#pragma once
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace metis {
    // Numeric Backend
    using NumericScalar = double;
    using NumericMatrix = Eigen::MatrixXd;

    // Symbolic Backend
    using SymbolicScalar = casadi::MX;
    // Note: Eigen wrapping of CasADi requires careful handling (Phase 2), 
    // for Phase 1 start with scalar types.
}


5. CI/CD (GitHub Actions)
Agent Action: Create .github/workflows/ci.yml.
Strategy: Rely entirely on Nix for the environment. This guarantees that if it works on your machine (in nix develop), it works in CI.
Snippet Reference:

YAML


name: Metis CI

on: [push, pull_request]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Nix
      uses: cachix/install-nix-action@v20
      with:
        nix_path: nixpkgs=channel:nixos-unstable

    - name: Build & Test (Flake Check)
      run: nix develop --command bash -c "mkdir build && cd build && cmake .. && make && ctest --output-on-failure"


6. Testing Strategy
Agent Action: Create tests/CMakeLists.txt and tests/test_core.cpp.
Requirements:
Use GoogleTest.
Write a "Dual-Mode" test pattern: A single test function templated on Scalar, instantiated twice (once for double, once for casadi::MX).
Example Test Pattern:

C++


// tests/test_core.cpp
#include <gtest/gtest.h>
#include "metis/core/MetisConcepts.hpp"

// Generic test logic
template <typename Scalar>
void test_scalar_properties() {
    Scalar a = 5.0;
    Scalar b = 2.0;
    // In symbolic mode, this builds a graph. In numeric, it computes.
    auto c = a + b; 
    // Verification logic differs slightly (eval vs value check)
    // ...
}

TEST(CoreTests, NumericMode) {
    test_scalar_properties<double>();
}

TEST(CoreTests, SymbolicMode) {
    test_scalar_properties<casadi::MX>();
}


7. Execution Order for Agent
Init: Initialize Git repository.
Nix: Write flake.nix and validate with nix develop (ensure shells load).
Skeleton: Create directory tree and empty files.
CMake: Write CMakeLists.txt configurations.
Code: Implement the MetisScalar concept (Phase 1).
Test: Write and pass the first dual-mode test using ctest.
CI: Commit workflows and verify green checkmark on GitHub.