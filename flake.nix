{
  description = "Metis: Traceable C++ Framework";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      treefmt-nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        stdenv = pkgs.llvmPackages_latest.stdenv;

        # Treefmt configuration
        treefmtEval = treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true;
          programs.clang-format.enable = true;
          # cmake-format might rename files so be careful, but generally safe to enable if tool exists
          programs.cmake-format.enable = true;
        };
      in
      {
        packages.default = stdenv.mkDerivation {
          pname = "metis";
          version = "2.0.0";
          src = ./.;

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
          ];

          buildInputs = [
            pkgs.eigen
            pkgs.casadi
          ];

          cmakeFlags = [
            "-DENABLE_COVERAGE=OFF"
            "-DBUILD_TESTING=OFF"
            "-DBUILD_EXAMPLES=OFF"
          ];
        };

        devShells.default = pkgs.mkShell.override { inherit stdenv; } {
          packages =
            with pkgs;
            [
              cmake
              ninja
              pkg-config
              eigen
              casadi
              gtest
              ccache
              clang-tools
              doxygen
              graphviz
              lcov
              llvmPackages_latest.llvm
            ]
            ++ [
              treefmtEval.config.build.wrapper
            ];

          shellHook = ''
            export CMAKE_PREFIX_PATH=${pkgs.eigen}:${pkgs.casadi}:${pkgs.gtest}
          '';
        };

        formatter = treefmtEval.config.build.wrapper;

        checks = {
          formatting = treefmtEval.config.build.check self;
        };
      }
    );
}
