# Native Type Migration Plan

## Goal
Scrub the `include/metis` headers to replace direct usages of `Eigen::` and `casadi::` types with their Metis native aliases (defined in `MetisTypes.hpp`). This improves encapsulation and consistency.

## User Review Required
> [!NOTE]
> `Eigen::MatrixBase<Derived>` and `Eigen::Index` will generally be preserved as they are fundamental to the template mechanics that Metis relies on.
> `casadi::Opti` and `casadi::OptiSol` members in optimization classes will remain as they are wrappers, but their public interfaces will be updated to return/accept Metis types where possible.

## Proposed Changes

### Core Headers (`include/metis/core/`)

#### [MODIFY] [Function.hpp](file:///home/tanged/sources/metis/include/metis/core/Function.hpp)
- Replace `Eigen::MatrixXd` with `NumericMatrix` in return types and arguments.
- Replace `Eigen::Matrix<Scalar, ...>` with `MetisMatrix<Scalar>`.
- Replace `std::vector<Eigen::MatrixXd>` with `std::vector<NumericMatrix>`.

#### [MODIFY] [MetisIO.hpp](file:///home/tanged/sources/metis/include/metis/core/MetisIO.hpp)
- Replace `Eigen::MatrixXd` in internal evaluation logic with `NumericMatrix`.
- Ensure `MetisTypes.hpp` is included.

### Math Headers (`include/metis/math/`)
Apply changes to all math headers, including `Trig.hpp`, `Rotations.hpp`, `FiniteDifference.hpp`, `DiffOps.hpp`, etc.

#### [MODIFY] *All Math Files*
- Replace `casadi::MX` with `SymbolicScalar` where it refers to the type.
- Replace `Eigen::Matrix<Scalar, ...>` with `MetisMatrix<Scalar>` or `MetisVector<Scalar>`.
- Ensure `MetisTypes.hpp` is included.

### Optimization Headers (`include/metis/optimization/`)

#### [MODIFY] [Opti.hpp](file:///home/tanged/sources/metis/include/metis/optimization/Opti.hpp)
- Replace `casadi::MX` in public API with `SymbolicScalar`.
- Replace `Eigen::Matrix` return types with `MetisMatrix` / `MetisVector`.

#### [MODIFY] [OptiSol.hpp](file:///home/tanged/sources/metis/include/metis/optimization/OptiSol.hpp)
- Replace `Eigen::MatrixXd` in `value()` return type with `NumericMatrix`.
- Replace `casadi::MX` casts with `SymbolicScalar` casts where appropriate.

## Verification Plan

### Automated Tests
- Run full CI suite: `./scripts/ci.sh`
- Run validation suite (tests + examples): `./scripts/verify.sh`
- Ensure no compilation errors.
- Verify that aliases resolve correctly and binary compatibility is maintained.

### Manual Verification
- Spot check a few headers to ensure no raw `Eigen::` or `casadi::` types leaked into the update (except where intended).
