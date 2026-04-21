# Metis Refinement Audit & Refactoring Plan

## Codebase Profile

| Metric | Value |
|--------|-------|
| **Headers** | 44 files, ~16,130 LOC |
| **Tests** | 35 files across 3 executables |
| **Examples** | 31 files across 5 directories |
| **Modules** | core/ (8), math/ (22), optimization/ (11), utils/ (1), root (2) |
| **Largest files** | MetisIO.hpp (1507), Interpolate.hpp (1277), Integrate.hpp (1147), Opti.hpp (1099), Sparsity.hpp (1083) |

---

# Phase 1: Audit Findings

## CRITICAL — Architectural Issues

### C1. Naming Convention Chaos (All Modules)

Three incompatible conventions coexist with no discernible rule:

| Convention | Examples | Frequency |
|-----------|----------|-----------|
| **snake_case** | `pce_mean`, `smolyak_sparse_grid`, `solve_sweep`, `subject_to_lower` | ~60% |
| **camelCase/lowercase** | `jacobian`, `cumtrapz`, `linspace`, `trapz` | ~30% |
| **mixed** | `sym_gradient`, `from_euler`, `to_rotation_matrix` | ~10% |

Same file can mix both: `Calculus.hpp` has `gradient`, `diff`, `trapz`, `cumtrapz`, `gradient_periodic` — no pattern.

**Enum values** also clash: `Solver::IPOPT` (ALL_CAPS) vs `CollocationScheme::Trapezoidal` (PascalCase) vs `ScalingIssueLevel::Warning` (PascalCase).

### C2. Detail Namespace Inconsistency (All Modules)

Two conventions, no rule:

- **Generic `detail`**: Sparsity, MetisIO, Function, IntegrateDiscrete, Interpolate, RootFinding, AutoDiff, ScatteredInterpolator, OrthogonalPolynomials, Linalg, SurrogateModel
- **Prefixed `<module>_detail`**: structural_detail, diagnostics_detail, polynomial_chaos_detail, integrate_detail, opti_detail, quadrature_detail, autodiff_detail, logic_detail

AutoDiff.hpp uses **both** (`detail` and `autodiff_detail`) in the same file.

### C3. MetisIO.hpp — 1507-Line Monolith with 750+ Lines of Duplicated HTML/JS

Four near-identical export function families with copy-pasted HTML/CSS/JavaScript:
- `export_graph_dot` / `export_graph_html` (MX graphs)
- `export_sx_graph_dot` / `export_sx_graph_html` (SX deep graphs)
- spy plot HTML generation in `Sparsity.hpp`

Each HTML export contains ~200 lines of inline JavaScript for pan/zoom/hover. Styling changes require editing in 3+ places.

### C4. Opti.hpp — 1099-Line Class with 40+ Public Methods

Responsibilities crammed into one file:
- Variable creation (4 overloads, 150 lines of near-identical scaling/bounds code)
- Constraint management (8 `subject_to` variants + 6 convenience wrappers)
- Objective setting (4 variants)
- Solve + parametric sweep
- Derivative constraint application (3 methods)
- Scaling analysis (150 lines)
- 85 lines of `opti_detail` helpers used only by `analyze_scaling()`

### C5. TranscriptionBase CRTP — Broken Polymorphism

Each of the 4 derived classes (Pseudospectral, DirectCollocation, BirkhoffPseudospectral, MultipleShooting) implements its own `setup()` with a **different signature** and **different options struct**. The CRTP base can't enforce a common interface. Lifecycle enforced via runtime flags (`setup_complete_`, `dynamics_set_`) instead of the type system. ~400 lines of duplicated setup boilerplate across the 4 files.

### C6. solve_sweep() Silent Data Loss (Opti.hpp:683-686)

```cpp
} catch (const std::exception &) {
    result.all_converged = false;
    break;  // Returns partial results, no error info, no indication which point failed
}
```

### C7. OptiSweep::objective() Returns Hardcoded 0.0

```cpp
double objective(size_t i) const {
    // ...
    return 0.0;  // Placeholder
}
```

Non-functional public method. Users must manually re-evaluate their objective.

### C8. Incomplete Symbolic Implementations

- **Quaternion.hpp:231-243**: `from_rotation_matrix()` symbolic branch only handles trace > 0 case. Comment says "WARNING: This will fail for large rotations (180 deg)."
- **Arithmetic.hpp:367-375**: `fmod()` matrix variant for CasADi may recurse infinitely.
- **SurrogateModel.hpp:147-150**: `softplus_scalar()` assumes `SymbolicScalar::logsumexp()` API exists without validation.

---

## MAJOR — Redundancy, Complexity, Style Drift

### M1. Gradient Function Triplication (Calculus.hpp)

`gradient_1d()` (24 lines), `gradient()` (98 lines), `gradient_periodic()` (72 lines) share ~70% code. All three reimplement spacing extraction with identical logic differing only in variable names.

### M2. Documentation Noise — AI-Generated Boilerplate

| File | Lines | Doxygen annotations | Ratio |
|------|-------|-------------------|----|
| Arithmetic.hpp | 660 | 206 | 31% |
| Trig.hpp | 243 | 77 | 32% |
| IntegratorStep.hpp | 300 | 61 | 20% |
| Logic.hpp | 509 | 117 | 23% |

Most annotations restate the function signature: `@brief Compute the sine of x` on `sin(x)`. Total: 1453 doxygen annotations across 43 files.

Leave doxygen in as  needed for documentation downstream for users. 

### M3. Dead/Deprecated Code Still Present

- **DiffOps.hpp**: 16-line deprecated redirect to AutoDiff.hpp + Calculus.hpp. Commented out in MetisMath.hpp but file still exists.
- **`Interp1D`** and **`MetisInterpolator`**: Backward-compat aliases for `Interpolator` at Interpolate.hpp:1101-1106.
- **`clip()`**: Pointless forwarding alias for `clamp()` in Logic.hpp:552.
- **OptiCache**: Single-method class that forwards to `metis::utils::read_json()`. Could be a static method on OptiSol.

### M4. Include Path Inconsistency

FiniteDifference.hpp uses relative paths (`../core/MetisError.hpp`) while all other 43 files use absolute paths (`metis/core/MetisError.hpp`).

### M5. Magic Numbers Without Justification

| File | Value | Context |
|------|-------|---------|
| ScatteredInterpolator.hpp:327 | `1e-10` | RBF regularization (not configurable) |
| IntegrateDiscrete.hpp:229,409 | `1e-100` | RMS fusion term |
| OrthogonalPolynomials.hpp:25 | `1e-12` | Legendre root tolerance |
| Trig.hpp:46 | `1e-15` | ThinPlateSpline zero threshold |

### M6. Validation Pattern Explosion (~50-100 Lines)

Identical patterns repeated across files:
```cpp
// Appears 5+ times in Spacing.hpp alone:
if (n < 1) throw InvalidArgument("funcname: n must be >= 1");

// Appears in Calculus, FiniteDifference, Logic, etc.:
if (y.size() != x.size()) throw InvalidArgument("funcname: size mismatch");
```

### M7. Variable Creation Overload Explosion (Opti.hpp:191-337)

4 `variable()` overloads with copy-pasted scaling/bounds logic. Changes to scaling require editing in 4 places.

### M8. Redundant Constraint Helpers (Opti.hpp:456-528)

6 trivial wrappers (`subject_to_lower`, `subject_to_upper`, `subject_to_bounds` x scalar + vector) that just call `opti_.subject_to(x >= bound)`.

### M9. OptiSol Silent Fallbacks

`num_function_evals()` and `num_iterations()` return `-1` instead of throwing or returning `optional<int>`. Users can't distinguish "stat not recorded" from "error."

### M10. Transcription Options Inconsistency

4 separate options structs with no common base:
- `PseudospectralOptions` (full name)
- `CollocationOptions` (full name)
- **`BirkhoffOptions`** (abbreviated — should be `BirkhoffPseudospectralOptions`)
- `MultiShootingOptions` (full name, but uses `n_intervals` instead of `n_nodes`)

### M11. CasADi Internals Leaked

`Opti::casadi_opti()` returns mutable reference to internal `casadi::Opti`. Users can bypass all Metis invariants.

### M12. JsonUtils.hpp — Naive Parser in Public API

130-line hand-rolled JSON parser using string operations. Silent error catching at line 117-119. Should either be replaced with a proper library or removed from the public API (only used by OptiCache/OptiSol warm-start).

---

## MINOR — Comment Noise, Small Naming, Gitignore

### m1. .cursorrules Stale Information

References `include/metis/linalg/` which doesn't exist (linear algebra is in `math/Linalg.hpp`).

### m2. MetisMath.hpp Redundant Include

Includes `metis/core/MetisError.hpp` directly, but it's already transitively included through every math header.

### m3. CasADi Version Not Pinned in CMakeLists.txt

```cmake
find_package(casadi REQUIRED)  # No version specified
```

### m4. Example Naming Inconsistency

Mix of `*_intro.cpp`, `*_demo.cpp`, `*_example.cpp`, and bare names. No consistent convention.

### m5. Root Directory Artifacts

32 generated files (`.dot`, `.html`, `.pdf`, `.json`) exist on disk from example runs. The `.gitignore` already excludes them (lines 61-65), so they're not tracked — this is cosmetic.

### m6. No Examples README

31 examples across 5 directories with no index, no difficulty guide, no feature mapping.

### m7. Getter/Accessor Naming Inconsistency

Mix of `states()` (bare noun), `get_category()` (get_ prefix), `diff_matrix()` (bare compound). No pattern.

---

## Test & Example Coverage Assessment

**Strengths**: Tests are well-organized, mirror source structure, use dual-mode (numeric + symbolic) testing. Complex physics problems (brachistochrone, harmonic oscillators) verified against analytical solutions. Coverage is generally good.

**Gaps**:
- No direct tests for MetisError.hpp, MetisConcepts.hpp, TranscriptionBase.hpp (only indirect through derived classes)
- Convergence studies minimal (only 1 in entire suite)
- No high-scale tests (>100 equations for structural analysis)
- JsonUtils error paths untested

**Examples**: Excellent physics fidelity, demonstrate real downstream patterns. All 31 use `#include <metis/metis.hpp>` — no direct CasADi/Eigen exposure. The top 3 features by example usage: symbolic computation (25/31), autodiff (22/31), optimization (8/31).

---

# Phase 2: Unified Standard

## Naming

| Entity | Convention | Examples |
|--------|-----------|----------|
| **Types/Classes/Structs** | PascalCase | `SparsityPattern`, `OptiSol`, `Function` |
| **Functions (free + member)** | snake_case | `solve_ivp`, `sparse_jacobian`, `analyze_scaling` |
| **Math primitives** | lowercase (matching std) | `sin`, `cos`, `sqrt`, `exp`, `pow`, `abs` |
| **Constants/Enum values** | PascalCase | `Solver::Ipopt`, `CollocationScheme::Trapezoidal` |
| **Template parameters** | PascalCase | `Scalar`, `Derived`, `Func` |
| **Files** | PascalCase.hpp | `Interpolate.hpp`, `RootFinding.hpp` (keep current) |
| **Detail namespaces** | `detail` (plain) | `namespace detail { }` — always nested inside `metis` |

**Rationale**: snake_case for functions matches C++ numerics tradition (Eigen, Boost.Math). Math primitives stay lowercase to match `std::sin`. PascalCase for types is universally expected in C++. Plain `detail` is simpler and sufficient (no collision risk inside `metis` namespace).

## Error Philosophy

- **Exceptions only** — `throw metis::InvalidArgument(...)` for precondition violations, `throw metis::RuntimeError(...)` for solver/integration failures. No silent fallbacks, no return codes, no `-1` sentinel values.
- **Validate at API boundaries** — public functions validate inputs. Internal/detail functions may skip validation for performance.
- **No asserts in release code** — asserts are for development invariants only (7 total currently, fine).

## Header Organization

```
include/metis/
  core/           <- Types, concepts, error, function wrapper, sparsity, I/O
  math/           <- Math operations, calculus, interpolation, integration, quadrature, etc.
  optimization/   <- Opti, transcriptions, solution, options, scaling
  utils/          <- Internal utilities (JsonUtils.hpp — make private or remove)
  metis.hpp       <- Uber umbrella (keep)
  using.hpp       <- Namespace convenience (keep, with warning)
```

**Layering invariant**: `math/` depends on `core/` only. `optimization/` depends on `core/` + `math/`. No reverse dependencies.

**Split/merge targets**:
- **Delete** DiffOps.hpp (deprecated redirect).
- **Delete** or inline OptiCache.hpp (trivial pass-through).
- Remove backward-compat aliases (`Interp1D`, `MetisInterpolator`, `clip`).

## Include Strategy

- **Granular**: Consumers can include specific headers (`#include <metis/math/Interpolate.hpp>`).
- **Umbrella**: `metis.hpp` pulls everything (acceptable for applications, not for libraries).
- **using.hpp** stays as explicit opt-in namespace pollution with clear warning.
- MetisMath.hpp stays as math-only umbrella.

## What Belongs in This Library vs. Consumers

**In Metis**: Dual-mode math primitives, autodiff, sparsity, optimization framework, ODE integration, interpolation, quadrature, polynomial chaos, root finding, structural analysis.

**Push to consumers**: JSON I/O (replace with proper library or make strictly internal). Example-specific CSV utilities.

## Template/constexpr Policy

- All physics/math functions templated on `MetisScalar` (this is correct).
- `constexpr` where possible for compile-time-evaluable functions (constants, small helpers).
- The `if constexpr` numeric/symbolic branching pattern is **inherent to the design** — it is the mechanism for dual-mode support. It's not a problem to fix; it's the architecture.

## Version Control

- Current `.gitignore` is adequate (already excludes `*.dot`, `*.html`, `*.pdf`, `*.json`, `logs/`, `build/`).
- `docs/examples/*.html` and `docs/images/*.png` correctly excepted.
- No changes needed.
