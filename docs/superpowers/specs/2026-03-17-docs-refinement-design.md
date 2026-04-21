# Docs Refinement Design -- Metis v2.0.0

## Context

Metis v2.0.0 was just released with a massive feature PR adding polynomial chaos, quadrature, sparse derivative pipelines, sensitivity regime selection, matrix-free HVPs, multi-method root finding, implicit function builders, structural diagnostics/transforms, second-order and mass-matrix integrators, policy-driven linear solves, scaling diagnostics, and many examples/tests. Documentation needs a consistency and usability pass.

**Audience priority:** Contributors (C) > New users (A) > Upgraders (B)

## 1. Doxygen Comment Standard

Standardize comment style across all 42 headers in `include/metis/` (core: 8, math: 21, optimization: 10, utils: 1, top-level: 2).

### Format

```cpp
/// @file Interpolate.hpp
/// @brief N-dimensional interpolation (Linear, Hermite, BSpline, Nearest)

namespace metis {

/// @brief One-line description of the class/struct
///
/// Optional 1-2 sentence elaboration when the brief isn't self-explanatory.
/// @see related_function, OtherClass
struct InterpolationResult { ... };

/// @brief One-line description of what the function does
/// @tparam T Scalar type (NumericScalar or SymbolicScalar)
/// @param grid Evenly-spaced grid points
/// @param values Function values at grid points
/// @return Interpolated value at the query point
/// @see interp1d, interpnd
template <typename T>
T interp2d(const std::vector<T>& grid, const std::vector<T>& values);

/// @brief Short enum description
enum class InterpolationMethod {
    Linear,   ///< Piecewise linear
    Hermite,  ///< Cubic Hermite with derivative matching
    BSpline,  ///< B-spline (order configurable)
    Nearest   ///< Nearest-neighbor lookup
};

} // namespace metis
```

### Rules

- Every file gets `@file` + `@brief` at the top
- Every public class/struct/function gets `@brief`
- Every public function gets `@param`/`@return`/`@tparam` annotations (mandatory, not selective)
- `@see` for cross-references between related APIs
- Enum values get `///<` inline comments
- `@example` blocks are allowed and encouraged where they exist -- do not remove them
- No `@details` blocks -- keep it concise
- Both `///` and `/** */` styles are accepted; maintain consistency within each file (do not mix styles in one file)
- Existing doxygen comments that already match this style are left alone

## 2. User Guide Standard Template

Standardize structure across all 18 guides in `docs/user_guides/`.

### Template

```markdown
# [Feature Name]

One paragraph: what this feature does, when you'd use it, and whether it works
in numeric mode, symbolic mode, or both.

## Quick Start

Minimal working example (10-20 lines) that a user can copy-paste and run.

## Core API

Main functions/classes, organized by usage. Each entry:
- Signature or usage pattern
- Brief description
- Short code snippet if the signature alone isn't clear

## Usage Patterns

Common workflows and recipes. Real-world-ish examples with explanation.

## Advanced Usage

(Optional) Power-user features, performance tuning, non-obvious capabilities.

## Diagnostics & Troubleshooting

(Optional) Common errors, diagnostic output interpretation, edge cases.

## See Also

- Links to related user guides
- Links to relevant example files (examples/...)
- Links to relevant header files for API details
```

### Rules

- Every guide gets H1 -> Quick Start -> Core API -> Usage Patterns flow
- The one-paragraph intro under H1 replaces any existing "Overview" or "Introduction" sections -- do not keep both
- "See Also" section is mandatory -- minimum one link to another user guide AND one link to an example or header file
- Code examples use `metis::` namespace (not `using namespace metis`)
- Guides that don't need Advanced/Diagnostics sections omit them
- No theory/math preamble before Quick Start

## 3. README.md Rewrite

Full rewrite as contributor-first landing page.

### Structure

```
# Metis

Elevator pitch (one paragraph).

## Building & Development

Prerequisites, nix develop, build/test/coverage scripts.

## Project Structure

Annotated directory tree (include/, examples/, tests/, docs/).

## Architecture Overview

5-10 sentences: template-first, dual backend, dispatch layer,
structural vs value logic. Link to docs/design_overview.md.

## Documentation

How docs are organized, how to generate (doxygen), link to GitHub Pages.
Table listing all 18 user guides with one-line descriptions.

## Key Features

Concise grouped list: Core, Math, Optimization, Interpolation,
Integration, Stochastic, Structural Analysis.

## Examples

Category breakdown of examples/ directory, how to build/run.

## Contributing

Code style, how to add tests/examples, PR workflow.
```

### Key changes from current

- Building & Development moves to top (contributor-first)
- Project structure expanded and prominent
- Architecture section added
- Feature list condensed and grouped
- Contributing section added
- Current full usage example replaced with a condensed (10-15 line) dual-mode snippet showing the core value proposition

## 4. Design Overview & Usage Guide Refresh

### design_overview.md

- Update to reflect v2.0.0 scope (PCE, quadrature, structural diagnostics, sparsity, root finding, linear solve policies, second-order integrators, scaling, error handling)
- Keep philosophical structure (Code Transformations, Template-First, Dual Backend, Dispatch, Structural vs Value, Branching, Loops)
- Remove or update completed roadmap phases
- Add cross-references to new user guides
- Target ~150 lines max (currently 89 -- growth ceiling, not reduction target)

### metis_usage_guide.md

- Audit against v2.0.0 API -- fix stale signatures, removed types, renamed enums
- Add sections for new features not currently covered
- Cross-reference user guides rather than duplicating content
- Standardize code examples to `metis::` namespace
- Serve as a map, not a manual -- trim comprehensive API tables and replace with brief descriptions that link to the relevant user guide for details

## 5. Doxyfile Configuration

- Add `PROJECT_NUMBER = 2.0.0`
- Verify `PROJECT_BRIEF` is set
- Ensure `USE_MDFILE_AS_MAINPAGE = README.md`
- Full intended INPUT: `README.md include docs/design_overview.md docs/metis_usage_guide.md docs/user_guides docs/patterns`
- EXCLUDE_PATTERNS: `docs/implementation_plans/* docs/saved_work/* docs/design_reviews/* docs/examples/*`
- Keep `EXTRACT_ALL=YES`
- Verify `GENERATE_TREEVIEW=YES`

### Verification

- Run `doxygen Doxyfile` with `WARN_IF_UNDOCUMENTED=YES` -- confirm zero warnings for public symbols
- Diff the Doxyfile and confirm only the specified keys changed

## Verification Criteria

| Area | Done when |
|------|-----------|
| Doxygen headers | Every `.hpp` has `@file`+`@brief`; every public function has `@brief`+`@param`+`@return`; `doxygen` runs with zero undocumented warnings |
| User guides | Every guide has H1, Quick Start, Core API, Usage Patterns, See Also sections |
| README | Follows the approved structure; builds contributor-first flow |
| design_overview.md | Covers all v2.0.0 modules; <= 150 lines |
| metis_usage_guide.md | No stale API references; links to user guides instead of duplicating API tables |
| Doxyfile | Only specified keys changed; generates clean HTML |

## Scope Exclusions

- `docs/implementation_plans/` -- left as-is
- `docs/design_reviews/` -- left as-is (excluded from Doxygen output)
- `docs/saved_work/` -- left as-is (excluded from Doxygen output)
- `docs/examples/` -- left as-is (excluded from Doxygen output)
- No CHANGELOG.md
- No new user guides -- only standardize existing 18
- Header comment content preserved -- only style standardized
