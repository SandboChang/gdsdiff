# gdsdiff Semantics

## Authority

The final geometry authority is `gdstk.boolean` on integer snapped polygons.
Broad-phase candidate discovery can reduce the amount of exact work, but it
does not decide equality by itself.

## Inputs

The comparator reads both GDS files with `gdstk.read_gds`. A selected top cell
is flattened through `Cell.get_polygons(apply_repetitions=True)` under the
configured path policy. Current MVP path policy is `gdstk`: paths are accepted
as polygons according to the installed `gdstk` behavior.

Layer selection is by `(layer, datatype)`. `all` means every layer present in
either input after extraction.

## Grid

The MVP requires one database-unit grid for both files. Explicit `dbu` and
`precision` values must match if both are supplied. Coordinates are snapped to
integer DBU values using the repository rounding policy
`nearest-ties-away-from-zero`.

The JSON report records the source unit/precision metadata and selected grid
semantics. A comparison that cannot be represented safely on this grid must
fail closed rather than silently weakening precision.

## Exact Difference Model

For each selected layer, the oracle computes:

- `old_minus_new`: geometry removed from the old file.
- `new_minus_old`: geometry added in the new file.
- `xor`: the union of both directional differences.

Areas are reported in DBU squared. Half-integer exact areas are represented as
strings ending in `.5` because polygon twice-area is tracked as an integer.

## Expected Differences

Expected differences are explicit policy regions:

- Whole-layer allowance: useful only when the entire layer/direction is known
  to be intentionally changed.
- Bounded allowance: preferred for local edits. The bbox and optional halo are
  applied in DBU coordinates.

Any component outside the allowed policy is `unexpected`. Reports and exit
codes must treat unresolved or unexpected differences as failures for CI.

## Hierarchy And Assemblies

For hierarchy-heavy assemblies, compare cell identities and reference placement
where possible before flattening large designs. Assembly profiles can check top
cell names, reference counts, expected layers, manifest columns, manifest row
counts, and expected files.

A wafer array that instantiates many chips is validated by combining chip-level
exact comparison with wafer readback/profile checks unless wafer-level geometry
is the actual subject.

## Current Backend Semantics

CPU broad-phase and CUDA native kernels are candidate generators. The exact
fallback remains `gdstk`. CUDA Python dispatch is not enabled as a production
pass/fail authority in Plan 1. Metal and Manhattan GPU Boolean work are
documented future optional campaigns.
