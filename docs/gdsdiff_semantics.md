# gdsdiff Semantics

## Authority

The independent reference is `gdstk.boolean` on integer snapped polygons.
Native and GPU evidence paths use the same geometry contract. Bounding boxes,
hashes, and total area alone do not establish exact coverage equality.

## Inputs

The CPU oracle reads both GDS files with `gdstk.read_gds` and flattens the
selected cells through `Cell.get_polygons(apply_repetitions=True)` under the
configured path policy. The default CPU path policy is `gdstk`. GPU evidence
can instead use native integer polygon extraction on a compatible source
grid; unsupported native records or transforms can require CPU fallback.

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
Source-polygon inventory is a separate comparison, so equal filled geometry
does not hide a split, merge, or duplicate change.

For proven sparse edits, both sides contain the same geometry off
the delta. Exact reduction keeps every changed polygon and every common
same-layer polygon whose bounds touch the changed bounds, then runs the
directional Booleans on that reduced input. A decline does not prove equality;
it returns control to the normal coverage path.

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

CPU broad-phase and optional CUDA/Metal native kernels generate accelerated
evidence under the same backend-neutral contracts. Full exact CPU verification
through `gdstk` remains the reference authority. CUDA installation is CPU-safe;
native compilation occurs only during explicit preparation or first explicit
CUDA use. Metal remains experimental until qualified on Apple Silicon hardware.
