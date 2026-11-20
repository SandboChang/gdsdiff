# gdsdiff Debugging And Release Checklist

## Debugging

When a comparison fails unexpectedly:

1. Run `python -m gdsdiff inspect` on each file to confirm the intended top,
   layer/datatype pairs, and DBU.
2. Inspect `report.json` for `old_input`, `new_input`, and `selection`.
3. Check `backend.gpu_work_performed`, `backend.gpu_requirement_met` when
   requested, and `exact_fallback.failures`. An available backend alone does
   not show that GPU work ran.
4. Check `status` and exactness, then inspect the affected per-layer components,
   directional vertices, and inventory in the sidecar.
5. Open `diff.gds` only as a visual aid; use report coordinates for decisions.
6. For chip edits, compare local windows around the intended feature.
7. For wafer/block edits, check top-cell identity, reference counts, layer
   presence, and manifests before deciding a full flattening diff is needed.
8. Treat exit code `2` as a hard failure in CI.

Common causes:

- Wrong top cell selected.
- Layer selector omitted the changed layer.
- Expected-difference region is too small or uses the wrong direction.
- Source GDS precision differs from assumed DBU semantics.
- The rebuild command failed before comparison.
- A wafer assembly changed by manifest/reference count without local chip
  geometry changing.

For an independent reference, run the same inputs and selection with
`--backend cpu --evidence-engine gdstk-oracle`. To audit a GPU comparison in
the same invocation, use `--audit-cpu-oracle`. It reopens both files and can
be much slower than normal native-buffer coverage. See
[report interpretation](gdsdiff_report.md) when artifacts differ in loop order
or when stage timings are nested.

## Release Checklist

- Run focused comparator tests:

```bash
.venv/bin/python -m pytest
```

- Run a direct CLI smoke check:

```bash
.venv/bin/python -m gdsdiff before.gds before.gds --top TOP --output-dir /tmp/chip2_compare
```

- Run workflow wrapper smoke checks for geometry-affecting project changes.
- Run benchmark smoke:

```bash
.venv/bin/python benchmarks/gdsdiff/run_benchmarks.py --pair smoke:before.gds:after.gds --mode windowed --tile-size 256 --output /tmp/gdsdiff_benchmark_smoke.json
```

- Confirm generated cache/artifact files are not accidentally committed.
- Confirm no generated `.gds` outputs are added to git.

## Known Limitations

- CUDA native components are prepared explicitly or on first explicit CUDA use;
  automatic selection uses prepared capabilities and retains CPU exact fallback.
- Metal behavior is covered by portable tests; real acceleration requires
  macOS on Apple Silicon with the Xcode command-line tools.
- GPU acceleration does not implement a general polygon Boolean oracle.
  Rectangle coverage, record deltas, and source-edge work have accelerator
  implementations; sparse and hole-aware corrections may use `gdstk`.
- A `gdstk` hole-link warning does not certify a polygon. Check the report
  for a verified sparse recovery, or use the independent CPU audit for
  complex source-edge results. An unrecovered numerical error is not an
  exact verdict.
- Full-wafer flattening can be impractical; use profile/readback checks unless
  wafer-level geometry itself changed.
- The comparator is only as stable as the pinned/local `gdstk` polygonization
  semantics for paths and Booleans.
