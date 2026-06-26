# gdsdiff Debugging And Release Checklist

## Debugging

When a comparison fails unexpectedly:

1. Confirm the selected top cell and layers.
2. Inspect `compare_report.json` for `old_input`, `new_input`, and
   `selection`.
3. Check `backend.fallback_reason` before assuming GPU code ran.
4. Inspect per-layer `components`, especially bbox and direction.
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

## Release Checklist

- Run focused comparator tests:

```bash
.venv/bin/python -m pytest -c scgds/tests/pytest.ini scgds/tests/gdsdiff
```

- Run a direct CLI smoke check:

```bash
.venv/bin/python -m scgds.gdsdiff before.gds before.gds --top TOP --output-dir /tmp/chip2_compare
```

- Run workflow wrapper smoke checks for geometry-affecting project changes.
- Run benchmark smoke:

```bash
.venv/bin/python benchmarks/gdsdiff/run_benchmarks.py --max-pairs 1 --mode windowed --tile-size 256 --output /tmp/gdsdiff_benchmark_smoke.json
```

- Confirm generated cache/artifact files are not accidentally committed.
- Confirm no generated `.gds` outputs are added to git.
- Confirm optional Gate F work remains disabled unless separately qualified.

## Known Limitations

- CUDA kernels are qualified as broad-phase/native components, but Python
  production dispatch still falls back to CPU exact verification.
- No Metal backend is implemented in this Linux/CUDA campaign.
- No GPU Boolean exact oracle is enabled. `gdstk` remains the exact authority.
- Full-wafer flattening can be impractical; use profile/readback checks unless
  wafer-level geometry itself changed.
- The comparator is only as stable as the pinned/local `gdstk` polygonization
  semantics for paths and Booleans.
