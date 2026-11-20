# gdsdiff benchmarks

Run the benchmark harness from an editable development installation:

```bash
python benchmarks/gdsdiff/run_benchmarks.py \
  --pair tiny:before.gds:after.gds \
  --repeats 7 \
  --output results/benchmark.json
```

The `gdsdiff-benchmark-v1` result measures the API's windowed and exact modes.
It does not include process startup or standard CLI artifact writing.

For CUDA CLI qualification, use a repo-local interpreter (native Linux shown):

```bash
.venv/bin/python -m gdsdiff before.gds after.gds --backend cuda --require-gpu --output-dir .dump/cuda-check
.venv/bin/python -m gdsdiff before.gds after.gds --backend cpu --evidence-engine gdstk-oracle --output-dir .dump/cpu-check
```

Pass the same selection, grid, and policy options and compare full-vertex and
inventory evidence before timing. Time the CUDA command in fresh processes:
one warm-up and at least seven measured runs with the same artifacts, device
preparation, and cache conditions. Report the median, the actual coverage
mode, and whether the geometry or OS file cache was warm. An independent CPU
audit is a separate measurement. Use the [report timing guide](../../docs/gdsdiff_report.md#read-timings)
for non-overlapping stage totals. Keep benchmark results in ignored `results/`.
