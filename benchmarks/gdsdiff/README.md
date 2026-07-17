# gdsdiff benchmarks

Run the benchmark harness from an editable development installation:

```bash
python benchmarks/gdsdiff/run_benchmarks.py \
  --pair tiny:before.gds:after.gds \
  --repeats 7 \
  --output results/benchmark.json
```

The `gdsdiff-benchmark-v1` result includes end-to-end wall time and comparator
timing fields. Use a warm-up before at least seven measured runs for performance
qualification, and compare exact semantic results before timing conclusions.
