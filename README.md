# gdsdiff

`gdsdiff` compares the geometry and hierarchy of two GDSII layouts and emits
neutral, machine-readable evidence. The CLI selects an available CUDA or Metal
backend automatically, with exact CPU `gdstk` as the reference and fallback.

The package is usable on CPU-only machines. CUDA, Metal, CMake, and compiler
toolchains are runtime capabilities, not mandatory Python dependencies.

## Install

```bash
python -m pip install gdsdiff
```

For local development from this checkout, use the `uv` version pinned in
`pyproject.toml`. On WSL:

```bash
UV_PROJECT_ENVIRONMENT=.venv_wsl UV_LINK_MODE=copy uv sync --locked --no-dev
UV_PROJECT_ENVIRONMENT=.venv_wsl_test UV_LINK_MODE=copy uv sync --locked --extra dev
TMPDIR=/tmp .venv_wsl_test/bin/python -m pytest
```

On WSL, `.venv_wsl` is the clean runtime environment and `.venv_wsl_test` owns
all test tooling. On native Linux, run `uv sync --locked --extra dev` and
`.venv/bin/python -m pytest`. On macOS, set `UV_PROJECT_ENVIRONMENT=.venv_mac`
for the sync and run tests with `.venv_mac/bin/python -m pytest`.

## Command line

```bash
python -m gdsdiff before.gds after.gds --output-dir comparison
```

`--output-dir` writes `report.json`, `exact_diff.md`, and the lossless
`diff_vertices.jsonl` sidecar. Coverage and source-polygon inventory are
separate, so a split or merge can be reported even if coverage is unchanged.
The default neutral comparison returns `0` for identical evidence, `1` for
differences to review, and `2` when the requested comparison cannot be verified.

To check a single file's units, cells, and geometry layers before comparing:

```bash
python -m gdsdiff inspect layout.gds
python -m gdsdiff inspect layout.gds --top TOP --json
```

Inspection counts stored records without expanding reference arrays. Use `--top`
when the file has more than one top-level cell.

To require CUDA or run an independent CPU reference:

```bash
python -m gdsdiff before.gds after.gds --backend cuda --require-gpu --output-dir cuda_diff
python -m gdsdiff before.gds after.gds --backend cpu --evidence-engine gdstk-oracle --output-dir cpu_diff
```

Check `backend.gpu_work_performed` in the CUDA report. Exact evidence can combine
CUDA record comparison with `gdstk` coverage, so GPU use does not imply that
every stage ran on device. For exact hole correction, the sparse route can
handle differing polygon counts after proving that omitted records match.
Add `--audit-cpu-oracle` to reread both inputs independently after GPU cleanup.

## Python API

```python
from gdsdiff import CompareConfig, compare_gds_files

result = compare_gds_files(
    CompareConfig("before.gds", "after.gds", backend="cpu")
)
print(result.status, result.exit_code)
```

Polygon extraction is public so downstream tools such as `ebeamtime` can share
the same zero-copy `PolygonBuffer` representation:

```python
from gdsdiff import extract_gds_polygons

extraction = extract_gds_polygons("layout.gds", layers="1:0,2:0")
print(extraction.buffer.polygon_count)
```

See the [command guide](docs/gdsdiff.md), [Python API](docs/api.md),
[report guide](docs/gdsdiff_report.md), and [agent instructions](agents/gdsdiff.md).

## Optional acceleration

- CUDA requires an NVIDIA GPU, a compatible driver, and the CUDA toolkit with
  `nvcc` available. On WSL, install the Windows NVIDIA display driver and the
  WSL CUDA toolkit, never a Linux display driver. `nvcc` is discovered from
  `PATH`, `CUDA_HOME`, `/usr/local/cuda`, or `/opt/cuda`.
- On-demand builds default to NVIDIA's `-arch=native`: `nvcc` detects visible
  GPUs and emits native SASS without PTX. For cross-machine builds, set
  `GDSDIFF_CUDA_ARCHITECTURES` to comma-separated compute capabilities without
  decimals (for example, `89,120`). Targets and standard `NVCC_*_FLAGS` are
  included in the content-addressed cache key.
- Metal requires Apple Silicon/macOS with the Xcode command-line tools.
- When a requested accelerator fails, `gdsdiff` reports the failing stage and
  attempts exact CPU comparison. With `--require-gpu` (Python: `require_gpu=True`),
  it returns exit 2 if no GPU comparison ran, even if the labeled CPU evidence
  is complete. A CPU audit after GPU comparison is not a CPU-only fallback.

Installation and import never compile GPU code. Inspect the CUDA toolchain with
`gdsdiff.inspect_cuda_toolchain()`, then build and verify the optional backend
with `gdsdiff.prepare_cuda()` or `python -m gdsdiff.cuda_setup --json`. Explicit
`backend="cuda"` use retains lazy preparation. Passive diagnostics and
`discover_native_capabilities()` report only already-prepared backends.

Native artifacts are stored in a private, content-addressed user cache keyed by
source, compiler/toolkit identity, build flags, and native GPU identity. The
optional geometry cache uses JSON plus non-executable NumPy arrays; legacy
pickle cache entries are ignored and rebuilt.

## License

GPL-3.0-only. See `LICENSE`.
