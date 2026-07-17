# gdsdiff

`gdsdiff` compares the geometry and hierarchy of two GDSII layouts and emits
neutral, machine-readable evidence. It supports exact CPU verification by
default and can use optional CUDA or Apple Metal acceleration when a compatible
system toolchain and device are available.

The package is usable on CPU-only machines. CUDA, Metal, CMake, and compiler
toolchains are runtime capabilities, not mandatory Python dependencies.

## Install

```bash
python -m pip install gdsdiff
```

For local development from this checkout:

```bash
UV_PROJECT_ENVIRONMENT=.venv_wsl UV_LINK_MODE=copy uv sync --locked --extra dev
TMPDIR=/tmp .venv_wsl/bin/python -m pytest
```

Use a repository-local virtual environment: `.venv_wsl` on Windows 11 through
WSL, `.venv_mac` on macOS, and `.venv` on native Linux.

## Command line

```bash
python -m gdsdiff before.gds after.gds --output-dir comparison
```

Exit codes are stable: `0` means the selected failure policy passed, `1` means
differences failed the policy, and `2` means the comparison could not produce
valid evidence. Reports use factual directions such as `coverage_before_only`
and `coverage_after_only`.

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

See `docs/` for comparison semantics, report format, engine details, and
debugging guidance.

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
- Missing or unusable accelerators fall back to CPU unless `require_gpu=True`
  is explicitly requested.

Installation and import never compile GPU code. Inspect the CUDA toolchain with
`gdsdiff.inspect_cuda_toolchain()`, then build and verify the optional backend
with `gdsdiff.prepare_cuda()` or `gdsdiff-prepare-cuda --json`. Explicit
`backend="cuda"` use retains lazy preparation. Passive diagnostics and
`discover_native_capabilities()` report only already-prepared backends.

Native artifacts are stored in a private, content-addressed user cache keyed by
source, compiler/toolkit identity, build flags, and native GPU identity. The
optional geometry cache uses JSON plus non-executable NumPy arrays; legacy
pickle cache entries are ignored and rebuilt.

## License

GPL-3.0-only. See `LICENSE`.
