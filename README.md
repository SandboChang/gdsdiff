# gdsdiff

`gdsdiff` compares the geometry and hierarchy of two GDSII layouts and emits
neutral, machine-readable evidence. It supports exact CPU verification by
default and can use optional CUDA or Apple Metal acceleration when a compatible
system toolchain and device are available.

The package is usable on CPU-only machines. CUDA, Metal, CMake, and compiler
toolchains are runtime capabilities, not mandatory Python dependencies.

## Install

For a future published release:

```bash
python -m pip install gdsdiff
```

For local development from this checkout:

```bash
python -m pip install -e ".[test]"
python -m pytest
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
  `nvcc` available. On WSL, install the NVIDIA WSL CUDA toolkit without a Linux
  display driver.
- On-demand builds default to NVIDIA's `-arch=native`: `nvcc` detects visible
  GPUs and emits native SASS without PTX. For cross-machine builds, set
  `GDSDIFF_CUDA_ARCHITECTURES` to comma-separated compute capabilities without
  decimals (for example, `89,120`). Targets and standard `NVCC_*_FLAGS` are
  included in the content-addressed cache key.
- Metal requires Apple Silicon/macOS with the Xcode command-line tools.
- Missing or unusable accelerators fall back to CPU unless `require_gpu=True`
  is explicitly requested.

Runtime capability details are available through
`gdsdiff.discover_native_capabilities()`.

## License

GPL-3.0-only. See `LICENSE`.
