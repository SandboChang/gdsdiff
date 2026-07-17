from __future__ import annotations

import json
import subprocess
import sys

from gdsdiff import backend_diagnostics


def test_backend_diagnostics_is_cpu_safe_and_structured():
    diagnostics = backend_diagnostics()

    assert diagnostics["capabilities"]["cpu_available"] is True
    assert diagnostics["platform"]["system"]
    assert diagnostics["python"]["version"]
    assert diagnostics["native_cache"]["environment"] == "GDSDIFF_NATIVE_CACHE_DIR"


def test_diagnostics_module_prints_json():
    completed = subprocess.run(
        [sys.executable, "-m", "gdsdiff.diagnostics"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["capabilities"]["cpu_available"] is True
