from __future__ import annotations

import json

import pytest

import gdsdiff.cuda_setup as cuda_setup


def test_prepare_cuda_reports_invalid_architecture_as_controlled_failure(monkeypatch):
    monkeypatch.setenv("GDSDIFF_CUDA_ARCHITECTURES", "invalid")
    with pytest.raises(cuda_setup.CudaPreparationError, match="invalid CUDA architecture"):
        cuda_setup.prepare_cuda()


def test_prepare_cuda_cli_uses_exit_two_for_unavailable_backend(monkeypatch, capsys):
    monkeypatch.setattr(
        cuda_setup,
        "prepare_cuda",
        lambda: (_ for _ in ()).throw(cuda_setup.CudaPreparationError("unavailable")),
    )
    assert cuda_setup.main(["--json"]) == 2
    assert json.loads(capsys.readouterr().out) == {"ready": False, "error": "unavailable"}
