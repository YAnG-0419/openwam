"""Exercise rank logging and failure teardown without GPU/model dependencies."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ENTRYPOINT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
HARNESS = r'''
import builtins
import io
import runpy
import sys
import traceback
import types

hydra = types.ModuleType("hydra")
hydra.main = lambda **kwargs: lambda fn: fn
sys.modules["hydra"] = hydra
omega = types.ModuleType("omegaconf")
omega.DictConfig = dict
omega.OmegaConf = types.SimpleNamespace(to_yaml=lambda cfg: "test config")
sys.modules["omegaconf"] = omega
torch = types.ModuleType("torch")
dist = types.ModuleType("torch.distributed")
dist.is_available = lambda: True
dist.is_initialized = lambda: True
dist.destroy_process_group = lambda: print("DISTRIBUTED_TEARDOWN")
torch.distributed = dist
sys.modules["torch"] = torch
sys.modules["torch.distributed"] = dist

namespace = runpy.run_path(sys.argv[1])
main = namespace["main"]
original_print = builtins.print
original_stdout = sys.stdout

def train(cfg):
    print("TRAIN_STDOUT")
    print("TRAIN_STDERR", file=sys.stderr)
    buffer = io.StringIO()
    print("FILE_OUTPUT", file=buffer)
    assert buffer.getvalue() == "FILE_OUTPUT\n"
    if sys.argv[2] == "fail":
        try:
            raise ValueError("nested worker error")
        except ValueError:
            traceback.print_exc()
        raise RuntimeError("original training failure")

main.__globals__["_train"] = train
main.__globals__["_install_startup_noise_filter"] = lambda: None
try:
    main({})
finally:
    assert builtins.print is original_print
    assert sys.stdout is original_stdout
    print("OUTPUT_RESTORED")
'''


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("fail", [False, True])
def test_training_output_and_teardown(rank, fail):
    result = subprocess.run(
        [sys.executable, "-c", HARNESS, str(ENTRYPOINT), "fail" if fail else "ok"],
        env={**os.environ, "RANK": str(rank), "LOCAL_RANK": str(rank)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (1 if fail else 0), result.stderr
    assert ("TRAIN_STDOUT" in result.stdout) == (rank == 0)
    assert "TRAIN_STDERR" in result.stderr
    assert "OUTPUT_RESTORED" in result.stdout
    assert ("DISTRIBUTED_TEARDOWN" in result.stdout) == (not fail)
    if fail:
        assert f"RANK {rank} EXCEPTION" in result.stderr
        assert "ValueError: nested worker error" in result.stderr
        assert "RuntimeError: original training failure" in result.stderr
        assert "Traceback (most recent call last)" in result.stderr
