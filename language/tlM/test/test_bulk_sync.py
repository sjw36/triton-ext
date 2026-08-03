"""Frontend tests for the `tlM` language extension's `bulk_sync` builtin.

These tests exercise the Python DSL surface (``triton.language.extra.tlM``)
through Triton's frontend: a ``@triton.jit`` kernel calls ``tlM.bulk_sync(...)``
and we assert that the resulting TTIR contains the ``mega.bulk_sync`` op with
the expected operands.

Like the other plugin tests in this repo (see
``pass/ArithmeticIntensity/test/test_arithmetic_intensity.py``), each test
spawns a subprocess running ``build_ttir.py`` with a modified environment:
``TRITON_PLUGIN_PATHS``, ``PYTHONPATH``, and ``LD_LIBRARY_PATH`` set. The
subprocess matters: the `mega` dialect plugin is ABI-bound to the Triton pinned
in ``ci/triton-hash.txt`` (the one under ``TRITON_INSTALL_DIR``), so the tests
must not run against whatever ``triton`` happens to be importable in the
ambient environment. ``PYTHONPATH`` also carries the pure-Python `tlM` package,
which self-registers as ``triton.language.extra.tlM`` on import.

`libtriton` enumerates ``TRITON_PLUGIN_PATHS`` at import time, registering both
the `mega` dialect and the ``mega_bulk_sync`` TritonOpBuilder method, so the
subprocess needs no explicit plugin-loading call.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# language/tlM/test/ -> repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = Path(os.environ.get("BUILD_DIR", PROJECT_ROOT / "build"))
TRITON_INSTALL_DIR = Path(os.environ["TRITON_INSTALL_DIR"])
LLVM_INSTALL_DIR = Path(os.environ["LLVM_INSTALL_DIR"])
BUILD_TTIR_SCRIPT = Path(__file__).resolve().parent / "build_ttir.py"
# The pure-Python `tlM` language extension lives here.
TLM_PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
PLUGIN_LIB = BUILD_DIR / "lib" / "libmega.so"


@pytest.fixture(scope="module")
def build_ttir():
    """Return a callable mapping a kernel name to its TTIR text."""

    if not PLUGIN_LIB.exists():
        pytest.fail(f"plugin library not found; build {PLUGIN_LIB}.")
    triton_python = TRITON_INSTALL_DIR / "python"
    if not triton_python.is_dir():
        # Without it the subprocess silently falls back to whatever `triton` is
        # installed, whose plugin ABI need not match the one we built against.
        pytest.fail(f"pinned Triton package not found at {triton_python}; "
                    f"re-fetch it with `ci/download-artifact.py triton`.")

    env_overrides = {
        "TRITON_PLUGIN_PATHS":
        str(PLUGIN_LIB),
        "PYTHONPATH":
        os.pathsep.join([str(triton_python), str(TLM_PYTHON_DIR)]),
        "LD_LIBRARY_PATH":
        str(LLVM_INSTALL_DIR / "lib"),
    }

    def _run(kernel: str) -> str:
        env = {**os.environ, **env_overrides}
        result = subprocess.run(
            [sys.executable, str(BUILD_TTIR_SCRIPT), kernel],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"build_ttir.py {kernel} failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}")
        return result.stdout

    return _run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bulk_sync_emits_op(build_ttir):
    """`tlM.bulk_sync(...)` lowers to a single `mega.bulk_sync` TTIR op."""
    ttir = build_ttir("straight_line")
    assert "mega.bulk_sync" in ttir, f"expected mega.bulk_sync in TTIR:\n{ttir}"
    assert ttir.count("mega.bulk_sync") == 1, (
        f"expected exactly one mega.bulk_sync op:\n{ttir}")


def test_bulk_sync_operand_types(build_ttir):
    """The emitted op carries the pointer/i32 operands in the declared order."""
    ttir = build_ttir("straight_line")
    # The assembly format prints the two pointer operand types after the colon:
    #   mega.bulk_sync %a, %r, %n, %s : !tt.ptr<i32>, !tt.ptr<i32>
    line = next((ln for ln in ttir.splitlines() if "mega.bulk_sync" in ln), "")
    assert line, f"no mega.bulk_sync line in TTIR:\n{ttir}"
    assert line.count("!tt.ptr<i32>") == 2, (
        f"expected two !tt.ptr<i32> operand types on:\n{line}")


def test_bulk_sync_in_branch(build_ttir):
    """The builtin works inside control flow (regression for region nesting)."""
    ttir = build_ttir("in_branch")
    assert "mega.bulk_sync" in ttir, f"expected mega.bulk_sync in TTIR:\n{ttir}"
