"""Frontend tests for the `tlM` language extension's `bulk_sync` builtin.

These tests exercise the Python DSL surface (``triton.language.extra.tlM``)
through Triton's frontend: a ``@triton.jit`` kernel calls ``tlM.bulk_sync(...)``
and we assert that the resulting TTIR contains the ``mega.bulk_sync`` op with
the expected operands.

Each test spawns a subprocess running ``build_ttir.py``. The subprocess needs
no environment overrides: `tlM` and the `mega` dialect are installed wheels, so
``import tlM`` loads the plugin and registers ``triton.language.extra.tlM``.
It stays a subprocess because loading a plugin mutates Triton's global dialect
and builder registries, which is worth keeping out of the pytest process.

The extensions must be installed first (``make build install``); the fixture
skips rather than fails if they are not, matching ``testing/test_plugins.py``.
There is no ABI-mismatch hazard to guard against here — a wheel is built
against the Triton it is installed next to.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BUILD_TTIR_SCRIPT = Path(__file__).resolve().parent / "build_ttir.py"


@pytest.fixture(scope="module")
def build_ttir():
    """Return a callable mapping a kernel name to its TTIR text."""

    for package in ("tlM", "triton_mega"):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"{package} not installed "
                        f"(run `make build && make install`)")

    def _run(kernel: str) -> str:
        result = subprocess.run(
            [sys.executable, str(BUILD_TTIR_SCRIPT), kernel],
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
