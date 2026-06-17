"""Frontend tests for the `tlM` language extension's `bulk_sync` builtin.

These tests exercise the Python DSL surface (``triton.language.extra.tlM``)
end-to-end through Triton's frontend: a ``@triton.jit`` kernel calls
``tlM.bulk_sync(...)``, we run the frontend code generator (``ast_to_ttir``)
with a *stub* GPU target, and assert that the resulting TTIR contains the
``mega.bulk_sync`` op with the expected operands.

This deliberately stops at TTIR (the frontend's output) rather than running a
full ``triton.compile``: ``mega.bulk_sync`` is lowered by the out-of-tree
``mega-bulk-sync-lowering`` pass, not by the standard TTIR->TTGIR pipeline, so
there is no GPU code-gen path for it. Building TTIR with a stub target needs no
GPU, mirroring Triton's own ``triton._filecheck`` frontend tests.

Plugin discovery (the `mega` dialect plugin provides both the `mega` dialect
and the ``mega_bulk_sync`` TritonOpBuilder method), in order of priority:
  * ``MEGA_PLUGIN`` - explicit path to ``libmega.so``
  * ``<repo>/build*/lib/libmega.so`` (most recently built)

If the plugin or Triton's bindings can't be loaded, the module is skipped so it
is safe to collect in environments that haven't built the dialect.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

# language/tlM/test/ -> repo root is three levels up.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
# The pure-Python `tlM` language extension lives here.
_TLM_PYTHON_DIR = pathlib.Path(__file__).resolve().parents[1] / "python"


def _resolve_plugin() -> pathlib.Path | None:
    if env := os.environ.get("MEGA_PLUGIN"):
        p = pathlib.Path(env)
        return p if p.exists() else None
    for build_dir in ("build", "build-release", "build-debug"):
        cand = PROJECT_ROOT / build_dir / "lib" / "libmega.so"
        if cand.exists():
            return cand
    return None


_plugin = _resolve_plugin()
_skip_reason: str | None = None
if _plugin is None:
    _skip_reason = ("mega dialect plugin (libmega.so) not found; build the "
                    "dialect or set MEGA_PLUGIN to override")
else:
    # Plugins are enumerated by libtriton at import time, so TRITON_PLUGIN_PATHS
    # must be set *before* importing triton.
    existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    if str(_plugin) not in existing.split(":"):
        os.environ["TRITON_PLUGIN_PATHS"] = (f"{_plugin}:{existing}"
                                             if existing else str(_plugin))

tlM = None
if _skip_reason is None:
    try:
        import triton  # noqa: E402
        import triton.language as tl  # noqa: E402, F401
        from triton.compiler import ASTSource, make_backend  # noqa: E402
        from triton.backends.compiler import GPUTarget  # noqa: E402
        from triton._C.libtriton import ir  # noqa: E402
    except ImportError as exc:
        _skip_reason = f"triton import failed: {exc}"
    else:
        # Register the `tlM` language extension (`triton.language.extra.tlM`).
        if str(_TLM_PYTHON_DIR) not in sys.path:
            sys.path.insert(0, str(_TLM_PYTHON_DIR))
        try:
            import tlM  # noqa: F401, E402
            import triton.language.extra.tlM as tlM  # noqa: E402
        except ImportError as exc:
            _skip_reason = f"tlM language extension import failed: {exc}"

pytestmark = pytest.mark.skipif(_skip_reason is not None,
                                reason=_skip_reason or "")

# A stub target lets us build TTIR via the frontend without a real GPU; see
# triton._filecheck for the same approach.
_STUB_TARGET = GPUTarget("cuda", 100, 32) if _skip_reason is None else None


def _build_ttir(kernel_fn, signature: dict[str, str]) -> str:
    """Run the frontend code generator for ``kernel_fn`` and return TTIR text."""
    backend = make_backend(_STUB_TARGET)
    src = ASTSource(fn=kernel_fn, signature=signature, constexprs={})
    options = backend.parse_options({"sanitize_overflow": False})

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(_STUB_TARGET, options, codegen_fns, module_map, context)
    return module.str_nodebug()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bulk_sync_emits_op():
    """`tlM.bulk_sync(...)` lowers to a single `mega.bulk_sync` TTIR op."""

    @triton.jit
    def kernel(arrival_ptr, release_ptr, sense):
        tlM.bulk_sync(arrival_ptr, release_ptr, tl.num_programs(0), sense)

    ttir = _build_ttir(
        kernel,
        signature={
            "arrival_ptr": "*i32",
            "release_ptr": "*i32",
            "sense": "i32",
        },
    )
    assert "mega.bulk_sync" in ttir, f"expected mega.bulk_sync in TTIR:\n{ttir}"
    assert ttir.count("mega.bulk_sync") == 1, (
        f"expected exactly one mega.bulk_sync op:\n{ttir}")


def test_bulk_sync_operand_types():
    """The emitted op carries the pointer/i32 operands in the declared order."""

    @triton.jit
    def kernel(arrival_ptr, release_ptr, sense):
        tlM.bulk_sync(arrival_ptr, release_ptr, tl.num_programs(0), sense)

    ttir = _build_ttir(
        kernel,
        signature={
            "arrival_ptr": "*i32",
            "release_ptr": "*i32",
            "sense": "i32",
        },
    )
    # The assembly format prints the two pointer operand types after the colon:
    #   mega.bulk_sync %a, %r, %n, %s : !tt.ptr<i32>, !tt.ptr<i32>
    line = next((ln for ln in ttir.splitlines() if "mega.bulk_sync" in ln), "")
    assert line, f"no mega.bulk_sync line in TTIR:\n{ttir}"
    assert line.count("!tt.ptr<i32>") == 2, (
        f"expected two !tt.ptr<i32> operand types on:\n{line}")


def test_bulk_sync_in_branch():
    """The builtin works inside control flow (regression for region nesting)."""

    @triton.jit
    def kernel(arrival_ptr, release_ptr, sense):
        pid = tl.program_id(0)
        if pid == 0:
            tlM.bulk_sync(arrival_ptr, release_ptr, tl.num_programs(0), sense)

    ttir = _build_ttir(
        kernel,
        signature={
            "arrival_ptr": "*i32",
            "release_ptr": "*i32",
            "sense": "i32",
        },
    )
    assert "mega.bulk_sync" in ttir, f"expected mega.bulk_sync in TTIR:\n{ttir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
