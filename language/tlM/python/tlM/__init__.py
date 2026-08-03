"""tlM language extension — out-of-tree Python DSL for the `mega` dialect.

Importing this package registers it as `triton.language.extra.tlM` (via
sys.modules), so `import triton.language.extra.tlM as tlM` works without a
filesystem symlink. The builtins emit `mega` dialect ops through custom
TritonOpBuilder methods provided by the `mega` dialect plugin (libmega.so).

The plugin has to be loaded into Triton's Python bindings before the first
kernel is compiled:

    import tlM
    tlM.register_plugin("<repo>/build/lib/libmega.so")

Usage:

    import triton.language.extra.tlM as tlM
    tlM.bulk_sync(arrival_ptr, release_ptr, num_programs, sense)
"""

__all__ = ["bulk_sync", "register_plugin"]

import os as _os
import sys as _sys

from .bulk_sync import bulk_sync

# Register this module as triton.language.extra.tlM so that
# `import triton.language.extra.tlM` works without a filesystem symlink.
import triton.language.extra as _extra

_sys.modules["triton.language.extra.tlM"] = _sys.modules[__name__]
_extra.tlM = _sys.modules[__name__]

_PLUGIN_NAME = "libmega.so"
_registered: set[str] = set()


def _plugin_paths_env() -> list[str]:
    env = _os.environ.get("TRITON_PLUGIN_PATHS", "")
    return [_os.path.abspath(p) for p in env.split(_os.pathsep) if p]


def _default_plugin_path() -> str:
    """Find `libmega.so` in `$MEGA_PLUGIN`, else in `$TRITON_PLUGIN_PATHS`."""
    if path := _os.environ.get("MEGA_PLUGIN"):
        return path
    for path in _plugin_paths_env():
        if _os.path.basename(path) == _PLUGIN_NAME:
            return path
    raise RuntimeError(
        f"cannot locate {_PLUGIN_NAME}: pass its path to register_plugin(), "
        f"or set MEGA_PLUGIN / TRITON_PLUGIN_PATHS.")


def register_plugin(path: str | None = None) -> str:
    """Make the `mega` dialect plugin available to Triton's Python bindings.

    Two registrations are needed, both before the first kernel compiles:

      * the dialect, so that `ir.load_dialects` loads `mega` into every
        compilation context;
      * the TritonOpBuilder method `create_mega_bulk_sync` that `bulk_sync`
        calls.

    How they happen depends on the Triton build. Newer Triton exposes
    `ir.extend_dialects_with` / `ir.builder.extend_with` and loads nothing on
    its own; older Triton (including the revision pinned in
    `ci/triton-hash.txt`) instead enumerates `TRITON_PLUGIN_PATHS` while
    `libtriton` is imported, so this call only checks that the plugin was in
    that list. Either way `TRITON_PLUGIN_PATHS` must be set *before*
    `import triton` for the older path to work.

    `path` defaults to `$MEGA_PLUGIN`, else to the `libmega.so` entry of
    `$TRITON_PLUGIN_PATHS`. Repeated calls with the same path are no-ops.

    Returns the path that was registered.
    """
    from triton._C.libtriton import ir

    path = _os.path.abspath(path or _default_plugin_path())
    if path in _registered:
        return path
    if not _os.path.exists(path):
        raise FileNotFoundError(f"{_PLUGIN_NAME} not found at {path}")
    if hasattr(ir, "extend_dialects_with"):
        ir.extend_dialects_with(path)
        ir.builder.extend_with(path)
    elif path not in _plugin_paths_env():
        raise RuntimeError(
            f"this Triton build loads plugins from TRITON_PLUGIN_PATHS when "
            f"`libtriton` is imported; set it to include {path} before "
            f"importing triton.")
    _registered.add(path)
    return path
