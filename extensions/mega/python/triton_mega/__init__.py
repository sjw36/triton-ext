"""
The `mega` extension for Triton: dialect, builder method, and lowering pass.

Importing this package registers three things with Triton's Python bindings,
all from the one bundled `libmega.so`:

  * the `mega` dialect, so `mega.bulk_sync` parses and prints in any
    compilation context;
  * the `mega_bulk_sync` TritonOpBuilder method, which the `tlM` language
    extension calls as `builder.create_mega_bulk_sync([...])` to emit the op;
  * the `mega-bulk-sync-lowering` pass, reachable as
    ``triton._C.libtriton.passes.plugin.add_mega_bulk_sync``, which rewrites
    each `mega.bulk_sync` into an explicit grid-wide barrier built from Triton
    ops.

All three must happen before the first kernel is compiled, which importing this
package ahead of `triton.compile` guarantees.

The first-access hoist is *not* here: it is an optional optimization over this
dialect and ships as its own `triton-mega-hoist-first-access` wheel.
"""

from pathlib import Path

import triton._C.libtriton as _libtriton

# Register the Mega extension library with Triton.
PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libmega.so"
_libtriton.ir.extend_dialects_with(str(PLUGIN_LIBRARY))  # adds dialects
_libtriton.ir.builder.extend_with(str(PLUGIN_LIBRARY))  # adds ops
_libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes
