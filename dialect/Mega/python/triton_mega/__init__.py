"""
The `mega` dialect for Triton, registered as an extension.

Importing this package registers two things with Triton's Python bindings:

  * the `mega` dialect, so `mega.bulk_sync` parses and prints in any
    compilation context;
  * the `mega_bulk_sync` TritonOpBuilder method, which the `tlM` language
    extension calls as `builder.create_mega_bulk_sync([...])` to emit the op.

Both must happen before the first kernel is compiled, which importing this
package ahead of `triton.compile` guarantees.
"""

from pathlib import Path

import triton._C.libtriton as _libtriton

# Register the Mega extension library with Triton.
PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libmega.so"
_libtriton.ir.extend_dialects_with(str(PLUGIN_LIBRARY))  # adds dialects
_libtriton.ir.builder.extend_with(str(PLUGIN_LIBRARY))  # adds ops
