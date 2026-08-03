"""
The MegaBulkSync lowering pass for Triton, registered as an extension.

Registers `mega-bulk-sync-lowering`, reachable from Python as
``triton._C.libtriton.passes.plugin.add_mega_bulk_sync``. The pass rewrites
each `mega.bulk_sync` into an explicit grid-wide barrier built from Triton ops.

Parsing IR that *contains* `mega.bulk_sync` additionally needs the dialect, so
import ``triton_mega`` alongside this package.
"""

from pathlib import Path

import triton._C.libtriton as _libtriton

# Register the MegaBulkSync extension library with Triton.
PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libmega_bulk_sync.so"
_libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes
