"""
The HoistFirstAccess pass for Triton, registered as an extension.

Registers `mega-hoist-first-access`, reachable from Python as
``triton._C.libtriton.passes.plugin.add_mega_hoist_first_access``. The pass
hoists first-access `tt.load`s above a `mega.bulk_sync` so the memory latency
overlaps the barrier's spin-wait.

It matches the barrier by mnemonic, but parsing IR that contains
`mega.bulk_sync` needs the dialect, so import ``triton_mega`` alongside this
package. Run it *before* `mega-bulk-sync-lowering`, while the barrier is still
a single op.
"""

from pathlib import Path

import triton._C.libtriton as _libtriton

# Register the HoistFirstAccess extension library with Triton.
PLUGIN_DIR = Path(__file__).resolve().parent
PLUGIN_LIBRARY = PLUGIN_DIR / "libmega_hoist_first_access.so"
_libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))  # adds passes
