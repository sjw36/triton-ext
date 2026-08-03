#!/usr/bin/env python3
"""Opt driver for the MegaBulkSync lowering pass.

Runs ``mega-bulk-sync-lowering`` over an MLIR input and prints the result.
Two extensions are imported: ``triton_mega`` registers the `mega` dialect so
`mega.bulk_sync` parses, and ``triton_mega_bulk_sync`` registers the pass that
lowers it. Both must be installed (``make build install``), along with Triton.

Run by hand with::

    ./mega_bulk_sync.py test/bulk_sync_lowering.mlir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing"))

import triton_mega  # noqa: E402, F401  registers the `mega` dialect on import
import triton_mega_bulk_sync  # noqa: E402, F401  registers the pass on import
from triton._C.libtriton import passes  # noqa: E402
from mlir_runner import run_passes  # noqa: E402

run_passes([passes.plugin.add_mega_bulk_sync])
