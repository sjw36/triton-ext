#!/usr/bin/env python3
"""Opt driver for the HoistFirstAccess pass.

Runs ``mega-hoist-first-access`` over an MLIR input and prints the result.
Two extensions are imported: ``triton_mega`` registers the `mega` dialect so
`mega.bulk_sync` parses, and ``triton_mega_hoist_first_access`` registers the
pass. Both must be installed (``make build install``), along with Triton.

No canonicalizer runs afterwards: the test checks op *order* around the
barrier, which a later pass could disturb.

Run by hand with::

    ./mega_hoist_first_access.py test/hoist_first_access.mlir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing"))

import triton_mega  # noqa: E402, F401  registers the `mega` dialect on import
import triton_mega_hoist_first_access  # noqa: E402, F401  registers the pass
from triton._C.libtriton import passes  # noqa: E402
from mlir_runner import run_passes  # noqa: E402

run_passes([passes.plugin.add_mega_hoist_first_access])
