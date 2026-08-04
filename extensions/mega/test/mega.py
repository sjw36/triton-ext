#!/usr/bin/env python3
"""Opt driver for the Mega dialect.

Parses an MLIR input using the `mega` dialect and runs ``canonicalize`` to
verify that `mega` ops round-trip and are not folded away. Importing
``triton_mega`` registers the dialect with Triton; both Triton and the
extension must be installed (``make build install``).

This driver runs canonicalization only. Lowering `mega.bulk_sync` is a separate
pipeline — see the ``mega_bulk_sync.py`` driver beside this one.

Run by hand with::

    ./mega.py test/bulk_sync.mlir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "testing"))

import triton_mega  # noqa: E402, F401  registers the dialect on import
from triton._C.libtriton import passes  # noqa: E402
from mlir_runner import run_passes  # noqa: E402

run_passes([passes.common.add_canonicalizer])
