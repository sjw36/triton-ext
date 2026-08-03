"""tlM language extension — out-of-tree Python DSL for the `mega` dialect.

Importing this package does two things:

  * imports ``triton_mega``, which registers the `mega` dialect and the
    `mega_bulk_sync` TritonOpBuilder method with Triton's Python bindings;
  * registers itself as ``triton.language.extra.tlM`` (via sys.modules), so
    ``import triton.language.extra.tlM as tlM`` works without a filesystem
    symlink.

Both happen at import time, so importing `tlM` before compiling is all the
setup there is — no ``TRITON_PLUGIN_PATHS``, ``PYTHONPATH``, or
``LD_LIBRARY_PATH``. The `triton-mega` wheel must be installed alongside this
one (``make build install``).

Usage:

    import tlM  # noqa: F401  registers triton.language.extra.tlM
    import triton.language.extra.tlM as tlM
    tlM.bulk_sync(arrival_ptr, release_ptr, num_programs, sense)
"""

__all__ = ["bulk_sync"]

import sys as _sys

# Registers the `mega` dialect and the `create_mega_bulk_sync` builder method
# that `bulk_sync` calls. Must precede the first kernel compile, which
# importing it here guarantees.
import triton_mega  # noqa: F401

from .bulk_sync import bulk_sync

# Register this module as triton.language.extra.tlM so that
# `import triton.language.extra.tlM` works without a filesystem symlink.
import triton.language.extra as _extra

_sys.modules["triton.language.extra.tlM"] = _sys.modules[__name__]
_extra.tlM = _sys.modules[__name__]
