"""tlM language extension — out-of-tree Python DSL for the `mega` dialect.

Importing this package registers it as `triton.language.extra.tlM` (via
sys.modules), so `import triton.language.extra.tlM as tlM` works without a
filesystem symlink. The builtins emit `mega` dialect ops through custom
TritonOpBuilder methods provided by the `mega` dialect plugin (libmega.so,
loaded via TRITON_PLUGIN_PATHS).

Usage:

    import triton.language.extra.tlM as tlM
    tlM.bulk_sync(arrival_ptr, release_ptr, num_programs, sense)
"""

__all__ = ["bulk_sync"]

from .bulk_sync import bulk_sync

# Register this module as triton.language.extra.tlM so that
# `import triton.language.extra.tlM` works without a filesystem symlink.
import sys as _sys
import triton.language.extra as _extra

_sys.modules["triton.language.extra.tlM"] = _sys.modules[__name__]
_extra.tlM = _sys.modules[__name__]
