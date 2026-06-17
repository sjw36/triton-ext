"""tlM language builtins.

These builtins emit `mega` dialect ops via custom TritonOpBuilder methods
registered by the `mega` dialect plugin (libmega.so):
  - mega_bulk_sync(arrival_ptr, release_ptr, num_programs, sense)
"""

import triton.language.core as tl


def _to_ir(value, _semantic):
    """Coerce a builtin argument to an MLIR Value handle."""
    if isinstance(value, tl.tensor):
        return value.handle
    return _semantic._convert_elem_to_ir_value(
        tl._unwrap_if_constexpr(value), require_i64=False)


@tl.builtin
def bulk_sync(arrival_ptr, release_ptr, num_programs, sense, _semantic=None):
    """Grid-wide barrier across all cooperating programs (CTAs).

    Emits a `mega.bulk_sync` op. Each program atomically increments the arrival
    counter at `arrival_ptr`; the last arriver resets it and publishes `sense`
    to the release flag at `release_ptr`, while the others spin until they
    observe `sense`. See the `mega-bulk-sync-lowering` pass for the lowering.
    """
    _semantic.builder.mega_bulk_sync([
        _to_ir(arrival_ptr, _semantic),
        _to_ir(release_ptr, _semantic),
        _to_ir(num_programs, _semantic),
        _to_ir(sense, _semantic),
    ])
