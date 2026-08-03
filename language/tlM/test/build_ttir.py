#!/usr/bin/env python3
"""
Build TTIR for a kernel that calls `tlM.bulk_sync` and print it to stdout.

Stops at TTIR (the frontend's output) rather than running a full
`triton.compile`: `mega.bulk_sync` is lowered by the out-of-tree
`mega-bulk-sync-lowering` pass, not by the standard TTIR->TTGIR pipeline, so
there is no GPU code-gen path for it. Building TTIR with a stub target needs no
GPU, mirroring Triton's own `triton._filecheck` frontend tests.

Requires the `triton-tlM` and `triton-mega` wheels to be installed
(`make build install`); importing `tlM` pulls in `triton_mega`, which
registers the dialect and the builder method.

Usage:
    python build_ttir.py <kernel>
"""

import sys

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend

# Importing `tlM` imports `triton_mega`, which registers the `mega` dialect
# and the `create_mega_bulk_sync` builder method, then registers itself as
# `triton.language.extra.tlM` — the same module object, so kernels below can
# call it under either name.
import tlM

# A stub target lets us build TTIR via the frontend without a real GPU.
STUB_TARGET = GPUTarget("cuda", 100, 32)

SIGNATURE = {
    "arrival_ptr": "*i32",
    "release_ptr": "*i32",
    "sense": "i32",
}


@triton.jit
def straight_line(arrival_ptr, release_ptr, sense):
    tlM.bulk_sync(arrival_ptr, release_ptr, tl.num_programs(0), sense)


@triton.jit
def in_branch(arrival_ptr, release_ptr, sense):
    pid = tl.program_id(0)
    if pid == 0:
        tlM.bulk_sync(arrival_ptr, release_ptr, tl.num_programs(0), sense)


KERNELS = {"straight_line": straight_line, "in_branch": in_branch}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in KERNELS:
        print(f"usage: {sys.argv[0]} <{'|'.join(KERNELS)}>", file=sys.stderr)
        return 2

    backend = make_backend(STUB_TARGET)
    src = ASTSource(fn=KERNELS[sys.argv[1]],
                    signature=SIGNATURE,
                    constexprs={})
    options = backend.parse_options({"sanitize_overflow": False})

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    module = src.make_ir(STUB_TARGET, options,
                         backend.get_codegen_implementation(options),
                         backend.get_module_map(), context)
    print(module.str_nodebug(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
