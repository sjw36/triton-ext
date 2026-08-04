#ifndef TRITON_EXT_PASS_MEGABULKSYNC
#define TRITON_EXT_PASS_MEGABULKSYNC

include "mlir/Pass/PassBase.td"

def MegaBulkSyncLowering : Pass<"mega-bulk-sync-lowering", "mlir::ModuleOp"> {
  let summary = "Lower mega.bulk_sync into a grid-wide barrier implementation";
  let description = [{
    This pass rewrites each `mega.bulk_sync` operation into an explicit
    grid-wide barrier built from Triton ops, equivalent to the following
    `triton.jit` implementation:

    ```python
    tl.debug_barrier()
    last = tl.atomic_add(arrival_ptr, 1, sem="acq_rel")
    if last == num_programs - 1:
        tl.atomic_xchg(arrival_ptr, 0, sem="release")
        tl.atomic_xchg(release_ptr, sense, sem="release")
    else:
        while tl.atomic_add(release_ptr, 0, sem="acquire") != sense:
            pass
    tl.debug_barrier()
    ```

    `tl.debug_barrier()` lowers to `ttg.barrier`, the atomics lower to
    `tt.atomic_rmw`, and the control flow uses `scf.if` / `scf.while`.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

#endif // TRITON_EXT_PASS_MEGABULKSYNC
