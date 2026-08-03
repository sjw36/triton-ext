#ifndef TRITON_EXT_PASS_HOISTFIRSTACCESS
#define TRITON_EXT_PASS_HOISTFIRSTACCESS

include "mlir/Pass/PassBase.td"

def MegaHoistFirstAccess : Pass<"mega-hoist-first-access", "mlir::ModuleOp"> {
  let summary = "Hoist first-access tensor loads above a mega.bulk_sync";
  let description = [{
    A grid-wide barrier (`mega.bulk_sync`) makes every cooperating program
    spin until all of them have arrived. That spin latency is otherwise
    wasted: while a program waits, it could already be fetching tensors it
    will need *after* the barrier.

    This pass performs that prefetch. For each `tt.load` that is dominated by
    a `mega.bulk_sync` in the same block, it hoists the load (together with
    the pure address-computation ops it depends on) to just before the
    barrier -- but only when the load is the *first access* of its tensor in
    the kernel.

    "First access" means no earlier load/store/atomic in the function touches
    the same base pointer. When that holds, the tensor is a kernel input (it
    is not produced by an earlier phase of this kernel), so reading it before
    the barrier observes the same value while overlapping the load with the
    barrier's wait. A load whose tensor *is* written before the barrier (the
    classic producer/consumer reason the barrier exists) is never hoisted.

    The pass only reorders existing ops within a block; it never crosses a
    region boundary, so loop trip counts are preserved. It is intended to run
    before `mega-bulk-sync-lowering`, while the barrier is still a single op.
  }];
}

#endif // TRITON_EXT_PASS_HOISTFIRSTACCESS
