#ifndef TRITON_EXT_LANGUAGE_MEGA_DIALECT
#define TRITON_EXT_LANGUAGE_MEGA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Mega_Dialect : Dialect {
    let name = "mega";
    let summary = "An out-of-tree MLIR dialect for bulk/collective primitives.";
    let description = [{
        The `mega` dialect is an out-of-tree Triton extension dialect that
        models coarse-grained ("mega") collective primitives, such as bulk
        asynchronous transfers and their synchronization points.
    }];
    let cppNamespace = "::mlir::triton::mega";
}

class Mega_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mega_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def Mega_BulkSyncOp : Mega_Op<"bulk_sync", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Grid-wide barrier across cooperating programs.";
    let description = [{
        The `mega.bulk_sync` operation implements a grid-wide ("bulk") barrier
        across all cooperating programs (CTAs). Each program atomically
        increments the shared arrival counter at `$arrival_ptr`; the last
        program to arrive (when the counter reaches `$num_programs - 1`) resets
        the counter and publishes `$sense` to the shared release flag at
        `$release_ptr`, while every other program spins until it observes the
        new `$sense` value.

        The op carries side effects and is intended to be lowered by the
        `mega-bulk-sync-lowering` pass into the following implementation:

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

        ```mlir
        mega.bulk_sync %arrival, %release, %num_programs, %sense
            : !tt.ptr<i32>, !tt.ptr<i32>
        ```
    }];
    let arguments = (ins
        AnyType:$arrival_ptr,
        AnyType:$release_ptr,
        I32:$num_programs,
        I32:$sense
    );
    let results = (outs);
    let assemblyFormat = [{
        $arrival_ptr `,` $release_ptr `,` $num_programs `,` $sense attr-dict
        `:` type($arrival_ptr) `,` type($release_ptr)
    }];
}

#endif // TRITON_EXT_LANGUAGE_MEGA_DIALECT
