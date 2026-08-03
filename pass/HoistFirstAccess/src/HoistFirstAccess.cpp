#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This pass prefetches loads across a grid-wide barrier. For a `tt.load` or
// `tt.descriptor_load`/`tt.descriptor_gather` that follows a `mega.bulk_sync`
// in the same block, if the load is the *first access* of its tensor anywhere
// in the kernel, the load (and the pure address / descriptor arithmetic it
// needs) is moved to just before the barrier so the memory latency overlaps
// the barrier's spin-wait.
//
//   %v = tt.load %y            %v = tt.load %y
//   tt.store %y, %v            tt.store %y, %v
//   mega.bulk_sync ...   =>    %x0 = tt.load %x      // hoisted (first access)
//   %x0 = tt.load %x           mega.bulk_sync ...
//
// A load of a tensor that was already accessed before the barrier (e.g. the
// data the barrier is there to publish) is left in place.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mega-hoist-first-access"

using namespace mlir;
namespace tt = mlir::triton;

namespace {

/// Matched by mnemonic so this pass does not need to link against the `mega`
/// dialect library.
static constexpr llvm::StringLiteral kBulkSyncName = "mega.bulk_sync";

/// Trace a (tensor of) pointer or tensor-descriptor value back to the root base
/// pointer by walking through the pure ops that preserve the underlying buffer.
/// A `tt.make_tensor_descriptor` is followed to its `base` pointer so that a
/// descriptor access and an ordinary pointer access to the same buffer trace to
/// the same root. Two accesses share "the same tensor" iff this returns the
/// same `Value`.
static Value getBasePointer(Value ptr) {
  while (Operation *def = ptr.getDefiningOp()) {
    if (auto addptr = dyn_cast<tt::AddPtrOp>(def)) {
      ptr = addptr.getPtr();
    } else if (auto desc = dyn_cast<tt::MakeTensorDescOp>(def)) {
      ptr = desc.getBase();
    } else if (isa<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp,
                   tt::BitcastOp>(def)) {
      ptr = def->getOperand(0);
    } else {
      break;
    }
  }
  return ptr;
}

/// Return the traced base of the buffer that `op` reads or writes in global
/// memory, or null if `op` is not an access we track. Covers both pointer-based
/// ops (`tt.load` / `tt.store` / atomics) and TMA descriptor ops
/// (`tt.descriptor_load` / `_store` / `_gather` / `_scatter` / `_reduce`).
static Value getAccessedBase(Operation *op) {
  Value handle;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    handle = load.getPtr();
  else if (auto store = dyn_cast<tt::StoreOp>(op))
    handle = store.getPtr();
  else if (auto rmw = dyn_cast<tt::AtomicRMWOp>(op))
    handle = rmw.getPtr();
  else if (auto cas = dyn_cast<tt::AtomicCASOp>(op))
    handle = cas.getPtr();
  else if (auto dl = dyn_cast<tt::DescriptorLoadOp>(op))
    handle = dl.getDesc();
  else if (auto dg = dyn_cast<tt::DescriptorGatherOp>(op))
    handle = dg.getDesc();
  else if (auto ds = dyn_cast<tt::DescriptorStoreOp>(op))
    handle = ds.getDesc();
  else if (auto dr = dyn_cast<tt::DescriptorReduceOp>(op))
    handle = dr.getDesc();
  else if (auto dsc = dyn_cast<tt::DescriptorScatterOp>(op))
    handle = dsc.getDesc();
  if (!handle)
    return Value();
  return getBasePointer(handle);
}

/// A memory read we can prefetch above the barrier: an ordinary `tt.load` or a
/// TMA descriptor load/gather.
static bool isLoadLike(Operation *op) {
  return isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op);
}

/// Collect, in dependency-first (topological) order, the ops that must move
/// with `load` so it can be placed before `sync`: `load` itself plus any pure
/// producers that currently sit after `sync` in the same block. Returns false
/// (and leaves `slice` undefined) if hoisting is unsafe -- e.g. a required
/// producer has side effects or lives in a different block/region.
static bool collectHoistableSlice(Operation *load, Operation *sync,
                                  const DominanceInfo &dom,
                                  SmallVectorImpl<Operation *> &slice) {
  Block *block = sync->getBlock();
  DenseSet<Operation *> visited;

  std::function<bool(Operation *)> visit = [&](Operation *op) -> bool {
    if (!visited.insert(op).second)
      return true;
    // Everything we move must stay in `sync`'s block so we never change the
    // execution count of an op nested in a loop/conditional.
    if (op->getBlock() != block)
      return false;
    for (Value operand : op->getOperands()) {
      // Already available before the barrier: nothing to move for it.
      if (dom.dominates(operand, sync))
        continue;
      Operation *def = operand.getDefiningOp();
      if (!def)
        return false; // block arg that doesn't dominate the barrier
      // A producer we'd need to move must be pure; otherwise reordering it
      // across the barrier (or other effects) could change behavior.
      if (!isMemoryEffectFree(def))
        return false;
      if (!visit(def))
        return false;
    }
    slice.push_back(op); // post-order => producers precede consumers
    return true;
  };

  return visit(load);
}

// To make available the auto-generated base classes in the `impl` namespace, we
// drop in the generated headers from `Passes.td`.
#define GEN_PASS_DEF_MEGAHOISTFIRSTACCESS
#include "Passes.h.inc"

struct MegaHoistFirstAccessPass
    : public impl::MegaHoistFirstAccessBase<MegaHoistFirstAccessPass> {
  MegaHoistFirstAccessPass() = default;

  void runOnOperation() override {
    getOperation()->walk([&](tt::FuncOp func) { processFunc(func); });
  }

  /// Find the closest `mega.bulk_sync` that precedes `op` in its own block, or
  /// null if there is none.
  static Operation *nearestPrecedingSync(Operation *op) {
    Operation *sync = nullptr;
    for (Operation &other : *op->getBlock()) {
      if (&other == op)
        break;
      if (other.getName().getStringRef() == kBulkSyncName)
        sync = &other;
    }
    return sync;
  }

  void processFunc(tt::FuncOp func) {
    // Pre-order index approximates program order; used to decide which memory
    // accesses happen "before" a candidate load.
    DenseMap<Operation *, unsigned> order;
    SmallVector<std::pair<unsigned, Value>> memAccesses; // (order, base ptr)
    SmallVector<Operation *> candidates;

    unsigned idx = 0;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      order[op] = idx++;
      if (Value base = getAccessedBase(op))
        memAccesses.emplace_back(order[op], base);
      if (isLoadLike(op))
        candidates.push_back(op);
    });

    DominanceInfo dom(func);

    for (Operation *load : candidates) {
      Operation *sync = nearestPrecedingSync(load);
      if (!sync)
        continue; // not after a barrier in its block

      Value base = getAccessedBase(load);
      unsigned loadOrder = order[load];

      // First access: no earlier memory op touches the same tensor.
      bool firstAccess = true;
      for (auto &[accessOrder, accessBase] : memAccesses) {
        if (accessOrder < loadOrder && accessBase == base) {
          firstAccess = false;
          break;
        }
      }
      if (!firstAccess)
        continue;

      SmallVector<Operation *> slice;
      if (!collectHoistableSlice(load, sync, dom, slice))
        continue;

      // Move the slice (producers first, load last) to just before the sync.
      for (Operation *op : slice)
        op->moveBefore(sync);
      LLVM_DEBUG(llvm::dbgs() << "hoisted first-access load above "
                              << kBulkSyncName << "\n");
    }
  }
};

} // namespace

// Include the MLIR pass plugin registry implementation.
#include "ExportPass.cpp"
