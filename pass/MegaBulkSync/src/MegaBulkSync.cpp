#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This pass lowers `mega.bulk_sync` into an explicit grid-wide barrier built
// from Triton ops, mirroring the reference `_grid_barrier` kernel:
//
//   tl.debug_barrier()
//   last = tl.atomic_add(arrival_ptr, 1, sem="acq_rel")
//   if last == num_programs - 1:
//       tl.atomic_xchg(arrival_ptr, 0, sem="release")
//       tl.atomic_xchg(release_ptr, sense, sem="release")
//   else:
//       while tl.atomic_add(release_ptr, 0, sem="acquire") != sense:
//           pass
//   tl.debug_barrier()
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mega-bulk-sync-lowering"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

/// Matched by mnemonic so this pass does not need to link against the `mega`
/// dialect library; only the operand layout is relied upon.
static constexpr llvm::StringLiteral kBulkSyncName = "mega.bulk_sync";

/// Build a scalar `tt.atomic_rmw` (one element, no mask) and return the loaded
/// old value.
static Value createAtomicRMW(OpBuilder &b, Location loc, tt::RMWOp kind,
                             Value ptr, Value val, tt::MemSemantic sem) {
  return tt::AtomicRMWOp::create(b, loc, val.getType(), kind, ptr, val,
                                 /*mask=*/Value(), sem, tt::MemSyncScope::GPU);
}

static void lowerBulkSync(Operation *op) {
  OpBuilder b(op);
  Location loc = op->getLoc();

  Value arrivalPtr = op->getOperand(0);
  Value releasePtr = op->getOperand(1);
  Value numPrograms = op->getOperand(2);
  Value sense = op->getOperand(3);

  Value c0 = arith::ConstantIntOp::create(b, loc, 0, 32);
  Value c1 = arith::ConstantIntOp::create(b, loc, 1, 32);

  // tl.debug_barrier()
  ttg::BarrierOp::create(b, loc, ttg::AddrSpace::All);

  // last = tl.atomic_add(arrival_ptr, 1, sem="acq_rel")
  Value last = createAtomicRMW(b, loc, tt::RMWOp::ADD, arrivalPtr, c1,
                               tt::MemSemantic::ACQUIRE_RELEASE);

  // last == num_programs - 1
  Value lastIdx = arith::SubIOp::create(b, loc, numPrograms, c1);
  Value isLast =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, last, lastIdx);

  auto ifOp = scf::IfOp::create(b, loc, isLast, /*withElseRegion=*/true);

  // then: reset the arrival counter and publish `sense` to the release flag.
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(ifOp.thenBlock());
    createAtomicRMW(b, loc, tt::RMWOp::XCHG, arrivalPtr, c0,
                    tt::MemSemantic::RELEASE);
    createAtomicRMW(b, loc, tt::RMWOp::XCHG, releasePtr, sense,
                    tt::MemSemantic::RELEASE);
  }

  // else: spin until the release flag is observed to equal `sense`.
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(ifOp.elseBlock());
    auto whileOp = scf::WhileOp::create(b, loc, TypeRange{}, ValueRange{});

    Block *before = b.createBlock(&whileOp.getBefore());
    b.setInsertionPointToStart(before);
    Value cur = createAtomicRMW(b, loc, tt::RMWOp::ADD, releasePtr, c0,
                                tt::MemSemantic::ACQUIRE);
    Value notReleased =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, cur, sense);
    scf::ConditionOp::create(b, loc, notReleased, ValueRange{});

    Block *after = b.createBlock(&whileOp.getAfter());
    b.setInsertionPointToStart(after);
    scf::YieldOp::create(b, loc);
  }

  // tl.debug_barrier()
  ttg::BarrierOp::create(b, loc, ttg::AddrSpace::All);

  op->erase();
}

// To make available the auto-generated base classes in the `impl` namespace, we
// drop in the generated headers from `Passes.td`.
#define GEN_PASS_DEF_MEGABULKSYNCLOWERING
#include "Passes.h.inc"

struct MegaBulkSyncLoweringPass
    : public impl::MegaBulkSyncLoweringBase<MegaBulkSyncLoweringPass> {
  MegaBulkSyncLoweringPass() = default;

  void runOnOperation() override {
    SmallVector<Operation *> targets;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() == kBulkSyncName) {
        if (op->getNumOperands() == 4)
          targets.push_back(op);
        else
          op->emitWarning("mega.bulk_sync expects 4 operands; skipping");
      }
    });

    for (Operation *op : targets)
      lowerBulkSync(op);
  }
};

} // namespace

// Include the MLIR pass plugin registry implementation.
#include "ExportPass.cpp"
