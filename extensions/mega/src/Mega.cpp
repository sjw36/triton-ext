#include "Mega.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Define the dialect itself; we need to define how it initializes.
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Dialect/Triton/IR/Types.h"

// For registering a custom TritonOpBuilder method (see below).
#include "Export.h"

#include "MegaDialect.cpp.inc"

namespace mlir::triton::mega {

void MegaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mega.cpp.inc"
      >();
}

} // namespace mlir::triton::mega

// Define the dialect operations.
#define GET_OP_CLASSES
#include "Mega.cpp.inc"

// ---------------------------------------------------------------------------
// Python frontend hook: a custom TritonOpBuilder method `mega_bulk_sync`.
//
// The `tlM` language extension (python/tlM) calls this from a `@tl.builtin`
// via `_semantic.builder.mega_bulk_sync([...])` to emit `mega.bulk_sync`.
//
// Following the convention used by other plugins (see uTLXPlugin.cpp), Triton
// prepends a result slot at operands[0]; the Python-provided values follow at
// operands[1..]. `mega.bulk_sync` has no results, so operands[0] is unused.
// ---------------------------------------------------------------------------
namespace {
void createBulkSyncBuilderOp(TritonOpBuilder &self,
                             std::vector<mlir::Value> &operands) {
  // operands[1] = arrival_ptr, [2] = release_ptr, [3] = num_programs, [4] =
  // sense
  if (operands.size() < 5)
    return;
  self.create<mlir::triton::mega::BulkSyncOp>(operands[1], operands[2],
                                              operands[3], operands[4]);
}

[[maybe_unused]] const triton::ext::support::Result registerBulkSyncBuilderOp =
    triton::ext::support::exportOp("mega_bulk_sync", createBulkSyncBuilderOp);
} // namespace

// Include the MLIR dialect plugin registry implementation.
using namespace mlir::triton::mega;
#include "ExportDialect.cpp"
