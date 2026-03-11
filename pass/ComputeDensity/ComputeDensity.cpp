#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/StrUtil.h"

#define DEBUG_TYPE "triton-compute-density"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONCOMPUTEDENSITY
#include "Passes.h.inc"
} // namespace mlir::triton

namespace {

enum ExprType {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
};

std::string makeExpr(ExprType type, const SmallVector<std::string> &exprs) {
  std::string op;
  int cnt = 0;
  for (auto expr : exprs) {
    if (expr == "0") {
      if (type == ExprType::Add) {
        continue;
      } else if (type == ExprType::Sub) {
        if (cnt != 0) {
          continue;
        }
      } else if (type == ExprType::Mul) {
        return "0";
      } else if (type == ExprType::Div) {
        assert(cnt == 0 && "Division by 0");
        return "1";
      } else if (type == ExprType::Rem) {
        assert(cnt == 0 && "Remainder by 0");
        return "0";
      }
    } else if (expr == "1") {
      if (type == ExprType::Div && cnt != 0) {
        continue;
      }
    }
    if (cnt > 0) {
      if (type == ExprType::Add) {
        op += " + ";
      } else if (type == ExprType::Sub) {
        op += " - ";
      } else if (type == ExprType::Mul) {
        op += " * ";
      } else if (type == ExprType::Div) {
        op += " / ";
      } else if (type == ExprType::Rem) {
        op += " % ";
      }
    }
    op += expr;
    cnt++;
  }
  if (cnt == 0) {
    return "0";
  }
  if (cnt > 1) {
    return "(" + op + ")";
  }
  return op;
}

// Compute Density Analysis Driver
// 1) Process every operation in every block
//    - collect load sizes for input bandwidth
//    - collect store sizes for output bandwidth
//    - collect arithmetic sizes for compute ops
//      - only count arithmetic ops that are stored to memory
// 2) Bottom up collect metrics and aggregate them to the function level
//    - for loop: multiply by loop iterations
//    - if/else: max CD of all branch
// 3) Annotate the function with the compute density (bandwidth and compute)
//    - add a new attribute to the function arguments

////////////////////////////////////////////////////////////////////////////////
// Metric class
////////////////////////////////////////////////////////////////////////////////
class Metric {
public:
  enum MetricKind {
    Load,
    Store,
    Compute,
    Other,
  };

  Metric(MetricKind kind = MetricKind::Other, int64_t size = 0,
         Type elementType = nullptr)
      : kind(kind), size(size), elementType(elementType) {}

  MetricKind getKind() const { return kind; }
  int64_t getSize() const { return size; }
  Type getElementType() const { return elementType; }

  void addSize(int64_t sz) { size += sz; }
  void setElementType(Type etype) { elementType = etype; }

  std::string getKindString() const {
    switch (kind) {
    case MetricKind::Load:
      return "Load";
    case MetricKind::Store:
      return "Store";
    case MetricKind::Compute:
      return "Compute";
    }
    return "Other";
  }

private:
  MetricKind kind;
  int64_t size;
  Type elementType;
};

////////////////////////////////////////////////////////////////////////////////
// BlockMetrics class
////////////////////////////////////////////////////////////////////////////////
class BlockMetrics {

  bool isLoadLikeOp(Operation *op) const {
    return isa<triton::LoadOp, triton::DescriptorLoadOp>(op);
  }

  bool isStoreLikeOp(Operation *op) const {
    return isa<triton::StoreOp, triton::DescriptorStoreOp>(op);
  }

  int64_t getNumElements(Type type) const {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
      auto elementSize = getNumElements(rankedType.getElementType());
      return rankedType.getNumElements() * elementSize;
    } else if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      return 1; // elements of pointee?
    } else if (auto tensorType = dyn_cast<triton::TensorDescType>(type)) {
      return getNumElements(tensorType.getBlockType());
    } else if (auto vectorType = dyn_cast<VectorType>(type)) {
      auto elementSize = getNumElements(vectorType.getElementType());
      return vectorType.getNumElements() * elementSize;
    }
    return 1;
  }

  Type getElementType(Type type) const {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
      return getElementType(rankedType.getElementType());
    } else if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      // should be int64_t for pointer
      return getElementType(ptrType.getPointeeType());
    } else if (auto vectorType = dyn_cast<VectorType>(type)) {
      return getElementType(vectorType.getElementType());
    } else if (auto tensorType = dyn_cast<triton::TensorDescType>(type)) {
      return getElementType(tensorType.getBlockType());
    }
    return type;
  }

  Metric calculateMetric(Value value) {
    auto *op = value.getDefiningOp();
    if (isLoadLikeOp(op)) {
      auto type = op->getResult(0).getType();
      return Metric(Metric::MetricKind::Load, getNumElements(type),
                    getElementType(type));
    } else if (isStoreLikeOp(op)) {
      auto type = op->getOperand(1).getType();
      return Metric(Metric::MetricKind::Store, getNumElements(type),
                    getElementType(type));
    } else if (isa<scf::YieldOp>(op)) {
      return Metric();
    } else if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // FLOPS = M * N * K * 2
      auto aType = cast<RankedTensorType>(dotOp.getA().getType());
      auto K = aType.getShape().back();
      auto cSize = getNumElements(dotOp.getC().getType());
      auto flops = cSize * K * 2;
      return Metric(Metric::MetricKind::Compute, flops,
                    getElementType(value.getType()));
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      int64_t flops = 0;
      for (auto inputTy : reduceOp.getInputTypes()) {
        flops += getNumElements(inputTy);
      }
      return Metric(Metric::MetricKind::Compute, flops,
                    getElementType(value.getType()));
    } else if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
      auto type = addPtrOp.getOffset().getType();
      return Metric(Metric::MetricKind::Compute, getNumElements(type),
                    getElementType(type));
    } else if (isa<triton::SplatOp, triton::BroadcastOp, triton::MakeRangeOp,
                   triton::ExpandDimsOp, triton::GetProgramIdOp,
                   triton::TransOp, triton::ReshapeOp>(op)) {
      return Metric(Metric::MetricKind::Compute, 0,
                    getElementType(value.getType()));
    } else if (isa<arith::SelectOp>(op)) {
      return Metric(Metric::MetricKind::Compute);
    } else if (op->hasTrait<OpTrait::Elementwise>()) {
      auto flops = getNumElements(value.getType());
      return Metric(Metric::MetricKind::Compute, flops,
                    getElementType(value.getType()));
    } else if (isa<arith::ConstantOp>(op)) {
      return Metric();
    } else if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op)) {
      return Metric();
    } else {
      LDBG("Value is not a dot or elementwise operation: " << value);
      auto flops = getNumElements(value.getType());
      return Metric(Metric::MetricKind::Compute, flops,
                    getElementType(value.getType()));
    }
  }

public:
  BlockMetrics(Block *block) : block(block) {
    for (auto &op : *block) {
      for (auto result : op.getResults()) {
        metricsMap.try_emplace(result, calculateMetric(result));
      }
      if (isLoadLikeOp(&op)) {
        loadOps.push_back(&op);
      } else if (isStoreLikeOp(&op)) {
        storeOps.push_back(&op);
      }
    }
  }

  std::optional<Metric> getMetric(Value value) const {
    auto it = metricsMap.find(value);
    if (it != metricsMap.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  const SmallVector<Operation *> &getLoadOps() const { return loadOps; }
  const SmallVector<Operation *> &getStoreOps() const { return storeOps; }

  Metric calculateChainMetric(Value value, DenseSet<Value> &visited,
                              SmallVector<Value> &edges) const {
    if (visited.contains(value)) {
      return Metric();
    }
    visited.insert(value);
    auto mval = getMetric(value);
    if (!mval || mval->getKind() != Metric::MetricKind::Compute) {
      edges.push_back(value);
      return Metric();
    }
    Metric totalMetric = *mval;
    for (auto operand : value.getDefiningOp()->getOperands()) {
      totalMetric.addSize(
          calculateChainMetric(operand, visited, edges).getSize());
    }
    return totalMetric;
  }

  void dump() const {
    llvm::errs() << "Block: ----------------------------------------------\n";
    llvm::errs() << "Block: " << *block << "\n";
    llvm::errs() << "Load Ops: " << loadOps.size() << "\n";
    llvm::errs() << "Store Ops: " << storeOps.size() << "\n";
    for (auto &metric : metricsMap) {
      llvm::errs() << "Metric: type= " << metric.second.getKindString()
                   << ", size= " << metric.second.getSize()
                   << ", elementType= " << metric.second.getElementType()
                   << "\n";
      if (metric.first.getDefiningOp()->getNumRegions() > 0) {
        llvm::errs() << "  - Value: " << metric.first.getDefiningOp()->getName()
                     << "\n";
      } else {
        llvm::errs() << "  - Value: " << metric.first << "\n";
      }
    }
  }

private:
  Block *block;
  SmallVector<Operation *> loadOps;
  SmallVector<Operation *> storeOps;
  Operation *yieldOp;
  DenseMap<Value, Metric> metricsMap;
  SmallVector<Metric> resultMetrics;
};

////////////////////////////////////////////////////////////////////////////////
// ComputeDensityAnalysisDriver class
////////////////////////////////////////////////////////////////////////////////
class ComputeDensityAnalysisDriver {

  BlockArgument findPointerParam(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (auto funcOp = dyn_cast<triton::FuncOp>(parentOp)) {
        assert(funcOp == func && "Expected function argument");
        if (isa<triton::PointerType>(blockArg.getType()) ||
            isa<triton::TensorDescType>(blockArg.getType())) {
          return blockArg;
        }
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        return findPointerParam(
            forOp.getInitArgs()[blockArg.getArgNumber() -
                                forOp.getNumInductionVars()]);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        assert(false && "Not implemented");
        // return findPointerParam(ifOp.getOperand(blockArg.getArgNumber() +
        // 1));
      } else {
        LDBG("Unsupported operation: " << parentOp->getName());
      }
    } else { // assert not scf::for, scf::if, scf::while, or func op
      auto defOp = value.getDefiningOp();
      for (auto operand : defOp->getOperands()) {
        auto blockArg = findPointerParam(operand);
        if (blockArg) {
          return blockArg;
        }
      }
    }
    return BlockArgument();
  }

  std::string getSymbolicValue(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (auto funcOp = dyn_cast<triton::FuncOp>(parentOp)) {
        return "args[" + std::to_string(blockArg.getArgNumber()) + "]";
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        assert(
            false &&
            "Loop iterations derived from loop carried values not implemented");
        auto argNumber = blockArg.getArgNumber() - forOp.getNumInductionVars();
        assert(argNumber >= 0 && argNumber < forOp.getInitArgs().size() &&
               "Cannot be dependent on induction variable");
        return getSymbolicValue(forOp.getInitArgs()[argNumber]);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        assert(false && "Not implemented");
        // return getSymbolicValue(ifOp.getCondition()) + " ? " +
        // getSymbolicValue(ifOp.getThenBlock().getTerminator()) + " : " +
        // getSymbolicValue(ifOp.getElseBlock().getTerminator());
      }
    } else if (auto constant =
                   dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
      auto value = cast<IntegerAttr>(constant.getValueAttr());
      return std::to_string(value.getInt());
    } else if (auto programIdOp =
                   dyn_cast<triton::GetProgramIdOp>(value.getDefiningOp())) {
      return "program_id[" + std::to_string(programIdOp.getAxisAsInt()) + "]";
    } else if (auto numProgramsOp =
                   dyn_cast<triton::GetNumProgramsOp>(value.getDefiningOp())) {
      return "num_programs[" + std::to_string(numProgramsOp.getAxisAsInt()) +
             "]";
    } else {
      SmallVector<std::string> operands;
      for (auto operand : value.getDefiningOp()->getOperands()) {
        operands.push_back(getSymbolicValue(operand));
      }
      auto *op = value.getDefiningOp();
      if (isa<arith::AddIOp>(op)) {
        return makeExpr(ExprType::Add, operands);
      } else if (isa<arith::SubIOp>(op)) {
        return makeExpr(ExprType::Sub, operands);
      } else if (isa<arith::MulIOp>(op)) {
        return makeExpr(ExprType::Mul, operands);
      } else if (isa<arith::DivSIOp>(op)) {
        return makeExpr(ExprType::Div, operands);
      } else if (isa<arith::DivUIOp>(op)) {
        return makeExpr(ExprType::Div, operands);
      } else if (isa<arith::RemSIOp>(op)) {
        return makeExpr(ExprType::Rem, operands);
      } else if (isa<arith::RemUIOp>(op)) {
        return makeExpr(ExprType::Rem, operands);
      }
    }
    // llvm::unreachable("Unsupported operation");
    assert(false && "Unsupported operation");
    return "";
  }

  std::string getSymbolicIterations(scf::ForOp forOp) {
    auto upperBound = getSymbolicValue(forOp.getUpperBound());
    auto lowerBound = getSymbolicValue(forOp.getLowerBound());
    auto step = getSymbolicValue(forOp.getStep());
    return makeExpr(ExprType::Div,
                    {makeExpr(ExprType::Sub, {upperBound, lowerBound}), step});
  }

  std::string calculateBandwidth(Operation *op, std::string symbolicSize) {
    auto parentOp = op->getParentOp();
    if (isa<FunctionOpInterface>(parentOp)) {
      return symbolicSize;
    } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      auto numIterations = getSymbolicIterations(forOp);
      symbolicSize = makeExpr(ExprType::Mul, {numIterations, symbolicSize});
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
      assert(false && "Not implemented");
      // auto thenSize = calculateBandwidth(ifOp.getThenBlock(), symbolicSize);
      // auto elseSize = calculateBandwidth(ifOp.getElseBlock(), symbolicSize);
      // symbolicSize = "max(" + thenSize + ", " + elseSize + ")";
    } else {
      LDBG("Unsupported operation: " << parentOp->getName());
      assert(false && "Unsupported operation");
      return "";
    }
    return calculateBandwidth(parentOp, symbolicSize);
  }

  std::string calculateCompute(Value value, DenseSet<Value> &visited,
                               SmallVector<Value> &edges) {
    std::string computeSize;
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      if (auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        int idx = blockArg.getArgNumber() - forOp.getNumInductionVars();
        if (idx >= 0) {
          auto initArg = forOp.getInitArgs()[blockArg.getArgNumber() -
                                             forOp.getNumInductionVars()];
          edges.push_back(initArg);
        } // else is loop carried value
      } else {
        assert(isa<FunctionOpInterface>(blockArg.getOwner()->getParentOp()) &&
               "Expected function argument");
        // edges.push_back(blockArg);
      }
      return computeSize;
    }

    auto defOp = value.getDefiningOp();
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      int idx =
          llvm::find(forOp.getResults(), value) - forOp.getResults().begin();
      auto yieldOp = forOp.getBody()->getTerminator();
      auto yieldValue = yieldOp->getOperand(idx);
      edges.push_back(yieldValue);
      return computeSize;
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(defOp)) {
      // Treat like normal compute op
    } else if (defOp->getNumRegions() > 0) {
      assert(false && "Not implemented");
    }

    auto &blockMetrics = metrics.at(defOp->getBlock());
    auto mval = blockMetrics.calculateChainMetric(value, visited, edges);
    return std::to_string(mval.getSize());
  }

  std::string calculateCompute(Value value, DenseSet<Value> &visited) {
    SmallVector<Value> edges;
    auto computeSize = calculateCompute(value, visited, edges);
    auto *valueOp = value.getDefiningOp();
    if (!computeSize.empty() && valueOp != nullptr) {
      computeSize = calculateBandwidth(valueOp, computeSize);
    }
    for (auto edge : edges) {
      auto edgeMetric = calculateCompute(edge, visited);
      if (!edgeMetric.empty()) {
        if (computeSize.empty()) {
          computeSize = edgeMetric;
        } else {
          computeSize = makeExpr(ExprType::Add, {edgeMetric, computeSize});
        }
      }
    }
    return computeSize;
  }

public:
  ComputeDensityAnalysisDriver(triton::FuncOp func)
      : func(func), bandwidthMetrics(func.getNumArguments()),
        computeMetrics(func.getNumArguments()) {}

  void run() {
    // Build metrics map
    func.walk<WalkOrder::PostOrder>(
        [&](Block *block) { metrics.try_emplace(block, block); });

    auto updateMetric = [&](std::string &metric, std::string symbolicSize) {
      if (metric.empty()) {
        metric = symbolicSize;
      } else {
        metric = makeExpr(ExprType::Add, {metric, symbolicSize});
      }
    };

    // Collect parameter metrics
    for (auto &[block, blockMetrics] : metrics) {
      // 1) Calculate load ops BW
      for (auto &loadOp : blockMetrics.getLoadOps()) {
        auto param = findPointerParam(loadOp->getOperand(0));
        auto metric = blockMetrics.getMetric(loadOp->getResult(0));
        if (metric) {
          auto symbolicSize =
              calculateBandwidth(loadOp, std::to_string(metric->getSize()));
          updateMetric(bandwidthMetrics[param.getArgNumber()], symbolicSize);
          // assert(getType == metric.getElementType())
        }
      }
      // 2) Calculate store ops BW
      // 3) Calculate store value Compute ops, only for values that are stored
      // to memory
      for (auto &storeOp : blockMetrics.getStoreOps()) {
        auto param = findPointerParam(storeOp->getOperand(0));
        auto storeValue = storeOp->getOperand(1);
        auto metric = blockMetrics.getMetric(storeValue);
        if (metric) {
          auto symbolicSize =
              calculateBandwidth(storeOp, std::to_string(metric->getSize()));
          updateMetric(bandwidthMetrics[param.getArgNumber()], symbolicSize);
          assert(computeMetrics[param.getArgNumber()].empty());
          DenseSet<Value> visited;
          auto computeSize = calculateCompute(storeValue, visited);
          if (!computeSize.empty()) {
            updateMetric(computeMetrics[param.getArgNumber()], computeSize);
          }
        }
      }
    }
    LLVM_DEBUG(dump());
  }

  std::optional<std::string> getBandwidthMetric(unsigned index) const {
    if (index >= bandwidthMetrics.size() || bandwidthMetrics[index].empty()) {
      return std::nullopt;
    }
    return bandwidthMetrics[index];
  }
  std::optional<std::string> getComputeMetric(unsigned index) const {
    if (index >= computeMetrics.size() || computeMetrics[index].empty()) {
      return std::nullopt;
    }
    return computeMetrics[index];
  }

  void dump() {
    llvm::errs() << "Compute Density Analysis Driver: "
                    "----------------------------------------------\n";
    llvm::errs() << "Function: " << func.getName() << "\n";
    for (auto [block, blockMetrics] : metrics) {
      blockMetrics.dump();
    }
    llvm::errs() << "Bandwidth Metrics: " << bandwidthMetrics.size() << "\n";
    for (int i = 0; i < func.getNumArguments(); i++) {
      llvm::errs() << "Bandwidth Metric: index= " << i
                   << ", size= " << bandwidthMetrics[i] << "\n";
    }
    llvm::errs() << "Compute Metrics: " << computeMetrics.size() << "\n";
    for (int i = 0; i < func.getNumArguments(); i++) {
      llvm::errs() << "Compute Metric: index= " << i
                   << ", size= " << computeMetrics[i] << "\n";
    }
  }

private:
  triton::FuncOp func;
  DenseMap<Block *, BlockMetrics> metrics;
  SmallVector<std::string> bandwidthMetrics;
  SmallVector<std::string> computeMetrics;
};

////////////////////////////////////////////////////////////////////////////////
// Pass ComputeDensity
////////////////////////////////////////////////////////////////////////////////
struct ComputeDensityPass
    : public triton::impl::TritonComputeDensityBase<
          ComputeDensityPass> {
  using TritonComputeDensityBase::TritonComputeDensityBase;

  // TODO: get callgraph (see Analysis/Allocation.h)
  void runOnOperation() override {
    for (auto func : getOperation().getOps<triton::FuncOp>()) {
      ComputeDensityAnalysisDriver driver(func);
      driver.run();
      // Apply to function parameters
      for (unsigned i = 0; i < func.getNumArguments(); i++) {
        if (auto bandwidth = driver.getBandwidthMetric(i)) {
          func.setArgAttr(
              i, "tt.bandwidth",
              StringAttr::get(func.getContext(), bandwidth.value()));
        }
        if (auto compute = driver.getComputeMetric(i)) {
          func.setArgAttr(i, "tt.compute",
                          StringAttr::get(func.getContext(), compute.value()));
        }
      }
    }
  }
};

} // namespace

// Include the MLIR pass plugin registry implementation
#include "TritonExtPass.cpp"
