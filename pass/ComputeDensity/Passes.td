#ifndef TRITON_EXT_PASS_COMPUTEDENSITY
#define TRITON_EXT_PASS_COMPUTEDENSITY

include "mlir/Pass/PassBase.td"

def TritonComputeDensity : Pass</*cli-arg*/"triton-compute-density", /*Op*/"mlir::ModuleOp"> {
  let summary = "Calculate the algorithmic compute density";
  let description = [{
    This pass will calculate the algorithmic compute density of the kernel and set an attribute
    for the equation for each parameter that returns a value. This includes an equation for `compute`
    and `bandwidth` per CTA.
  }];
}

#endif // TRITON_EXT_PASS_COMPUTEDENSITY
