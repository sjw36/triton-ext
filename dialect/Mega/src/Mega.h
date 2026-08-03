#ifndef TRITON_EXT_LANGUAGE_MEGA_H
#define TRITON_EXT_LANGUAGE_MEGA_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Declare the dialect itself.
#include "MegaDialect.h.inc"

// Declare the dialect operations.
#define GET_OP_CLASSES
#include "Mega.h.inc"

#endif // TRITON_EXT_LANGUAGE_MEGA_H
