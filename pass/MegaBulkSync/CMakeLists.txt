# ---------------------------------------------------------------------------
# MegaBulkSync — standalone Triton plugin for the bulk-sync lowering pass.
#
# Builds a single self-contained `libmega_bulk_sync.so`, bundled into the
# `triton_mega_bulk_sync` wheel and loaded on import (see
# python/triton_mega_bulk_sync/__init__.py). It provides the
# `mega-bulk-sync-lowering` pass and registers it with Triton's pass pipeline.
#
# The pass matches `mega.bulk_sync` by mnemonic, so it does not link against
# the `mega` dialect plugin; parsing IR that contains the op still requires
# `triton_mega` to be imported first.
#
# Driven by scikit-build-core (`pip install .`), or configured directly:
#
#   cmake -S pass/MegaBulkSync -B pass/MegaBulkSync/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build pass/MegaBulkSync/build
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(mega_bulk_sync LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
configure_triton_extension()

# ---------------------------------------------------------------------------
# TableGen: MegaBulkSync pass declarations → Passes.h.inc.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(MLIRMegaBulkSyncIncGen)
add_dependencies(mlir-generic-headers MLIRMegaBulkSyncIncGen)

# ---------------------------------------------------------------------------
# Plugin library. MegaBulkSync.cpp includes support/ExportPass.cpp to register
# the pass; support/Export.cpp implements tritonGetPluginInfo().
# ---------------------------------------------------------------------------
add_library(${TRITON_EXT_NAME} SHARED
    src/MegaBulkSync.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp")
add_dependencies(${TRITON_EXT_NAME} MLIRMegaBulkSyncIncGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}"
    TRITON_EXT_NAME=${TRITON_EXT_NAME}
    TRITON_EXT_CLASS=MegaBulkSyncLoweringPass)

# Shared compile flags + link-only-against-libtriton (single source of truth).
target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION triton_mega_bulk_sync)
