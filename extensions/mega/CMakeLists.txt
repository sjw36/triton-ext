# ---------------------------------------------------------------------------
# mega — Triton plugin for the out-of-tree `mega` MLIR dialect and everything
# inseparable from it: the `mega-bulk-sync-lowering` pass that lowers
# `mega.bulk_sync`, and the `tlM` Python DSL that emits it.
#
# Builds a single self-contained `libmega.so`, bundled into the `triton_mega`
# wheel and loaded on import (see python/triton_mega/__init__.py), which
# registers the dialect, the `mega_bulk_sync` TritonOpBuilder method, and the
# lowering pass in one go. `tlM` is pure Python and ships in the same wheel.
#
# The first-access hoist is a separate extension: it is an optimization over
# this dialect rather than part of it (see pass/HoistFirstAccess).
#
# Driven by scikit-build-core (`pip install .`), or configured directly:
#
#   cmake -S extensions/mega -B extensions/mega/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build extensions/mega/build
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(mega LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
configure_triton_extension()

# Extension-specific include: source headers, e.g. `#include "Mega.h"`.
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
# TableGen: Mega dialect declarations and definitions. The dialect defines no
# types or attributes, so only the dialect and op generators run.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS src/Mega.td)
mlir_tablegen(MegaDialect.h.inc   -gen-dialect-decls -dialect=mega)
mlir_tablegen(MegaDialect.cpp.inc -gen-dialect-defs  -dialect=mega)
mlir_tablegen(Mega.h.inc          -gen-op-decls)
mlir_tablegen(Mega.cpp.inc        -gen-op-defs)
add_public_tablegen_target(MLIRMegaIncGen)
add_dependencies(mlir-generic-headers MLIRMegaIncGen)

# ---------------------------------------------------------------------------
# TableGen: MegaBulkSync pass declarations → Passes.h.inc.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS src/Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(MLIRMegaBulkSyncIncGen)
add_dependencies(mlir-generic-headers MLIRMegaBulkSyncIncGen)

# ---------------------------------------------------------------------------
# Plugin library. src/Mega.cpp includes support/ExportDialect.cpp to register
# the dialect and calls support::exportOp() to register the builder method;
# src/MegaBulkSync.cpp includes support/ExportPass.cpp to register the pass;
# support/Export.cpp implements tritonGetPluginInfo() over all three registries.
# ---------------------------------------------------------------------------
add_library(${TRITON_EXT_NAME} SHARED
    src/Mega.cpp
    src/MegaBulkSync.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp")
add_dependencies(${TRITON_EXT_NAME} MLIRMegaIncGen MLIRMegaBulkSyncIncGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}")

# TRITON_EXT_NAME and TRITON_EXT_CLASS are per-translation-unit, not
# per-target: support/ExportDialect.cpp and support/ExportPass.cpp each
# register exactly one entity, keyed on TOSTRING(TRITON_EXT_NAME) and
# instantiating TRITON_EXT_CLASS. A plugin exporting both a dialect and a pass
# therefore sets them per source file rather than on the target. Both names are
# API — TRITON_EXT_NAME is what the pass is called from Python
# (`passes.plugin.add_mega_bulk_sync`) and what the dialect is registered under
# — so they keep the values they had while these were two extensions.
set_source_files_properties(src/Mega.cpp PROPERTIES COMPILE_DEFINITIONS
    "TRITON_EXT_NAME=mega;TRITON_EXT_CLASS=MegaDialect")
set_source_files_properties(src/MegaBulkSync.cpp PROPERTIES COMPILE_DEFINITIONS
    "TRITON_EXT_NAME=mega_bulk_sync;TRITON_EXT_CLASS=MegaBulkSyncLoweringPass")

# Shared compile flags + link-only-against-libtriton (single source of truth).
target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION triton_mega)
