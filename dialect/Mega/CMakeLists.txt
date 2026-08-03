# ---------------------------------------------------------------------------
# Mega — standalone Triton plugin for the out-of-tree `mega` MLIR dialect.
#
# Builds a single self-contained `libmega.so`, bundled into the `triton_mega`
# wheel and loaded on import (see python/triton_mega/__init__.py). It provides
# the `mega` dialect (currently `mega.bulk_sync`) and the `mega_bulk_sync`
# TritonOpBuilder method that the `tlM` language extension emits through.
#
# Driven by scikit-build-core (`pip install .`), or configured directly:
#
#   cmake -S dialect/Mega -B dialect/Mega/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build dialect/Mega/build
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
# Plugin library. src/Mega.cpp includes support/ExportDialect.cpp to register
# the dialect and calls support::exportOp() to register the builder method;
# support/Export.cpp implements tritonGetPluginInfo().
# ---------------------------------------------------------------------------
add_library(${TRITON_EXT_NAME} SHARED
    src/Mega.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp")
add_dependencies(${TRITON_EXT_NAME} MLIRMegaIncGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}"
    TRITON_EXT_NAME=${TRITON_EXT_NAME}
    TRITON_EXT_CLASS=MegaDialect)

# Shared compile flags + link-only-against-libtriton (single source of truth).
target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME} LIBRARY DESTINATION triton_mega)
