# ---------------------------------------------------------------------------
# HoistFirstAccess — standalone Triton plugin for the barrier-prefetch pass.
#
# Builds a single self-contained `libmega_hoist_first_access.so`, bundled into
# the `triton_mega_hoist_first_access` wheel and loaded on import (see
# python/triton_mega_hoist_first_access/__init__.py). It provides the
# `mega-hoist-first-access` pass and registers it with Triton's pass pipeline.
#
# The pass inspects Triton ops (`tt.load` / `tt.addptr` / ...), so it needs
# libtriton's symbols at load time; that is exactly the link recipe
# `triton_ext::plugin` encodes, so no extension-specific link flags are needed.
#
# Driven by scikit-build-core (`pip install .`), or configured directly:
#
#   cmake -S pass/HoistFirstAccess -B pass/HoistFirstAccess/build -G Ninja \
#       -DLLVM_INSTALL_DIR=$(python ci/pick_local_artifact.py 'llvm-*') \
#       -DTRITON_WHEEL_DIR=$(python ci/probe_triton_wheel.py)
#   cmake --build pass/HoistFirstAccess/build
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(mega_hoist_first_access LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/triton-ext.cmake")
configure_triton_extension()

# ---------------------------------------------------------------------------
# TableGen: HoistFirstAccess pass declarations → Passes.h.inc.
# ---------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extension)
add_public_tablegen_target(MLIRMegaHoistFirstAccessIncGen)
add_dependencies(mlir-generic-headers MLIRMegaHoistFirstAccessIncGen)

# ---------------------------------------------------------------------------
# Plugin library. HoistFirstAccess.cpp includes support/ExportPass.cpp to
# register the pass; support/Export.cpp implements tritonGetPluginInfo().
# ---------------------------------------------------------------------------
add_library(${TRITON_EXT_NAME} SHARED
    src/HoistFirstAccess.cpp
    "${TRITON_EXT_SUPPORT_DIR}/Export.cpp")
add_dependencies(${TRITON_EXT_NAME} MLIRMegaHoistFirstAccessIncGen)

target_compile_definitions(${TRITON_EXT_NAME} PRIVATE
    TRITON_EXT_VERSION="${TRITON_EXT_VERSION}"
    TRITON_EXT_NAME=${TRITON_EXT_NAME}
    TRITON_EXT_CLASS=MegaHoistFirstAccessPass)

# Shared compile flags + link-only-against-libtriton (single source of truth).
target_link_libraries(${TRITON_EXT_NAME} PRIVATE triton_ext::plugin)

set_target_properties(${TRITON_EXT_NAME} PROPERTIES
    PREFIX "lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(TARGETS ${TRITON_EXT_NAME}
    LIBRARY DESTINATION triton_mega_hoist_first_access)
