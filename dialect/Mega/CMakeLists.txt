triton_extension(MegaDialect)

# Add dialect TableGen; creates `MLIRMegaIncGen` target.
add_mlir_dialect(Mega mega)
add_mlir_doc(Mega Mega Dialects/ -gen-dialect-doc -dialect=mega)

# Add TableGen files as includes.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add compile definitions for generated code.
add_compile_definitions(
    TRITON_EXT_NAME=${PROJECT_NAME}
    TRITON_EXT_CLASS=${TRITON_EXT_CLASS}
)

# Build the dialect extension library (SHARED).
add_mlir_library(${PROJECT_NAME}
    Mega.cpp

    SHARED

    DEPENDS
    MLIRMegaIncGen

    LINK_LIBS PUBLIC
    TritonExtensionSupport
    MLIRDialect
    LLVMSupport
    MLIRSupport
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    "${TRITON_EXT_LIBRARY_DIR}")
