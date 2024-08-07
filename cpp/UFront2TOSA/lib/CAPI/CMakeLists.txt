add_llvm_library(UfrontCAPI SHARED
    TosaConverter.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
find_package(ZLIB REQUIRED)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        cnpy-static
)

target_link_libraries(UfrontCAPI PRIVATE ${LIBS} ${ZLIB_LIBRARIES})
