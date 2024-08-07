add_subdirectory(UfrontToTosa)
add_subdirectory(TestElidedToConst)

add_llvm_library(UfrontConversion 
    Passes.cpp

    DEPENDS
    UfrontToTosaConversion
    TestElidedToConstConversion
)
