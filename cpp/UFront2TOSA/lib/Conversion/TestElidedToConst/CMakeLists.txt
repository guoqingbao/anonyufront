add_mlir_conversion_library(TestElidedToConstConversion
    TestElidedToConst.cpp
    InitWeight.cpp
    Random.cpp
    
    DEPENDS
    UfrontConversionIncGen
)
