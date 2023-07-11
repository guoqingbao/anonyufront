add_mlir_conversion_library(UfrontToTosaConversion
    UfrontToTosa.cpp
    Patterns.cpp
    Util.cpp
    Patterns/ActivationPatterns.cpp
    Patterns/BinaryPatterns.cpp
    Patterns/MultiheadAttentionPattern.cpp
    Patterns/NormPatterns.cpp
    Patterns/Parameter.cpp
    Patterns/BinaryScalarPatterns.cpp

    DEPENDS
    UfrontConversionIncGen
)
