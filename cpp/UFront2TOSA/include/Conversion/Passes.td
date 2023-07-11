#ifndef UFRONT_CONVERSION_PASSES
#define UFRONT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertUfrontToTosa : Pass<"convert-ufront-to-tosa"> {
    let constructor = "mlir::ufront::createConvertUfrontToTosa()";
    let dependentDialects = ["mlir::tosa::TosaDialect"];
}

def TestConvertElidedToConst : Pass<"test-convert-elided-to-const"> {
    let constructor = "mlir::ufront::createTestConvertElidedToConst()";
    let dependentDialects = ["mlir::tosa::TosaDialect"];
}

#endif // UFRONT_CONVERSION_PASSES
