#ifndef UFRONT_CONVERSION_HPP
#define UFRONT_CONVERSION_HPP

#include "TestElidedToConst/TestElidedToConst.hpp"
#include "UfrontToTosa/UfrontToTosa.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.hpp.inc"

#endif  // UFRONT_CONVERSION_HPP
