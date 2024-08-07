#ifndef UFRONT_DIALECT_HPP
#define UFRONT_DIALECT_HPP

#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// clang-format off
#include "mlir/IR/Dialect.h"
#include "Dialect/Ufront/IR/UfrontDialect.h.inc"
// clang-format on

namespace mlir {
namespace ufront {
void registerConversionPasses();
}  // namespace ufront
}  // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/Ufront/IR/Ufront.h.inc"

#endif  // UFRONT_DIALECT_HPP
