#ifndef UFRONT_TO_TOSA_HPP
#define UFRONT_TO_TOSA_HPP

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ufront {

#define GEN_PASS_DEF_CONVERTUFRONTTOTOSA
#define GEN_PASS_DECL_CONVERTUFRONTTOTOSA
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createConvertUfrontToTosa();

}  // namespace ufront
}  // namespace mlir

#endif  // UFRONT_TO_TOSA_HPP
