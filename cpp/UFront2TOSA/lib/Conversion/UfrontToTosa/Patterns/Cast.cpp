#include "../Patterns.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::ufront {

auto CastConverter::matchAndRewrite(CastOp op, PatternRewriter& rewriter) const
    -> LogicalResult {
  auto dtype = op.getDtype();

  Type elementTy = StringSwitch<Type>(dtype)
                       .Case("Int32", rewriter.getI32Type())
                       .Case("Int64", rewriter.getI64Type())
                       .Case("Half", rewriter.getF16Type())
                       .Case("BHalf", rewriter.getF16Type())
                       .Case("Float", rewriter.getF32Type())
                       .Case("Double", rewriter.getF64Type())
                       .Case("Bool", rewriter.getI1Type());

  auto uncastType = op.getType();
  auto castType = RankedTensorType::get(uncastType.getShape(), elementTy);

  rewriter.replaceOpWithNewOp<tosa::CastOp>(op, castType, op.getInput());
  return success();
};

}  // namespace mlir::ufront
