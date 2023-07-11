#include "../Patterns.hpp"

namespace mlir {
namespace ufront {

Value getScalarTensor(double value, OpBuilder& builder) {
  auto type = RankedTensorType::get({}, builder.getF32Type());
  auto elem = builder.getF32FloatAttr(value);
  auto attr = DenseElementsAttr::get(type, elem);
  return builder.create<tosa::ConstOp>(builder.getUnknownLoc(), type, attr);
}

LogicalResult SaddConverter::matchAndRewrite(SaddOp sadd,
                                             PatternRewriter& rewriter) const {
  auto scalar = getScalarTensor(sadd.getScalar().convertToDouble(), rewriter);
  rewriter.replaceOpWithNewOp<tosa::AddOp>(sadd, sadd.getType(),
                                           sadd.getInput(), scalar);
  return success();
};

LogicalResult SmultiplyConverter::matchAndRewrite(
    SmultiplyOp smultiply, PatternRewriter& rewriter) const {
  auto scalar =
      getScalarTensor(smultiply.getScalar().convertToDouble(), rewriter);
  auto shift = rewriter.getI32IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(smultiply, smultiply.getType(),
                                           smultiply.getInput(), scalar, shift);
  return success();
}

LogicalResult StrueDivConverter::matchAndRewrite(
    StrueDivOp struediv, PatternRewriter& rewriter) const {
  auto scalarVal = struediv.getScalar().convertToDouble();
  auto reciprocal = 1.0 / scalarVal;
  auto scalar = getScalarTensor(reciprocal, rewriter);
  auto shift = rewriter.getI32IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(struediv, struediv.getType(),
                                           struediv.getInput(), scalar, shift);
  return success();
}

}  // namespace ufront
}  // namespace mlir
