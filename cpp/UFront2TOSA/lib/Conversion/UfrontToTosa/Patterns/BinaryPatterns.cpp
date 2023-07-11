#include "../Patterns.hpp"
#include "../Util.hpp"

namespace mlir {
namespace ufront {

LogicalResult AddConverter::matchAndRewrite(AddOp add,
                                            PatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<tosa::AddOp>(add, add.getType(), add.getLhs(),
                                           add.getRhs());
  return success();
};

LogicalResult MultiplyConverter::matchAndRewrite(
    MultiplyOp multiply, PatternRewriter& rewriter) const {
  auto lhs = multiply.getLhs();
  auto rhs = multiply.getRhs();
  auto type = multiply.getType();
  auto shift = rewriter.getI32IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(multiply, type, lhs, rhs, shift);
  return success();
}

LogicalResult BatchMatmulConverter::matchAndRewrite(
    BatchMatmulOp bmm, PatternRewriter& rewriter) const {
  auto type = bmm.getType();
  auto lhs = bmm.getLhs();
  auto rhs = bmm.getRhs();
  rewriter.replaceOpWithNewOp<tosa::MatMulOp>(bmm, type, lhs, rhs);
  return success();
}

LogicalResult SubtractConverter::matchAndRewrite(
    SubtractOp subtract, PatternRewriter& rewriter) const {
  auto lhs = subtract.getLhs();
  auto rhs = subtract.getRhs();
  auto type = subtract.getType();
  rewriter.replaceOpWithNewOp<tosa::SubOp>(subtract, type, lhs, rhs);
  return success();
}

LogicalResult MatmulConverter::matchAndRewrite(
    MatmulOp matmul, PatternRewriter& rewriter) const {
  auto type = matmul.getType();
  auto lhs = matmul.getLhs();
  auto rhs = matmul.getRhs();

  auto convert3d = [&](decltype(lhs) v) {
    auto shape = llvm::to_vector(v.getType().getShape());
    shape.insert(shape.begin(), 1);
    auto res = reshape(v, shape, rewriter);
    return res;
  };

  auto reshapedLhs = convert3d(lhs);
  auto reshapedRhs = convert3d(rhs);

  auto shape2d = type.getShape();
  auto shape3d = llvm::to_vector(shape2d);
  shape3d.insert(shape3d.begin(), 1);

  auto resultType = RankedTensorType::get(shape3d, type.getElementType());
  auto result = rewriter.create<tosa::MatMulOp>(matmul.getLoc(), resultType,
                                                reshapedLhs, reshapedRhs);

  rewriter.replaceOp(matmul, reshape(result, shape2d, rewriter));
  return success();
}

}  // namespace ufront
}  // namespace mlir
