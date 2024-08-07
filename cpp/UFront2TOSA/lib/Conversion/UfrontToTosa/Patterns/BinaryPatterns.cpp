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

  auto lhsTy = lhs.getType();
  auto rhsTy = rhs.getType();
  assert(lhsTy.getRank() == rhsTy.getRank() &&
         "operator types must be the same rank");

  if (auto rank = lhsTy.getRank(); rank == 3) {
    rewriter.replaceOpWithNewOp<tosa::MatMulOp>(bmm, type, lhs, rhs);
  } else {
    SmallVector<int64_t> newLhsDims(3, 1);
    SmallVector<int64_t> newRhsDims(3, 1);

    for (auto i = 0; i < rank - 2; ++i) {
      newLhsDims[0] *= lhsTy.getDimSize(i);
      newRhsDims[0] *= rhsTy.getDimSize(i);
    }

    for (auto i = rank - 2, j = i; i < rank; ++i) {
      newLhsDims[i - j + 1] = lhsTy.getDimSize(i);
      newRhsDims[i - j + 1] = rhsTy.getDimSize(i);
    }

    auto newLhs = reshape(lhs, newLhsDims, rewriter);
    auto newRhs = reshape(rhs, newRhsDims, rewriter);
    auto newVal = matmul(newLhs, newRhs, rewriter);

    rewriter.replaceOp(bmm, reshape(newVal, type.getShape(), rewriter));
  }

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
