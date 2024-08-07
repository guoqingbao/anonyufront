#include "Util.hpp"

#include <functional>
#include <numeric>

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace ufront {

using namespace tosa;

Value transpose(Value tensor, ArrayRef<int64_t> perms, OpBuilder& builder) {
  auto loc = tensor.getLoc();

  auto attrs = SmallVector<Attribute>{};
  auto toAttr = [&](int64_t i) { return builder.getI64IntegerAttr(i); };
  transform(perms, std::back_inserter(attrs), toAttr);

  // auto permsShape = ArrayRef<int64_t>{static_cast<int64_t>(perms.size())};
  auto permsType = RankedTensorType::get({static_cast<int64_t>(perms.size())}, builder.getI64Type());
  auto permsAttr = DenseElementsAttr::get(permsType, attrs);
  auto permsValue = builder.create<ConstOp>(loc, permsType, permsAttr);

  auto oldType = tensor.getType().cast<ShapedType>();
  auto indexDimMap = DenseMap<size_t, int64_t>{};

  for (auto [i, dim] : llvm::enumerate(oldType.getShape())) {
    indexDimMap[i] = dim;
  }

  auto newShape = SmallVector<int64_t>{};
  for (auto perm : perms) {
    newShape.emplace_back(indexDimMap[perm]);
  }

  auto newType = RankedTensorType::get(newShape, oldType.getElementType());
  return builder.create<TransposeOp>(loc, newType, tensor, permsValue);
}

Value reduceSum(Value tensor, uint64_t axis, OpBuilder& builder) {
  auto type = tensor.getType().cast<ShapedType>();
  auto shape = type.getShape();

  if (axis < 0 || axis >= shape.size()) {
    return tensor;
  }

  auto newShape = SmallVector<int64_t>{shape};
  newShape[axis] = 1;
  auto newType = RankedTensorType::get(newShape, type.getElementType());

  return builder.create<ReduceSumOp>(tensor.getLoc(), newType, tensor, axis);
}

Value matmul(Value lhs, Value rhs, OpBuilder& builder) {
  auto lhsType = lhs.getType().cast<ShapedType>();
  auto rhsType = rhs.getType().cast<ShapedType>();

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();

  auto validate = [](decltype(lhsShape) lhsShape, decltype(rhsShape) rhsShape) {
    return lhsShape.size() == 3 && rhsShape.size() == 3 &&
           lhsShape[2] == rhsShape[1];
  };
  assert(validate(lhsShape, rhsShape) && "Invalid inputs for matmul\n");

  auto newShape = SmallVector<int64_t>{lhsShape};
  newShape[2] = rhsShape[2];
  auto newType = RankedTensorType::get(newShape, lhsType.getElementType());

  return builder.create<MatMulOp>(lhs.getLoc(), newType, lhs, rhs);
}

Value reshape(Value tensor, ArrayRef<int64_t> dims, OpBuilder& builder) {
  auto oldType = tensor.getType().cast<ShapedType>();
  auto oldShape = oldType.getShape();
  auto oldProd = std::accumulate(oldShape.begin(), oldShape.end(), 1L,
                                 std::multiplies<int64_t>());

  auto newProd =
      std::accumulate(dims.begin(), dims.end(), 1L, std::multiplies<int64_t>());

  assert(oldProd == newProd && "Invalid new shape\n");

  auto newType = RankedTensorType::get(dims, oldType.getElementType());
  return builder.create<ReshapeOp>(tensor.getLoc(), newType, tensor, dims);
}

Value constant(double value, Type type, OpBuilder& builder) {
  auto attr = getDenseFloatAttr(value, type, builder);
  return builder.create<ConstOp>(builder.getUnknownLoc(), type, attr);
}

Value constantScalar(double value, Type elemTy, OpBuilder& builder) {
  auto type = RankedTensorType::get({}, elemTy);
  return constant(value, type, builder);
}

DenseElementsAttr getDenseFloatAttr(double value, Type type,
                                    OpBuilder& builder) {
  auto tensorTy = cast<TensorType>(type);
  auto elemTy = tensorTy.getElementType();
  return DenseElementsAttr::get(type, builder.getFloatAttr(elemTy, value));
}

DenseElementsAttr getDenseIntegerAttr(int64_t value, Type type,
                                      OpBuilder& builder) {
  auto tensorTy = cast<TensorType>(type);
  auto elemTy = tensorTy.getElementType();
  return DenseElementsAttr::get(type, builder.getIntegerAttr(elemTy, value));
}

DenseElementsAttr getDenseElementsAttr(double value, Type type,
                                       OpBuilder& builder) {
  auto elemTy = cast<ShapedType>(type).getElementType();

  if (isa<IntegerType>(elemTy)) {
    return getDenseIntegerAttr(static_cast<int64_t>(value), type, builder);
  }

  return getDenseFloatAttr(value, type, builder);
}

SmallVector<int64_t> getIntValueFromArrayAttr(ArrayAttr array) {
  auto valueIt = [](Attribute attr) {
    return attr.cast<IntegerAttr>().getInt();
  };

  auto values = SmallVector<int64_t>{};
  transform(array, std::back_inserter(values), valueIt);
  return values;
}

std::optional<Value> getConstTensor(OpBuilder &rewriter, Operation *op,
                                    float value) {
  auto loc = op->getLoc();
  auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
  auto vAttr = getDenseFloatAttr(value, const_type, rewriter);
  auto const_op = rewriter.create<tosa::ConstOp>(loc, const_type, vAttr);
  return const_op.getResult();
}

Value approximateErfOp(OpBuilder &rewriter,
                                Operation *op, Value x) {
    // Using:
    // https://en.wikipedia.org/wiki/Error_function#Numerical_approximations with
    // maximum error as 5 x 10^-4 where a1 = 0.278393, a2 = 0.230389, a3 =
    // 0.000972, a4 = 0.078108.
    //
    // Erf = 1 - 1 / (1 + a1X + a2X + a3X + a4X)^4

  return rewriter.create<math::ErfOp>(op->getLoc(), x);

  //  auto outType = x.getType().cast<TensorType>();
  // auto loc = op->getLoc();
  // auto absX = rewriter.create<tosa::AbsOp>(loc, outType, x);
  // auto zero = getConstTensor(rewriter, op, 0).value();
  // auto one = getConstTensor(rewriter, op, 1).value();

  // auto a1 = getConstTensor(rewriter, op, 0.278393).value();
  // auto a1X = rewriter.create<tosa::MulOp>(loc, outType, a1, absX,
  // /*shift=*/0); auto sum = rewriter.create<tosa::AddOp>(loc, outType, a1X,
  // one);

  // auto a2 = getConstTensor(rewriter, op, 0.230389).value();
  // auto x2 = rewriter.create<tosa::MulOp>(loc, outType, absX, absX,
  // /*shift=*/0); auto a2X = rewriter.create<tosa::MulOp>(loc, outType, a2, x2,
  // /*shift=*/0); sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a2X);

  // auto a3 = getConstTensor(rewriter, op, 0.000972).value();
  // auto x3 = rewriter.create<tosa::MulOp>(loc, outType, x2, absX,
  // /*shift=*/0); auto a3X = rewriter.create<tosa::MulOp>(loc, outType, a3, x3,
  // /*shift=*/0); sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a3X);

  // auto a4 = getConstTensor(rewriter, op, 0.078108).value();
  // auto x4 = rewriter.create<tosa::MulOp>(loc, outType, x3, absX,
  // /*shift=*/0); auto a4X = rewriter.create<tosa::MulOp>(loc, outType, a4, x4,
  // /*shift=*/0); sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a4X);

  // auto rcprl = rewriter.create<tosa::ReciprocalOp>(loc, outType, sum);
  // auto rcprl2 =
  //     rewriter.create<tosa::MulOp>(loc, outType, rcprl, rcprl, /*shift=*/0);
  // auto rcprl4 =
  //     rewriter.create<tosa::MulOp>(loc, outType, rcprl2, rcprl2,
  //     /*shift=*/0);
  // auto erf = rewriter.create<tosa::SubOp>(loc, outType, one, rcprl4);

  // // Deal with negative x.
  // auto cond = rewriter.create<tosa::GreaterEqualOp>(
  //     loc,
  //     RankedTensorType::get(outType.getShape(), rewriter.getIntegerType(1)),
  //     x, zero);
  // auto negateErf = rewriter.create<tosa::NegateOp>(loc, outType, erf);

  // return rewriter.create<tosa::SelectOp>(loc, outType, cond, erf,
  // negateErf);
  }

}  // namespace ufront
}  // namespace mlir
