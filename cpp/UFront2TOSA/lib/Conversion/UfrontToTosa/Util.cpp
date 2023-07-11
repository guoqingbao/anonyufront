#include "Util.hpp"

#include <functional>
#include <numeric>

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
  return DenseElementsAttr::get(type, builder.getF32FloatAttr(value));
}

SmallVector<int64_t> getIntValueFromArrayAttr(ArrayAttr array) {
  auto valueIt = [](Attribute attr) {
    return attr.cast<IntegerAttr>().getInt();
  };

  auto values = SmallVector<int64_t>{};
  transform(array, std::back_inserter(values), valueIt);
  return values;
}

}  // namespace ufront
}  // namespace mlir
