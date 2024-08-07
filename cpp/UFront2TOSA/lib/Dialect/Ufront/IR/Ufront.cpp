#include "Dialect/Ufront/IR/Ufront.hpp"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#define GET_OP_CLASSES
#include "Dialect/Ufront/IR/Ufront.cpp.inc"
#include "Dialect/Ufront/IR/UfrontDialect.cpp.inc"

namespace mlir {
namespace ufront {

void UfrontDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Ufront/IR/Ufront.cpp.inc"
      >();
}

void ElidedOp::build(OpBuilder& builder, OperationState& state,
                     ArrayRef<int64_t> shape, Type elementType) {
  auto type = RankedTensorType::get(shape, elementType);
  return build(builder, state, type);
}

LogicalResult ExpandOp::verify() {
  auto sizes = getSizes();
  auto inTy = getInput().getType();

  if (static_cast<size_t>(inTy.getRank()) != sizes.size()) {
    emitError() << "Length of sizes must be equal to rank of input\n";
    return failure();
  }

  for (auto [attr, dim] : llvm::zip(sizes, inTy.getShape())) {
    auto attrVal = attr.cast<IntegerAttr>().getInt();
    if (attrVal == -1 || attrVal == dim) {
      continue;
    }

    if (dim != 1) {
      emitError() << "Dim to be expanded must be 1\n";
      return failure();
    }
  }

  return success();
}

void ConcatOp::build(OpBuilder& builder, OperationState& state,
                     ValueRange inputs, uint64_t axis) {
  if (inputs.size() == 1) {
    build(builder, state, inputs[0].getType(), inputs[0], axis);
  }

  auto tensorTy = dyn_cast<TensorType>(inputs[0].getType());
  assert(tensorTy && "inputs must be tensor type");
  assert(static_cast<uint64_t>(tensorTy.getRank()) > axis &&
         "axis must be less than rank");

  SmallVector<int64_t> shape{tensorTy.getShape()};

  for (auto input : inputs.drop_front()) {
    auto inputTy = dyn_cast<TensorType>(input.getType());
    assert(inputTy && "inputs must be tensor type");
    assert(inputTy.getRank() == tensorTy.getRank() &&
           "all inputs must have same rank");
    assert(inputTy.getElementType() == tensorTy.getElementType() &&
           "all inputs must have same element type");

    for (auto i : llvm::seq(0L, tensorTy.getRank())) {
      if (static_cast<uint64_t>(i) == axis) {
        continue;
      }

      assert(inputTy.getDimSize(i) == tensorTy.getDimSize(i) &&
             "all inputs must have same shape except axis");
    }

    shape[axis] += inputTy.getDimSize(axis);
  }

  auto type = RankedTensorType::get(shape, tensorTy.getElementType());
  build(builder, state, type, inputs, axis);
}

}  // namespace ufront
}  // namespace mlir
