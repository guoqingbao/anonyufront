#include "../Patterns.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::ufront {

using arith::IndexCastOp;
using linalg::GenericOp;
using linalg::IndexOp;
using linalg::YieldOp;
using tensor::EmptyOp;
using tensor::ExtractOp;
using utils::IteratorType;

auto EmbeddingConverter::matchAndRewrite(EmbeddingOp op,
                                         PatternRewriter& rewriter) const
    -> LogicalResult {
  auto input = op.getInput();
  auto inputTy = input.getType();

  auto weight = op.getWeight();
  auto weightTy = weight.getType();

  SmallVector<int64_t, 3> shape;
  shape.emplace_back(inputTy.getDimSize(0));
  shape.emplace_back(inputTy.getDimSize(1));
  shape.emplace_back(weightTy.getDimSize(1));

  auto loc = op->getLoc();
  auto empty = rewriter.create<EmptyOp>(loc, shape, weightTy.getElementType());

  SmallVector<Type, 1> types{empty.getType()};
  SmallVector<Value, 2> inputs{input, weight};
  SmallVector<Value, 1> outputs{empty};
  SmallVector<AffineMap, 4> indexingMaps;
  SmallVector<IteratorType, 4> iteratorTypes(4, IteratorType::parallel);

  iteratorTypes[2] = IteratorType::reduction;

  // input: affine_map<(i, j, k, l) -> (i, j)>
  // weight: affine_map<(i, j, k, l) -> (k, l)>
  // result: affine_map<(i, j, k, l) -> (i, j, l)>
  auto identity = rewriter.getMultiDimIdentityMap(4);
  indexingMaps.emplace_back(identity.dropResults({2, 3}));
  indexingMaps.emplace_back(identity.dropResults({0, 1}));
  indexingMaps.emplace_back(identity.dropResult(2));

  auto payload = [&](OpBuilder& builder, Location loc, ValueRange args) {
    auto index =
        rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), args[0])
            .getResult();
    auto k = rewriter.create<IndexOp>(loc, 3);
    auto element = rewriter.create<ExtractOp>(loc, weight, ValueRange{index, k})
                       .getResult();
    rewriter.create<YieldOp>(loc, element);
  };

  rewriter.replaceOpWithNewOp<GenericOp>(op, types, inputs, outputs,
                                         indexingMaps, iteratorTypes, payload);
  return success();
}

}  // namespace mlir::ufront
