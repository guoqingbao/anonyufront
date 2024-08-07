#include "MultiheadAttention.hpp"

#include <numeric>

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::ufront::multihead_attention {

auto rankOf(Tensor tensor) -> int64_t { return tensor.getType().getRank(); }

auto shapeOf(Tensor tensor) -> ArrayRef<int64_t> {
  return tensor.getType().getShape();
}

auto dimSizeOf(Tensor tensor, int64_t dim) -> int64_t {
  return tensor.getType().getDimSize(dim);
}

auto mhaShapeCheck(Tensor query, Tensor key, Tensor value,
                   Optional<Tensor> keyPaddingMask, Optional<Tensor> attnMask,
                   int64_t numHeads) -> bool {
  assert(rankOf(query) == 2 || rankOf(query) == 3);
  assert(rankOf(query) == rankOf(key) && rankOf(query) == rankOf(value));
  assert(!keyPaddingMask || rankOf(*keyPaddingMask) == rankOf(query) - 1);
  assert(!attnMask || rankOf(*attnMask) == 2 || rankOf(*attnMask) == 3);

  if (rankOf(query) == 2 && attnMask.has_value() && rankOf(*attnMask) == 3) {
    auto actualShape = shapeOf(*attnMask);
    auto expectedShape =
        ArrayRef{numHeads, dimSizeOf(query, 0), dimSizeOf(key, 0)};
    assert(llvm::equal(actualShape, expectedShape));
  }

  return rankOf(query) == 3;
}

auto unsqueeze(Tensor tensor, int64_t dim, OpBuilder& builder) -> Tensor {
  auto type = tensor.getType();
  auto shape = type.getShape();

  auto newShape = SmallVector<int64_t>{shape};
  newShape.insert(newShape.begin() + dim, 1);

  auto newType = RankedTensorType::get(newShape, type.getElementType());
  return builder
      .create<tosa::ReshapeOp>(builder.getUnknownLoc(), newType, tensor,
                               newShape)
      .getResult();
}

auto canonicalMask(Optional<Tensor> opt, OpBuilder& builder)
    -> Optional<Tensor> {
  // TODO: implement
}

auto inProjectionPacked(Tensor query, Tensor key, Tensor value,
                        Optional<Tensor> inProjWeight,
                        Optional<Tensor> inProjBias, OpBuilder& builder)
    -> Tuple<Tensor, Tensor, Tensor> {
  // TODO: implement
}

auto inProjection(Tensor query, Tensor key, Tensor value, Tensor qProjWeight,
                  Tensor kProjWeight, Tensor vProjWeight,
                  Optional<Tensor> qProjBias, Optional<Tensor> kProjBias,
                  Optional<Tensor> vProjBias, OpBuilder& builder)
    -> Tuple<Tensor, Tensor, Tensor> {
  // TODO: implement
}

auto pad(Tensor tensor, ArrayRef<int64_t> dims, OpBuilder& builder) -> Tensor {
  // TODO: implement
}

auto forward(OpBuilder& builder, Tensor query, Tensor key, Tensor value,
             int64_t embedDimToCheck, int64_t numHeads,
             Optional<Tensor> inProjWeight, Optional<Tensor> inProjBias,
             Optional<Tensor> biasK, Optional<Tensor> biasV, bool addZeroAttn,
             double dropout, Tensor outProjWeight, Optional<Tensor> outProjBias,
             bool training = true, Optional<Tensor> keyPaddingMask = None,
             bool needWeights = true, Optional<Tensor> attnMask = None,
             bool useSeparateProjWeight = false,
             Optional<Tensor> qProjWeight = None,
             Optional<Tensor> kProjWeight = None,
             Optional<Tensor> vProjWeight = None,
             Optional<Tensor> staticK = None, Optional<Tensor> staticV = None,
             bool averageAttnWeights = true, bool isCasual = false)
    -> Tuple<Tensor, Optional<Tensor>> {
  auto loc = builder.getUnknownLoc();
  auto isBatched =
      mhaShapeCheck(query, key, value, keyPaddingMask, attnMask, numHeads);

  if (!isBatched) {
    query = unsqueeze(query, 1, builder);
    key = unsqueeze(key, 1, builder);
    value = unsqueeze(value, 1, builder);
    if (keyPaddingMask.has_value()) {
      keyPaddingMask = unsqueeze(*keyPaddingMask, 0, builder);
    }
  }

  auto tgtLen = dimSizeOf(query, 0);
  auto bsz = dimSizeOf(query, 1);
  auto embedDim = dimSizeOf(query, 2);
  auto srcLen = dimSizeOf(key, 0);

  keyPaddingMask = canonicalMask(keyPaddingMask, builder);

  assert(!isCasual || attnMask.has_value());
  if (isCasual && !keyPaddingMask.has_value() && !needWeights) {
    attnMask = None;
  } else {
    attnMask = canonicalMask(attnMask, builder);

    if (keyPaddingMask.has_value()) {
      isCasual = false;
    }
  }

  assert(embedDim == embedDimToCheck);
  auto headDim = embedDim / numHeads;
  assert(headDim * numHeads == embedDim);

  if (useSeparateProjWeight) {
    assert(
        llvm::equal(shapeOf(key).take_front(2), shapeOf(value).take_front(2)));
  } else {
    assert(llvm::equal(shapeOf(key), shapeOf(value)));
  }

  Optional<Tensor> q, k, v;
  if (!useSeparateProjWeight) {
    assert(inProjWeight.has_value());
    auto inproj = inProjectionPacked(query, key, value, inProjWeight,
                                     inProjBias, builder);
    q = std::get<0>(inproj);
    k = std::get<1>(inproj);
    v = std::get<2>(inproj);
  } else {
    assert(qProjWeight.has_value() && kProjWeight.has_value() &&
           vProjWeight.has_value());
    auto inproj =
        inProjection(query, key, value, *qProjWeight, *kProjWeight,
                     *vProjWeight, inProjBias, inProjBias, inProjBias, builder);
    q = std::get<0>(inproj);
    k = std::get<1>(inproj);
    v = std::get<2>(inproj);
  }

  if (auto mask = *attnMask; mask) {
    assert(rankOf(mask) == 2 || rankOf(mask) == 3);

    if (rankOf(mask) == 2) {
      auto correctSize = ArrayRef{tgtLen, srcLen};
      assert(llvm::equal(correctSize, shapeOf(mask)));
      attnMask = unsqueeze(mask, 0, builder);
    } else {
      auto correctSize = ArrayRef{bsz * numHeads, tgtLen, srcLen};
      assert(llvm::equal(correctSize, shapeOf(mask)));
    }
  }

  if (biasK.has_value() && biasV.has_value()) {
    assert(!staticK.has_value() && !staticV.has_value());
    auto repeatAndConcat = [&](Tensor tensor, Tensor bias) -> Tensor {
      auto biasShape = SmallVector<int64_t>{shapeOf(bias)};
      biasShape[1] *= bsz;
      auto biasType =
          RankedTensorType::get(biasShape, getElementTypeOrSelf(bias));

      SmallVector<Tensor> biases(bsz, bias);
      auto biasesTensor =
          builder.create<ConcatOp>(loc, biasType, biases, 1).getResult();

      auto tensorShape = SmallVector<int64_t>{shapeOf(tensor)};
      tensorShape[0] += biasShape[0];
      auto tensorType =
          RankedTensorType::get(tensorShape, getElementTypeOrSelf(tensor));

      return builder
          .create<ConcatOp>(loc, tensorType,
                            ArrayRef<Tensor>{tensor, biasesTensor}, 0)
          .getResult();
    };

    k = repeatAndConcat(*k, *biasK);
    v = repeatAndConcat(*v, *biasV);

    if (attnMask.has_value()) {
      attnMask = pad(*attnMask, {0, 1}, builder);
    }

    if (keyPaddingMask.has_value()) {
      keyPaddingMask = pad(*attnMask, {0, 1}, builder);
    }
  }

  q = reshape(*q, {bsz * numHeads, tgtLen, headDim}, builder);
  q = transpose(*q, {1, 0, 2}, builder);

  if (!staticK.has_value()) {
    k = reshape(*k, {dimSizeOf(*k, 0), bsz * numHeads, headDim}, builder);
    k = transpose(*k, {1, 0, 2}, builder);
  } else {
    assert(dimSizeOf(*k, 0) == bsz * numHeads);
    assert(dimSizeOf(*k, 2) == headDim);
    k = *staticK;
  }

  if (!staticV.has_value()) {
    v = reshape(*v, {dimSizeOf(*v, 0), bsz * numHeads, headDim}, builder);
    v = transpose(*v, {1, 0, 2}, builder);
  } else {
    assert(dimSizeOf(*v, 0) == bsz * numHeads);
    assert(dimSizeOf(*v, 2) == headDim);
    v = *staticV;
  }

  if (addZeroAttn) {
    auto zerosType = RankedTensorType::get({bsz * numHeads, 1, headDim},
                                           getElementTypeOrSelf(*q));
    auto zeros = constant(0.0, zerosType, builder);
    k = builder.create<ConcatOp>(loc, ValueRange{*k, zeros}, 1);
    v = builder.create<ConcatOp>(loc, ValueRange{*v, zeros}, 1);

    if (attnMask.has_value()) {
      attnMask = pad(*attnMask, {0, 1}, builder);
    }

    if (keyPaddingMask.has_value()) {
      keyPaddingMask = pad(*keyPaddingMask, {0, 1}, builder);
    }
  }

  srcLen = dimSizeOf(*k, 0);

  if (keyPaddingMask.has_value()) {
    SmallVector<int64_t> correctSize{bsz, srcLen};
    assert(llvm::equal(correctSize, shapeOf(*keyPaddingMask)));

    keyPaddingMask = reshape(*keyPaddingMask, {bsz, 1, 1, srcLen}, builder);
    auto shape = shapeOf(*keyPaddingMask);
    auto numElements =
        std::accumulate(shape.begin(), shape.end(), 1L, std::multiplies<>());
    auto dim = static_cast<int64_t>(std::cbrt(numElements));
    auto type = RankedTensorType::get({dim, numHeads, dim, dim},
                                      getElementTypeOrSelf(*keyPaddingMask));
    keyPaddingMask = builder.create<ExpandOp>(
        loc, type, *keyPaddingMask,
        builder.getI64ArrayAttr({dim, numHeads, dim, dim}));
    keyPaddingMask =
        reshape(*keyPaddingMask, {bsz * numHeads, 1, srcLen}, builder);

    if (attnMask.has_value()) {
      attnMask = builder.create<tosa::AddOp>(loc, attnMask->getType(),
                                             *attnMask, *keyPaddingMask);
    } else {
      attnMask = keyPaddingMask;
    }
  }
}

auto forward(ForwardConfig config, OpBuilder& builder)
    -> Tuple<Tensor, Optional<Tensor>> {
  auto [query, key, value, embedDimToCheck, numHeads, inProjWeight, inProjBias,
        biasK, biasV, addZeroAttn, dropout, outProjWeight, outProjBias,
        training, keyPaddingMask, needWeights, attnMask, useSeparateProjWeight,
        qProjWeight, kProjWeight, vProjWeight, staticK, staticV,
        averageAttnWeights, isCasual] = config;
  return forward(builder, query, key, value, embedDimToCheck, numHeads,
                 inProjWeight, inProjBias, biasK, biasV, addZeroAttn, dropout,
                 outProjWeight, outProjBias, training, keyPaddingMask,
                 needWeights, attnMask, useSeparateProjWeight, qProjWeight,
                 kProjWeight, vProjWeight, staticK, staticV, averageAttnWeights,
                 isCasual);
}

}  // namespace mlir::ufront::multihead_attention
