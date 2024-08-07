#include "../Patterns.hpp"
#include "../Util.hpp"
#include "Conversion/Passes.hpp"

namespace mlir {
namespace ufront {

struct MultiheadAttentionHelper {
  template <typename T>
  static T unary(Value v, OpBuilder& builder) {
    return builder.create<T>(v.getLoc(), v.getType(), v);
  }

  static Value linear(Value x, OpBuilder& builder) {
    return unary<LinearOp>(x, builder);
  }

  static Value activation(Value x, OpBuilder& builder) {
    return unary<ReluOp>(x, builder);
  }

  static Value reshapeToBatches(Value x, int64_t numHeads, OpBuilder& builder) {
    auto type = x.getType().cast<ShapedType>();
    auto batchSize = type.getDimSize(0);
    auto seqLen = type.getDimSize(1);
    auto inFeature = type.getDimSize(2);
    auto subDim = inFeature / numHeads;

    x = reshape(x, {batchSize, seqLen, numHeads, subDim}, builder);
    x = transpose(x, {0, 2, 1, 3}, builder);
    return reshape(x, {batchSize * numHeads, seqLen, subDim}, builder);
  }

  static Value reshapeFromBatches(Value x, int64_t numHeads,
                                  OpBuilder& builder) {
    auto type = x.getType().cast<ShapedType>();
    auto batchSize = type.getDimSize(0);
    auto seqLen = type.getDimSize(1);
    auto inFeature = type.getDimSize(2);

    batchSize /= numHeads;
    auto outDim = inFeature * numHeads;

    x = reshape(x, {batchSize, numHeads, seqLen, inFeature}, builder);
    x = transpose(x, {0, 2, 1, 3}, builder);
    return reshape(x, {batchSize, seqLen, outDim}, builder);
  }

  // q, k, v are 3d tensors
  static Value dot(Value q, Value k, Value v, OpBuilder& builder,
                   Value mask = nullptr) {
    auto trans = transpose(k, {0, 2, 1}, builder);
    auto mm = matmul(q, trans, builder);

    auto type = q.getType().cast<ShapedType>();
    auto elemTy = type.getElementType();
    auto dk = type.getDimSize(2);
    auto scalar = constantScalar(static_cast<double>(dk), elemTy, builder);
    auto rsqrt = unary<tosa::RsqrtOp>(scalar, builder);

    auto shift = builder.getI32IntegerAttr(0);

    auto mmLoc = mm.getLoc();
    auto mmType = mm.getType();
    auto scores = builder.create<tosa::MulOp>(mmLoc, mmType, mm, rsqrt, shift)
                      .getResult();

    if (mask) {
      auto loc = scores.getLoc();
      auto type = scores.getType();
      auto attr = builder.getF64FloatAttr(-1e9);
      scores = builder.create<MaskedFillOp>(loc, type, scores, mask, attr);
    }

    auto attention = unary<SoftmaxOp>(scores, builder);
    return matmul(attention, v, builder);
  }
};

LogicalResult MultiheadAttentionConverter::matchAndRewrite(
    MultiheadAttentionOp mha, PatternRewriter& rewriter) const {
  using Helper = MultiheadAttentionHelper;

  auto qkv = SmallVector<Value>{mha.getQuery(), mha.getKey(), mha.getValue()};
  auto weights = SmallVector<Value>{mha.getWeightQ(), mha.getWeightK(), mha.getWeightV()};
  auto biases = SmallVector<Value>{mha.getBiasQ(), mha.getBiasK(), mha.getBiasV()};

  auto mask = mha.getMask();
  auto numHeads = static_cast<int64_t>(mha.getNumHeads());
  auto weight_transposed = mha.getWeightTransposed();

  BoolAttr wt = nullptr;
  if (weight_transposed && weight_transposed.has_value())
    wt = rewriter.getBoolAttr(weight_transposed.value());

  for (auto [i, val] : enumerate(qkv)) {
    if (weights[i]) {
      if (biases[i]) {
        val = rewriter.create<LinearOp>(val.getLoc(), val.getType(), val, weights[i], biases[i], wt);
      } else {
        val = rewriter.create<LinearOp>(val.getLoc(), val.getType(), val, weights[i], nullptr, wt);
      }
    } else {
      val = Helper::linear(val, rewriter);
    }
    // val = Helper::activation(val, rewriter);
    val = Helper::reshapeToBatches(val, numHeads, rewriter);
    qkv[i] = val;
  }

  if (mask) {
    auto loc = mha->getLoc();
    auto type = mask.getType();
    auto elemTy = type.getElementType();
    auto shape = SmallVector<int64_t>{type.getShape()};
    shape[0] *= numHeads;
    auto attr = rewriter.getI64ArrayAttr(shape);
    auto newType = RankedTensorType::get(shape, elemTy);
    mask = rewriter.create<ExpandOp>(loc, newType, mask, attr);
  }

  auto res = Helper::dot(qkv[0], qkv[1], qkv[2], rewriter, mask);
  res = Helper::reshapeFromBatches(res, numHeads, rewriter);
  auto weight_o = mha.getWeightO();
  if (weight_o) {
    auto bias_o = mha.getBiasO();
    if (bias_o) {
      res = rewriter.create<LinearOp>(res.getLoc(), res.getType(), res, weight_o, bias_o, wt);
    } else {
      res = rewriter.create<LinearOp>(res.getLoc(), res.getType(), res, weight_o, nullptr, wt);
    }
  } else {
    res = Helper::linear(res, rewriter);
  }
  // res = Helper::activation(res, rewriter);
  rewriter.replaceOp(mha, res);
  return success();
}

}  // namespace ufront
}  // namespace mlir
