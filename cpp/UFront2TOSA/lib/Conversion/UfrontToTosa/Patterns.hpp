#include "Conversion/UfrontToTosa/UfrontToTosa.hpp"
#include "Dialect/Ufront/IR/Ufront.hpp"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace ufront {

void populateConvertUfrontToTosaPatterns(RewritePatternSet& patterns);

class AddConverter : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp add,
                                PatternRewriter& rewriter) const override;
};

class ReluConverter : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReluOp relu,
                                PatternRewriter& rewriter) const override;
};

class FlatConverter : public OpRewritePattern<FlatOp> {
  using OpRewritePattern<FlatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FlatOp flat,
                                PatternRewriter& rewriter) const override;
};

class Conv2DConverter : public OpRewritePattern<Conv2DOp> {
  using OpRewritePattern<Conv2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Conv2DOp conv,
                                PatternRewriter& rewriter) const override;
};

class BatchNormConverter : public OpRewritePattern<BatchNormOp> {
  using OpRewritePattern<BatchNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BatchNormOp bn,
                                PatternRewriter& rewriter) const override;
};

class LinearConverter : public OpRewritePattern<LinearOp> {
  using OpRewritePattern<LinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LinearOp linear,
                                PatternRewriter& rewriter) const override;
};

class SoftmaxConverter : public OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern<SoftmaxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SoftmaxOp softmax,
                                PatternRewriter& rewriter) const override;
};

class Pool2DConverter : public OpRewritePattern<Pool2DOp> {
  using OpRewritePattern<Pool2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Pool2DOp pool,
                                PatternRewriter& rewriter) const override;
};

class ReshapeConverter : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp reshape,
                                PatternRewriter& rewriter) const override;
};

class ConcatConverter : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern<ConcatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatOp concat,
                                PatternRewriter& rewriter) const override;
};

class DropoutConverter : public OpRewritePattern<DropoutOp> {
  using OpRewritePattern<DropoutOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DropoutOp dropout,
                                PatternRewriter& rewriter) const override;
};

class TransposeConverter : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransposeOp transpose,
                                PatternRewriter& rewriter) const override;
};

class ExpandConverter : public OpRewritePattern<ExpandOp> {
  using OpRewritePattern<ExpandOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpandOp expand,
                                PatternRewriter& rewriter) const override;
};

class GeluConverter : public OpRewritePattern<GeluOp> {
  using OpRewritePattern<GeluOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GeluOp gelu,
                                PatternRewriter& rewriter) const override;
};

class SliceConverter : public OpRewritePattern<SliceOp> {
  using OpRewritePattern<SliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SliceOp slice,
                                PatternRewriter& rewriter) const override;
};

class LayerNormConverter : public OpRewritePattern<LayerNormOp> {
  using OpRewritePattern<LayerNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LayerNormOp ln,
                                PatternRewriter& rewriter) const override;
};

class MultiplyConverter : public OpRewritePattern<MultiplyOp> {
  using OpRewritePattern<MultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MultiplyOp multiply,
                                PatternRewriter& rewriter) const override;
};

class SigmoidConverter : public OpRewritePattern<SigmoidOp> {
  using OpRewritePattern<SigmoidOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SigmoidOp sigmoid,
                                PatternRewriter& rewriter) const override;
};

class SiluConverter : public OpRewritePattern<SiluOp> {
  using OpRewritePattern<SiluOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SiluOp silu,
                                PatternRewriter& rewriter) const override;
};

class HardSigmoidConverter : public OpRewritePattern<HardSigmoidOp> {
  using OpRewritePattern<HardSigmoidOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(HardSigmoidOp hs,
                                PatternRewriter& rewriter) const override;
};

class HardSwishConverter : public OpRewritePattern<HardSwishOp> {
  using OpRewritePattern<HardSwishOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(HardSwishOp hs,
                                PatternRewriter& rewriter) const override;
};

class MultiheadAttentionConverter
    : public OpRewritePattern<MultiheadAttentionOp> {
  using OpRewritePattern<MultiheadAttentionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MultiheadAttentionOp mha,
                                PatternRewriter& rewriter) const override;
};

class BatchMatmulConverter : public OpRewritePattern<BatchMatmulOp> {
  using OpRewritePattern<BatchMatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BatchMatmulOp bmm,
                                PatternRewriter& rewriter) const override;
};

class MaskedFillConverter : public OpRewritePattern<MaskedFillOp> {
  using OpRewritePattern<MaskedFillOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MaskedFillOp mf,
                                PatternRewriter& rewriter) const override;
};

class ChunkConverter : public OpRewritePattern<ChunkOp> {
  using OpRewritePattern<ChunkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ChunkOp chunk,
                                PatternRewriter& rewriter) const override;
};

class MeanConverter : public OpRewritePattern<MeanOp> {
  using OpRewritePattern<MeanOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MeanOp mean,
                                PatternRewriter& rewriter) const override;
};

class ParameterConverter : public OpRewritePattern<ParameterOp> {
  using OpRewritePattern<ParameterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ParameterOp parameter,
                                PatternRewriter& rewriter) const override;
};

class SmultiplyConverter : public OpRewritePattern<SmultiplyOp> {
  using OpRewritePattern<SmultiplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SmultiplyOp smultiply,
                                PatternRewriter& rewriter) const override;
};

class SaddConverter : public OpRewritePattern<SaddOp> {
  using OpRewritePattern<SaddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SaddOp sadd,
                                PatternRewriter& rewriter) const override;
};

class SplitConverter : public OpRewritePattern<SplitOp> {
  using OpRewritePattern<SplitOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SplitOp split,
                                PatternRewriter& rewriter) const override;
};

class SubtractConverter : public OpRewritePattern<SubtractOp> {
  using OpRewritePattern<SubtractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SubtractOp subtract,
                                PatternRewriter& rewriter) const override;
};

class MatmulConverter : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MatmulOp matmul,
                                PatternRewriter& rewriter) const override;
};

class StrueDivConverter : public OpRewritePattern<StrueDivOp> {
  using OpRewritePattern<StrueDivOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(StrueDivOp matmul,
                                PatternRewriter& rewriter) const override;
};

class ClipConverter : public OpRewritePattern<ClipOp> {
  using OpRewritePattern<ClipOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ClipOp relu,
                                PatternRewriter& rewriter) const override;
};

}  // namespace ufront
}  // namespace mlir
