#include "../Patterns.hpp"
#include "../Util.hpp"

namespace mlir {
namespace ufront {

LogicalResult ReluConverter::matchAndRewrite(ReluOp relu,
                                             PatternRewriter& rewriter) const {
  auto maxFp = rewriter.getF32FloatAttr(std::numeric_limits<float>::max());
  auto minFp = rewriter.getF32FloatAttr(0);
  auto maxInt = rewriter.getI64IntegerAttr(std::numeric_limits<int>::max());
  auto minInt = rewriter.getI64IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::ClampOp>(
      relu, relu.getType(), relu.getInput(), minInt, maxInt, minFp, maxFp);
  return success();
}

LogicalResult ClipConverter::matchAndRewrite(ClipOp clip,
                                             PatternRewriter& rewriter) const {
  auto minFp = clip.getMinimum();
  auto maxFp = clip.getMaximum();
  rewriter.replaceOpWithNewOp<tosa::ClampOp>(
      clip, clip.getType(), clip.getInput(), minFp.convertToDouble(), maxFp.convertToDouble(), minFp, maxFp);
  return success();
}

// %op1 = tosa.EXP(%logits)
// %op2 = tosa.REDUCE_SUM(op1) {reduce_axis=(%logits.rank - 1)}
// %op3 = tosa.RECIPROCAL(%op2)
// %output = tosa.MUL(%op1, %op3)

LogicalResult SoftmaxConverter::matchAndRewrite(
    SoftmaxOp softmax, PatternRewriter& rewriter) const {
  auto loc = softmax.getLoc();
  auto input = softmax.getInput();
  auto tp = input.getType();

  auto exp = rewriter.create<tosa::ExpOp>(loc, input.getType(), input);
  auto sum = exp.getResult();
  auto sumType = sum.getType();
  // auto sumShape = sumType.cast<ShapedType>().getShape();

  // for (auto [axis, dim] : enumerate(sumShape)) {
  //   if (dim == 1) {
  //     continue;
  //   }

  //   sum = reduceSum(sum, axis, rewriter);
  // }
  sum = reduceSum(sum, tp.getRank() - 1, rewriter);

  auto rec = rewriter.create<tosa::ReciprocalOp>(loc, sum.getType(), sum);
  auto shift = rewriter.getI32IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(softmax, sumType, exp, rec, shift);

  return success();
}

// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
struct GeluHelper {
  static constexpr auto PI = 3.141592653589793;
  static constexpr auto COEFFICIENT = 0.044715;

  // halfPart(x) = 0.5 * x
  static Value halfPart(Value x, OpBuilder& builder) {
    constexpr auto HALF = 0.5;
    constexpr auto SHIFT = 0;

    auto loc = x.getLoc();
    auto type = x.getType();
    auto attr = getDenseFloatAttr(HALF, type, builder);
    auto half = builder.create<tosa::ConstOp>(loc, type, attr);
    return builder.create<tosa::MulOp>(loc, type, x, half, SHIFT);
  }

  // sqrtPart(x) = sqrt(2 / pi)
  static Value sqrtPart(Value x, OpBuilder& builder) {
    constexpr auto HALF = 0.5;

    auto loc = x.getLoc();
    auto type = x.getType();

    auto halfPiAttr = getDenseFloatAttr(PI * HALF, type, builder);
    auto halfPi = builder.create<tosa::ConstOp>(loc, type, halfPiAttr);

    return builder.create<tosa::RsqrtOp>(loc, type, halfPi);
  }

  // powPart(x) = x + 0.044715 * x^3
  static Value powPart(Value x, OpBuilder& builder) {
    constexpr auto CUBE = 3.0;
    constexpr auto SHIFT = 0;

    auto loc = x.getLoc();
    auto type = x.getType();

    auto coAttr = getDenseFloatAttr(COEFFICIENT, type, builder);
    auto cubeAttr = getDenseFloatAttr(CUBE, type, builder);

    auto coefficient = builder.create<tosa::ConstOp>(loc, type, coAttr);
    auto cube = builder.create<tosa::ConstOp>(loc, type, cubeAttr);
    auto pow = builder.create<tosa::PowOp>(loc, type, x, cube);
    auto mul = builder.create<tosa::MulOp>(loc, type, pow, coefficient, SHIFT);

    return builder.create<tosa::AddOp>(loc, type, x, mul);
  }
  //approximation
  //0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z ** 3)))

  //non-approximation
  //0.5 * z * (1 + erf(z / sqrt(2)))
    static std::optional<Value> getConstTensor(OpBuilder &rewriter, Operation *op,
                                        float value) {
      auto loc = op->getLoc();
      auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
      auto vAttr = getDenseFloatAttr(value, const_type, rewriter);
      auto const_op = rewriter.create<tosa::ConstOp>(loc, const_type, vAttr);
      return const_op.getResult();
    }

    static Value approximateErfOp(OpBuilder &rewriter,
                                Operation *op, Value x) {
    // Using:
    // https://en.wikipedia.org/wiki/Error_function#Numerical_approximations with
    // maximum error as 5 x 10^-4 where a1 = 0.278393, a2 = 0.230389, a3 =
    // 0.000972, a4 = 0.078108.
    //
    // Erf = 1 - 1 / (1 + a1X + a2X + a3X + a4X)^4

    auto outType = x.getType().cast<TensorType>();
    auto loc = op->getLoc();
    auto absX = rewriter.create<tosa::AbsOp>(loc, outType, x);
    auto zero = getConstTensor(rewriter, op, 0).value();
    auto one = getConstTensor(rewriter, op, 1).value();

    auto a1 = getConstTensor(rewriter, op, 0.278393).value();
    auto a1X = rewriter.create<tosa::MulOp>(loc, outType, a1, absX, /*shift=*/0);
    auto sum = rewriter.create<tosa::AddOp>(loc, outType, a1X, one);

    auto a2 = getConstTensor(rewriter, op, 0.230389).value();
    auto x2 = rewriter.create<tosa::MulOp>(loc, outType, absX, absX, /*shift=*/0);
    auto a2X = rewriter.create<tosa::MulOp>(loc, outType, a2, x2, /*shift=*/0);
    sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a2X);

    auto a3 = getConstTensor(rewriter, op, 0.000972).value();
    auto x3 = rewriter.create<tosa::MulOp>(loc, outType, x2, absX, /*shift=*/0);
    auto a3X = rewriter.create<tosa::MulOp>(loc, outType, a3, x3, /*shift=*/0);
    sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a3X);

    auto a4 = getConstTensor(rewriter, op, 0.078108).value();
    auto x4 = rewriter.create<tosa::MulOp>(loc, outType, x3, absX, /*shift=*/0);
    auto a4X = rewriter.create<tosa::MulOp>(loc, outType, a4, x4, /*shift=*/0);
    sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a4X);

    auto rcprl = rewriter.create<tosa::ReciprocalOp>(loc, outType, sum);
    auto rcprl2 =
        rewriter.create<tosa::MulOp>(loc, outType, rcprl, rcprl, /*shift=*/0);
    auto rcprl4 =
        rewriter.create<tosa::MulOp>(loc, outType, rcprl2, rcprl2, /*shift=*/0);
    auto erf = rewriter.create<tosa::SubOp>(loc, outType, one, rcprl4);

    // Deal with negative x.
    auto cond = rewriter.create<tosa::GreaterEqualOp>(
        loc,
        RankedTensorType::get(outType.getShape(), rewriter.getIntegerType(1)), x,
        zero);
    auto negateErf = rewriter.create<tosa::NegateOp>(loc, outType, erf);

    return rewriter.create<tosa::SelectOp>(loc, outType, cond, erf, negateErf);
  }

  static Value buildUnitNormalCdf(OpBuilder &rewriter,
                                  Operation *op, Value x) {
    auto zero = getConstTensor(rewriter, op, 0).value();
    auto one = getConstTensor(rewriter, op, 1).value();
    auto loc = op->getLoc();

    // buildNormalCdf, mean = zero, sigma = one
    auto outType = x.getType();
    auto mean = zero;
    Value xMinusMean = rewriter.create<tosa::SubOp>(loc, outType, x, mean);
    // rsqrt of 2
    Value rsqrt2 =
        getConstTensor(rewriter, op, 0.70710678).value();
    Value erfArg = rewriter.create<tosa::MulOp>(loc, outType, xMinusMean, rsqrt2,
                                                /*shift=*/0);
    Value erf = approximateErfOp(rewriter, op, erfArg);
    Value erfPlus1 = rewriter.create<tosa::AddOp>(loc, outType, one, erf);
    Value oneHalf = getConstTensor(rewriter, op, 0.5).value();
    Value normalCdf = rewriter.create<tosa::MulOp>(loc, outType, oneHalf,
                                                  erfPlus1, /*shift=*/0);
    return normalCdf;
  }

  static Value gelu(Operation *op, Value x, OpBuilder& builder, bool approximate) {
    constexpr auto ONE = 1.0;
    constexpr auto SHIFT = 0;
    // constexpr auto TWO = 2.0;

    auto loc = x.getLoc();
    auto type = x.getType();
    auto oneAttr = getDenseFloatAttr(ONE, type, builder);
    auto one = builder.create<tosa::ConstOp>(loc, type, oneAttr);
    if (approximate) {
      auto half = halfPart(x, builder);
      auto sqrt = sqrtPart(x, builder);
      auto pow = powPart(x, builder);
      auto mul = builder.create<tosa::MulOp>(loc, type, sqrt, pow, SHIFT);
      auto tanh = builder.create<tosa::TanhOp>(loc, type, mul);
      auto add = builder.create<tosa::AddOp>(loc, type, one, tanh);
      return builder.create<tosa::MulOp>(loc, type, half, add, SHIFT);

    } 
    else { 
      return buildUnitNormalCdf(builder, op, x); //from torch-mlir
      // auto half = halfPart(x, builder);

      // auto twoAttr = getDenseFloatAttr(TWO, type, builder);
      // auto two = builder.create<tosa::ConstOp>(loc, type, twoAttr);

      // auto rsqrt = builder.create<tosa::RsqrtOp>(loc, type, two);
      // auto mul = builder.create<tosa::MulOp>(loc, type, x, rsqrt, SHIFT);
      // // auto erf = builder.create<tosa::erf>(loc, type, mul);
      // auto add = builder.create<tosa::AddOp>(loc, type, one, erf);
      // return = builder.create<tosa::MulOp>(loc, type, half, add, SHIFT);
    }
  }
};

LogicalResult GeluConverter::matchAndRewrite(GeluOp gelu,
                                             PatternRewriter& rewriter) const {

  auto approximate = gelu.getApproximate();
  rewriter.replaceOp(gelu, GeluHelper::gelu(gelu, gelu.getInput(), rewriter, approximate));
  return success();
}

LogicalResult SiluConverter::matchAndRewrite(SiluOp silu,
                                             PatternRewriter& rewriter) const {
  auto loc = silu->getLoc();
  auto input = silu.getInput();
  auto type = silu.getType();
  auto shift = rewriter.getI32IntegerAttr(0);

  auto sigmoid = rewriter.create<tosa::SigmoidOp>(loc, type, input);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(silu, type, input, sigmoid, shift);
  return success();
}

LogicalResult SigmoidConverter::matchAndRewrite(
    SigmoidOp sigmoid, PatternRewriter& rewriter) const {
  auto input = sigmoid.getInput();
  auto type = sigmoid.getType();
  rewriter.replaceOpWithNewOp<tosa::SigmoidOp>(sigmoid, type, input);
  return success();
}

Value hardsigmoidPiecewise(Value x, Value geValue, Value leValue,
                           Value elseValue, OpBuilder& builder) {
  constexpr auto UPPER = 3.0;
  constexpr auto LOWER = -3.0;

  auto type = x.getType().cast<ShapedType>();
  auto elemTy = type.getElementType();
  auto loc = x.getLoc();

  auto upper = constantScalar(UPPER, elemTy, builder);
  auto lower = constantScalar(LOWER, elemTy, builder);

  auto condType = RankedTensorType::get(type.getShape(), builder.getI1Type());
  auto ge = builder.create<tosa::GreaterEqualOp>(loc, condType, x, upper);
  auto le = builder.create<tosa::GreaterEqualOp>(loc, condType, lower, x);

  auto selectLe =
      builder.create<tosa::SelectOp>(loc, type, le, leValue, elseValue);
  return builder.create<tosa::SelectOp>(loc, type, ge, geValue, selectLe);
}

LogicalResult HardSigmoidConverter::matchAndRewrite(
    HardSigmoidOp hs, PatternRewriter& rewriter) const {
  auto x = hs.getInput();
  auto type = hs.getType();
  auto elemTy = type.getElementType();
  auto loc = hs->getLoc();

  auto zero = constantScalar(0.0, elemTy, rewriter);
  auto one = constantScalar(1.0, elemTy, rewriter);

  auto oneSixth = constantScalar(1.0 / 6.0, elemTy, rewriter);
  auto oneHalf = constantScalar(0.5, elemTy, rewriter);

  auto shift = rewriter.getI32IntegerAttr(0);
  auto mul = rewriter.create<tosa::MulOp>(loc, type, x, oneSixth, shift);
  auto add = rewriter.create<tosa::AddOp>(loc, type, mul, oneHalf);

  rewriter.replaceOp(hs, hardsigmoidPiecewise(x, one, zero, add, rewriter));
  return success();
}

LogicalResult HardSwishConverter::matchAndRewrite(
    HardSwishOp hs, PatternRewriter& rewriter) const {
  auto x = hs.getInput();
  auto type = hs.getType();
  auto elemTy = type.getElementType();
  auto loc = hs->getLoc();

  auto three = constantScalar(3.0, elemTy, rewriter);

  auto zero = constantScalar(0.0, elemTy, rewriter);
  auto oneSixth = constantScalar(1.0 / 6.0, elemTy, rewriter);

  auto shift = rewriter.getI32IntegerAttr(0);
  auto add = rewriter.create<tosa::AddOp>(loc, type, x, three);
  auto mul = rewriter.create<tosa::MulOp>(loc, type, add, oneSixth, shift);
  auto res = rewriter.create<tosa::MulOp>(loc, type, x, mul, shift);

  rewriter.replaceOp(hs, hardsigmoidPiecewise(x, x, zero, res, rewriter));
  return success();
}

}  // namespace ufront
}  // namespace mlir
