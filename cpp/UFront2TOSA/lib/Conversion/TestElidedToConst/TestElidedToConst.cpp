#include "Conversion/TestElidedToConst/TestElidedToConst.hpp"

#include <math.h>
#ifdef USEMKL
#include <mkl/mkl_vsl.h>
#include <mkl/mkl_vsl_defines.h>
#endif

#ifdef USEOMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

#include "Dialect/Ufront/IR/Ufront.hpp"
#include "InitWeight.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ufront {

#ifdef USEMKL
std::vector<float> getNormalArray(long size, float mean, float std) {
  static constexpr auto RNG = VSL_BRNG_MT19937;
  static constexpr auto METHOD = VSL_RNG_METHOD_GAUSSIAN_ICDF;
  static std::random_device seed;

  auto arr = std::make_unique<float[]>(size);

  VSLStreamStatePtr stream;
  vslNewStream(&stream, RNG, seed());
  vsRngGaussian(METHOD, stream, size, arr.get(), mean, std);
  return std::vector<float>{arr.get(), arr.get() + size};
}

std::vector<float> getUniformArray(long size, float min, float max) {
  static constexpr auto RNG = VSL_BRNG_MT19937;
  static constexpr auto METHOD = VSL_RNG_METHOD_UNIFORM_STD;
  static std::random_device seed;

  auto arr = std::make_unique<float[]>(size);

  VSLStreamStatePtr stream;
  vslNewStream(&stream, RNG, seed());
  vsRngUniform(METHOD, stream, size, arr.get(), min, max);
  return std::vector<float>{arr.get(), arr.get() + size};
}
#else
std::vector<float> getNormalArray(long size, float mean, float std) {
  static std::random_device seed;
  static std::mt19937_64 rng{seed()};
  std::normal_distribution<> dist{mean, std};
  std::vector<float> array(size);

#pragma omp parallel for
  for (auto iter = array.begin(); iter != array.end(); ++iter) {
    *iter = dist(rng);
  }

  return array;
}

std::vector<float> getUniformArray(long size, float min, float max) {
  static std::random_device seed;
  static std::mt19937_64 rng{seed()};
  std::uniform_real_distribution<> dist{min, max};
  std::vector<float> array(size);

#pragma omp parallel for
  for (auto iter = array.begin(); iter != array.end(); ++iter) {
    *iter = dist(rng);
  }

  return array;
}
#endif

std::tuple<int64_t, int64_t> getFeaturesAndSize(ElidedOp elided) {
  auto type = elided.getType();
  auto outShapeAttr = elided->getAttrOfType<ArrayAttr>("output_shape");

  assert(outShapeAttr && "requires attribute `output_shape`");
  // assert(outShapeAttr.size() == 4 && "`output_shape` must be 4D");

  auto mul = std::multiplies<int64_t>();

  SmallVector<int64_t> dims;
  for (auto attr : outShapeAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    assert(intAttr && "dims of `output_shape` must be integer");
    dims.emplace_back(intAttr.getInt());
  }
  auto features = std::accumulate(dims.begin() + 1, dims.end(), 1L, mul);

  auto shape = type.getShape();
  auto size = std::accumulate(shape.begin(), shape.end(), 1L, mul);
  return std::make_tuple(features, size);
}

Value initWeightForInceptionV3Conv2D(ElidedOp elided, OpBuilder& builder) {
  auto attr = truncNormal(elided, 0, 0.1, -2, 2);
  return builder.create<tosa::ConstOp>(elided.getLoc(), elided.getType(), attr);
}

Value initWeightForInceptionV3Linear(ElidedOp elided, OpBuilder& builder) {
  auto attr = truncNormal(elided, 0, 0.1, -2, 2);
  return builder.create<tosa::ConstOp>(elided.getLoc(), elided.getType(), attr);
}

Value initWeightForResNet50Conv2D(ElidedOp elided, OpBuilder& builder) {
  auto attr = kaimingNormal(elided, 0, "fan_out", "relu");
  return builder.create<tosa::ConstOp>(elided.getLoc(), elided.getType(), attr);
}

Value initWeightForResNet50Linear(ElidedOp elided, OpBuilder& builder) {
  auto type = elided.getType();
  auto shape = type.getShape();
  auto size = std::accumulate(shape.begin(), shape.end(), 1L,
                              std::multiplies<int64_t>());

  auto memory = reinterpret_cast<float*>(malloc(size * sizeof(float)));
  auto attr = DenseElementsAttr::get(type, llvm::ArrayRef(memory, size));
  free(memory);

  return builder.create<tosa::ConstOp>(elided->getLoc(), type, attr);
}

Value initWeightForConv2D(ElidedOp elided, OpBuilder& builder) {
  auto func = elided->getParentOfType<func::FuncOp>();
  auto model = func->getAttrOfType<StringAttr>("model");
  if (model) {
    using Fn = function_ref<Value(ElidedOp, OpBuilder&)>;
    auto fn = StringSwitch<Fn>(model.str())
                  .Case("InceptionV3", initWeightForInceptionV3Conv2D)
                  .Case("ResNet50", initWeightForResNet50Conv2D)
                  .Default(nullptr);
    if (fn) {
      return fn(elided, builder);
    }
  }

  auto [features, size] = getFeaturesAndSize(elided);
  auto type = elided.getType();

  auto scale = sqrtf32(2.0 / features);
  std::vector<float> values = getNormalArray(size, 0, 1);
  for (auto iter = values.begin(); iter != values.end(); ++iter) {
    *iter *= scale;
  }

  // TODO: case [batch > 1]

  auto attr = DenseElementsAttr::get(type, llvm::ArrayRef(values));
  return builder.create<tosa::ConstOp>(elided.getLoc(), type, attr);
}

// TODO: refactor
Value initWeightForLinear(ElidedOp elided, OpBuilder& builder) {
  auto func = elided->getParentOfType<func::FuncOp>();
  auto model = func->getAttrOfType<StringAttr>("model");
  if (model) {
    using Fn = function_ref<Value(ElidedOp, OpBuilder&)>;
    auto fn = StringSwitch<Fn>(model.str())
                  .Case("InceptionV3", initWeightForInceptionV3Linear)
                  .Case("ResNet50", initWeightForResNet50Linear)
                  .Default(nullptr);
    if (fn) {
      return fn(elided, builder);
    }
  }

  auto [features, size] = getFeaturesAndSize(elided);
  auto type = elided.getType();

  auto range = sqrtf32(1.0 / features);
  auto values = getUniformArray(size, -range, range);

  // TODO: case [batch > 1]

  auto attr = DenseElementsAttr::get(type, llvm::ArrayRef(values));
  return builder.create<tosa::ConstOp>(elided->getLoc(), type, attr);
}

class ElidedConverter : public OpRewritePattern<ElidedOp> {
  using OpRewritePattern<ElidedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElidedOp elided,
                                PatternRewriter& rewriter) const override {
    auto init = elided->getAttrOfType<StringAttr>("init");
    if (!init) {
      return failure();
    }

    using Fn = function_ref<Value(ElidedOp, PatternRewriter&)>;
    auto fn = StringSwitch<Fn>(init.strref())
                  .Case("conv2d", initWeightForConv2D)
                  .Case("linear", initWeightForLinear)
                  .Default(nullptr);

    if (!fn) {
      return failure();
    }

    rewriter.replaceOp(elided, fn(elided, rewriter));
    return success();
  }
};

class TestConvertElidedToConst
    : public impl::TestConvertElidedToConstBase<TestConvertElidedToConst> {
  void runOnOperation() override {
    auto patterns = RewritePatternSet{&getContext()};
    patterns.add<ElidedConverter>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createTestConvertElidedToConst() {
  return std::make_unique<TestConvertElidedToConst>();
}

}  // namespace ufront
}  // namespace mlir
