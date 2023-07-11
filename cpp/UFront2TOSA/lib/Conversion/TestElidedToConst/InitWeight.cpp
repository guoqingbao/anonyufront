#include "InitWeight.hpp"

#include <cmath>
#include <numeric>
#include <random>

#include "Random.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace ufront {

template <typename Iter>
int64_t prod(Iter start, Iter end, int64_t init) {
  return std::accumulate(start, end, init, std::multiplies<int64_t>());
}

std::tuple<int64_t, int64_t> calculateFanInAndFanOut(Value tensor) {
  auto type = cast<TensorType>(tensor.getType());
  auto shape = type.getShape();
  auto in = shape[1];
  auto out = shape[0];
  auto field = prod(shape.begin() + 2, shape.end(), 1);
  return std::make_tuple(in * field, out * field);
}

int64_t calculateCorrectFan(Value tensor, StringRef mode) {
  auto [in, out] = calculateFanInAndFanOut(tensor);
  return mode == "fan_in" ? in : out;
}

float calculateGain(StringRef nonlinearity, float a) {
  // TODO: support other nonlinearity
  return StringSwitch<float>(nonlinearity)
      .Case("relu", sqrtf32(2.0))
      .Default(0);
}

Attribute kaimingNormal(Value tensor, float a, StringRef mode,
                        StringRef nonlinearity) {
  auto type = tensor.getType();
  assert(isa<TensorType>(type) && "expected tensor type");

  auto fan = calculateCorrectFan(tensor, mode);
  auto gain = calculateGain(nonlinearity, a);
  auto std = gain / sqrtf32(fan);

  auto shape = cast<TensorType>(type).getShape();
  auto size = prod(shape.begin(), shape.end(), 1);

  std::vector<float> nums(size);
  normal_(nums, 0, std);

  return DenseElementsAttr::get(type, llvm::ArrayRef(nums));
}

float normCdf(float x) { return (1 + std::erf(x / sqrtf32(2))) / 2; }

float erfinv(float x) {
  float tt1, tt2, lnx, sgn;
  sgn = (x < 0) ? -1.0f : 1.0f;

  x = (1 - x) * (1 + x);  // x = 1 - x*x;
  lnx = logf(x);

  tt1 = 2 / (M_PI * 0.147) + 0.5f * lnx;
  tt2 = 1 / (0.147) * lnx;

  return (sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
}

float clamp(float x, float min, float max) {
  return x < min ? min : (x > max ? max : x);
}

Attribute truncNormal(Value tensor, float mean, float std, float min,
                      float max) {
  auto type = tensor.getType();
  assert(isa<TensorType>(type) && "expected tensor type");

  auto lower = normCdf((min - mean) / std);
  auto upper = normCdf((max - mean) / std);

  auto shape = cast<TensorType>(type).getShape();
  auto size = prod(shape.begin(), shape.end(), 1);

  std::vector<float> nums(size);
  uniform_(nums, 2 * lower - 1, 2 * upper - 1);

  for (auto iter = nums.begin(); iter != nums.end(); ++iter) {
    auto x = mean + std * erfinv(*iter) * sqrtf32(2);
    *iter = clamp(x, min, max);
  }

  return DenseElementsAttr::get(type, llvm::ArrayRef(nums));
}

}  // namespace ufront
}  // namespace mlir