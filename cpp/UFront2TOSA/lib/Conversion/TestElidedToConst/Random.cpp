#include "Random.hpp"

#include <random>

std::vector<float> normal(int64_t size, float mean, float std) {
  std::vector<float> nums(size);
  normal_(nums, mean, std);
  return nums;
}

std::vector<float> uniform(int64_t size, float min, float max) {
  std::vector<float> nums(size);
  uniform_(nums, min, max);
  return nums;
}

void normal_(std::vector<float>& nums, float mean, float std) {
#ifdef USEMKL
  static constexpr auto RNG = VSL_BRNG_MT19937;
  static constexpr auto METHOD = VSL_RNG_METHOD_GAUSSIAN_ICDF;
  static std::random_device seed;

  auto arr = std::make_unique<float[]>(size);

  VSLStreamStatePtr stream;
  vslNewStream(&stream, RNG, seed());
  vsRngGaussian(METHOD, stream, size, arr.get(), mean, std);

  std::copy(arr.get(), arr.get() + size, nums.begin());
#else
  static std::random_device seed;
  static std::mt19937_64 rng{seed()};
  std::normal_distribution<> dist{mean, std};

#pragma omp parallel for
  for (auto iter = nums.begin(); iter != nums.end(); ++iter) {
    *iter = dist(rng);
  }
#endif
}

void uniform_(std::vector<float>& nums, float min, float max) {
#ifdef USEMKL
  static constexpr auto RNG = VSL_BRNG_MT19937;
  static constexpr auto METHOD = VSL_RNG_METHOD_UNIFORM_STD;
  static std::random_device seed;

  auto arr = std::make_unique<float[]>(size);

  VSLStreamStatePtr stream;
  vslNewStream(&stream, RNG, seed());
  vsRngUniform(METHOD, stream, size, arr.get(), min, max);

  std::copy(arr.get(), arr.get() + size, nums.begin());
#else
  static std::random_device seed;
  static std::mt19937_64 rng{seed()};
  std::uniform_real_distribution<> dist{min, max};

#pragma omp parallel for
  for (auto iter = nums.begin(); iter != nums.end(); ++iter) {
    *iter = dist(rng);
  }
#endif
}