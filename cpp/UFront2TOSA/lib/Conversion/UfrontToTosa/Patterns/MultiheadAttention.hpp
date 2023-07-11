#include "../Patterns.hpp"
#include "../Util.hpp"
#include "Conversion/Passes.hpp"

namespace mlir::ufront::multihead_attention {

using Tensor = TypedValue<TensorType>;

template <typename... T>
using Tuple = std::tuple<T...>;

constexpr auto None = std::nullopt;

struct ForwardConfig {
  Tensor query;
  Tensor key;
  Tensor value;
  int64_t embedDimToCheck;
  int64_t numHeads;
  Optional<Tensor> inProjWeight;
  Optional<Tensor> inProjBias;
  Optional<Tensor> biasK;
  Optional<Tensor> biasV;
  bool addZeroAttn;
  double dropout;
  Tensor outProjWeight;
  Optional<Tensor> outProjBias;
  bool training = true;
  Optional<Tensor> keyPaddingMask = None;
  bool needWeights = true;
  Optional<Tensor> attnMask = None;
  bool useSeparateProjWeight = false;
  Optional<Tensor> qProjWeight = None;
  Optional<Tensor> kProjWeight = None;
  Optional<Tensor> vProjWeight = None;
  Optional<Tensor> staticK = None;
  Optional<Tensor> staticV = None;
  bool averageAttnWeights = true;
  bool isCasual = false;
};

auto forward(ForwardConfig config, OpBuilder& builder)
    -> Tuple<Tensor, Optional<Tensor>>;

}  // namespace mlir::ufront::multihead_attention
