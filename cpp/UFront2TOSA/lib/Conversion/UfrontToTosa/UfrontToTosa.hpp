#include "Conversion/UfrontToTosa/UfrontToTosa.hpp"
#include "Dialect/Ufront/IR/Ufront.hpp"

namespace mlir {
namespace ufront {

class ConvertUfrontToTosa
    : public impl::ConvertUfrontToTosaBase<ConvertUfrontToTosa> {
  void runOnOperation() override;
};

}  // namespace ufront
}  // namespace mlir
