#include "UfrontToTosa.hpp"

#include "Patterns.hpp"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ufront {

std::unique_ptr<Pass> createConvertUfrontToTosa() {
  return std::make_unique<ConvertUfrontToTosa>();
}

void ConvertUfrontToTosa::runOnOperation() {
  auto patterns = RewritePatternSet{&getContext()};
  populateConvertUfrontToTosaPatterns(patterns);

  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace ufront
}  // namespace mlir
