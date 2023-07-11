#include "Conversion/Passes.hpp"
#include "Dialect/Ufront/IR/Ufront.hpp"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
  using namespace mlir;

  auto registry = DialectRegistry{};
  registerAllDialects(registry);
  registry.insert<ufront::UfrontDialect>();

  registerConversionPasses();

  return failed(
      MlirOptMain(argc, argv, "`Ufront` to `Tosa` converter\n", registry));
}
