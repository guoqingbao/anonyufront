#include "CAPI/TosaConverter.h"

#include <llvm-16/llvm/Support/raw_ostream.h>

#include <cstring>

#include "Conversion/TestElidedToConst/TestElidedToConst.hpp"
#include "Conversion/UfrontToTosa/UfrontToTosa.hpp"
#include "Dialect/Ufront/IR/Ufront.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

const char* ufront_to_tosa(const char* ufront) {
  using namespace mlir;

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<ufront::UfrontDialect>();

  MLIRContext context;
  context.appendDialectRegistry(registry);

  ParserConfig parserConfig{&context};
  auto module = parseSourceString<ModuleOp>(ufront, parserConfig);

  PassManager pm{&context};
  pm.addPass(ufront::createConvertUfrontToTosa());
  pm.addPass(ufront::createTestConvertElidedToConst());

  if (failed(pm.run(*module))) {
    return nullptr;
  }

  std::string mlir;
  llvm::raw_string_ostream ostream{mlir};
  module->print(ostream);

  char* cstr = new char[mlir.size() + 1];
  std::strcpy(cstr, mlir.c_str());

  return cstr;
}
