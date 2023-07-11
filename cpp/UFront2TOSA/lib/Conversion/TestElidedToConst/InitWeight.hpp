#pragma once

#include <vector>

#include "mlir/IR/Value.h"

namespace mlir {
namespace ufront {

Attribute kaimingNormal(Value tensor, float a, StringRef mode,
                        StringRef nonlinearity);
Attribute truncNormal(Value tensor, float mean, float std, float min,
                      float max);

}  // namespace ufront
}  // namespace mlir
