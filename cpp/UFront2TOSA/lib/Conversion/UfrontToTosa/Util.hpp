#include "Conversion/UfrontToTosa/UfrontToTosa.hpp"

namespace mlir {
namespace ufront {

Value transpose(Value tensor, ArrayRef<int64_t> perms, OpBuilder& builder);
Value reduceSum(Value tensor, uint64_t axis, OpBuilder& builder);
Value matmul(Value lhs, Value rhs, OpBuilder& builder);
Value reshape(Value tensor, ArrayRef<int64_t> dims, OpBuilder& builder);
Value constant(double value, Type type, OpBuilder& builder);
Value constantScalar(double value, Type elemTy, OpBuilder& builder);

DenseElementsAttr getDenseFloatAttr(double value, Type type,
                                    OpBuilder& builder);

SmallVector<int64_t> getIntValueFromArrayAttr(ArrayAttr array);

}  // namespace ufront
}  // namespace mlir
