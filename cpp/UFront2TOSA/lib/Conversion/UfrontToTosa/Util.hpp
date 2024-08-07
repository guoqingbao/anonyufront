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
DenseElementsAttr getDenseIntegerAttr(int64_t value, Type type,
                                      OpBuilder& builder);
DenseElementsAttr getDenseElementsAttr(double value, Type type,
                                       OpBuilder& builder);

SmallVector<int64_t> getIntValueFromArrayAttr(ArrayAttr array);
std::optional<Value> getConstTensor(OpBuilder &rewriter, Operation *op,
                                    float value);
Value approximateErfOp(OpBuilder &rewriter,
                                Operation *op, Value x);

}  // namespace ufront
}  // namespace mlir
