#include "lazy_tensor_core/csrc/ops/cumsum.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

CumSum::CumSum(const torch::lazy::Value& input, lazy_tensors::int64 dim,
               c10::optional<at::ScalarType> dtype)
    : TsNode(torch::lazy::OpKind(at::aten::cumsum), {input},
           /*num_outputs=*/1,
           torch::lazy::MHash(dim, OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr CumSum::Clone(OpList operands) const {
  return torch::lazy::MakeNode<CumSum>(operands.at(0), dim_, dtype_);
}

std::string CumSum::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  if (dtype_) {
    ss << ", dtype=" << *dtype_;
  }
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
