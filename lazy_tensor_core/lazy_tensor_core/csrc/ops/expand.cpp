#include "lazy_tensor_core/csrc/ops/expand.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Expand::Expand(const torch::lazy::Value& input, std::vector<lazy_tensors::int64> size,
               bool is_scalar_expand)
    : TsNode(torch::lazy::OpKind(at::aten::expand), {input},
           /*num_outputs=*/1,
           torch::lazy::MHash(size, is_scalar_expand)),
      size_(std::move(size)),
      is_scalar_expand_(is_scalar_expand) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Expand::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Expand>(operands.at(0), size_, is_scalar_expand_);
}

std::string Expand::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", size=(" << lazy_tensors::StrJoin(size_, ", ")
     << "), is_scalar_expand=" << is_scalar_expand_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
