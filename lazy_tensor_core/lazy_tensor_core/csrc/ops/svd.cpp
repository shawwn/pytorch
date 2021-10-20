#include "lazy_tensor_core/csrc/ops/svd.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

SVD::SVD(const torch::lazy::Value& input, bool some, bool compute_uv)
    : TsNode(torch::lazy::OpKind(at::aten::svd), {input},
           /*num_outputs=*/3, torch::lazy::MHash(some, compute_uv)),
      some_(some),
      compute_uv_(compute_uv) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr SVD::Clone(OpList operands) const {
  return torch::lazy::MakeNode<SVD>(operands.at(0), some_, compute_uv_);
}

std::string SVD::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", some=" << some_
     << ", compute_uv=" << compute_uv_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
