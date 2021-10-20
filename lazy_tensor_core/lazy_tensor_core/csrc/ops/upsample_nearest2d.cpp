#include "lazy_tensor_core/csrc/ops/upsample_nearest2d.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_join.h"
#include "lazy_tensors/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

UpsampleNearest::UpsampleNearest(const torch::lazy::Value& input,
                                 std::vector<lazy_tensors::int64> output_size)
    : TsNode(torch::lazy::OpKind(at::aten::upsample_nearest2d), {input},
           /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr UpsampleNearest::Clone(OpList operands) const {
  return torch::lazy::MakeNode<UpsampleNearest>(operands.at(0), output_size_);
}

std::string UpsampleNearest::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=("
     << lazy_tensors::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
