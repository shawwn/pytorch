#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ConstantPadNd::ConstantPadNd(const torch::lazy::Value& input,
                             std::vector<lazy_tensors::int64> pad,
                             const at::Scalar& value)
    : TsNode(torch::lazy::OpKind(at::aten::constant_pad_nd), {input},
           /*num_outputs=*/1,
           torch::lazy::MHash(pad, ScalarHash(value))),
      pad_(std::move(pad)),
      value_(value) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr ConstantPadNd::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ConstantPadNd>(operands.at(0), pad_, value_);
}

std::string ConstantPadNd::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", pad=(" << lazy_tensors::StrJoin(pad_, ", ")
     << ")"
     << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
