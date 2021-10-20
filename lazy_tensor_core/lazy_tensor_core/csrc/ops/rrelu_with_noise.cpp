#include "lazy_tensor_core/csrc/ops/rrelu_with_noise.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

RreluWithNoise::RreluWithNoise(const torch::lazy::Value& input, const torch::lazy::Value& seed,
                               const at::Scalar& lower, const at::Scalar& upper,
                               bool training)
    : TsNode(torch::lazy::OpKind(at::aten::rrelu_with_noise), {input, seed},
           lazy_tensors::ShapeUtil::MakeTupleShape(
               {ir::GetShapeFromTsValue(input), GetShapeFromTsValue(input)}),
           /*num_outputs=*/2,
           torch::lazy::MHash(ScalarHash(lower), ScalarHash(upper),
                                     training)),
      lower_(std::move(lower)),
      upper_(std::move(upper)),
      training_(training) {}

NodePtr RreluWithNoise::Clone(OpList operands) const {
  return torch::lazy::MakeNode<RreluWithNoise>(operands.at(0), operands.at(1), lower_,
                                  upper_, training_);
}

std::string RreluWithNoise::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", lower=" << lower_ << ", upper=" << upper_
     << ", training=" << training_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
