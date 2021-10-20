#include "lazy_tensor_core/csrc/ops/amp_update_scale.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(const torch::lazy::Value& growth_tracker,
                                    const torch::lazy::Value& current_scale) {
  return lazy_tensors::ShapeUtil::MakeTupleShape(
      {ir::GetShapeFromTsValue(growth_tracker), GetShapeFromTsValue(current_scale)});
}

}  // namespace

AmpUpdateScale::AmpUpdateScale(const torch::lazy::Value& current_scale,
                               const torch::lazy::Value& growth_tracker,
                               const torch::lazy::Value& found_inf,
                               double scale_growth_factor,
                               double scale_backoff_factor, int growth_interval)
    : TsNode(torch::lazy::OpKind(at::aten::_amp_update_scale_),
           {current_scale, growth_tracker, found_inf},
           NodeOutputShape(growth_tracker, current_scale),
           /*num_outputs=*/2),
      scale_growth_factor_(scale_growth_factor),
      scale_backoff_factor_(scale_backoff_factor),
      growth_interval_(growth_interval) {}

NodePtr AmpUpdateScale::Clone(OpList operands) const {
  return torch::lazy::MakeNode<AmpUpdateScale>(operands[0], operands[1], operands[2],
                                  scale_growth_factor_, scale_backoff_factor_,
                                  growth_interval_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
