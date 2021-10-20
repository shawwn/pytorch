#include "lazy_tensor_core/csrc/ops/unselect.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Unselect::Unselect(const torch::lazy::Value& target, const torch::lazy::Value& source,
                   lazy_tensors::int64 dim, lazy_tensors::int64 start,
                   lazy_tensors::int64 end, lazy_tensors::int64 stride)
    : TsNode(ltc_unselect, {target, source}, ir::GetShapeFromTsValue(target),
           /*num_outputs=*/1,
           torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

NodePtr Unselect::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Unselect>(operands.at(0), operands.at(1), dim_, start_, end_,
                            stride_);
}

std::string Unselect::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
