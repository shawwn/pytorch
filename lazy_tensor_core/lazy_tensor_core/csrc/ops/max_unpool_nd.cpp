#include "lazy_tensor_core/csrc/ops/max_unpool_nd.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

c10::Symbol MaxUnpoolNdSymbol(lazy_tensors::int64 spatial_dim_count) {
  switch (spatial_dim_count) {
    case 2:
      return at::aten::max_unpool2d;
    case 3:
      return at::aten::max_unpool3d;
    default:
      LTC_ERROR() << "Invalid number of spatial dimensions: "
                  << spatial_dim_count;
  }
}

}  // namespace

MaxUnpoolNd::MaxUnpoolNd(const torch::lazy::Value& input, const torch::lazy::Value& indices,
                         std::vector<lazy_tensors::int64> output_size)
    : TsNode(torch::lazy::OpKind(MaxUnpoolNdSymbol(output_size.size())), {input, indices},
           /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr MaxUnpoolNd::Clone(OpList operands) const {
  return torch::lazy::MakeNode<MaxUnpoolNd>(operands.at(0), operands.at(1), output_size_);
}

std::string MaxUnpoolNd::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=("
     << lazy_tensors::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
