#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensors/computation_client/computation_client.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class DeviceData : public TsNode {
 public:
  DeviceData(std::shared_ptr<lazy_tensors::client::Data> data);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::shared_ptr<lazy_tensors::client::Data>& data() const {
    return data_;
  }

  static const DeviceData* Cast(const torch::lazy::Node* node);

 private:
  std::shared_ptr<lazy_tensors::client::Data> data_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
