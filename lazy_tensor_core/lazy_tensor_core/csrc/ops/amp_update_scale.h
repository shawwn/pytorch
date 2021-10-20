#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AmpUpdateScale : public TsNode {
 public:
  AmpUpdateScale(const torch::lazy::Value& current_scale, const torch::lazy::Value& growth_tracker,
                 const torch::lazy::Value& found_inf, double scale_growth_factor,
                 double scale_backoff_factor, int growth_interval);

  NodePtr Clone(OpList operands) const override;

  double scale_growth_factor() const { return scale_growth_factor_; }

  double scale_backoff_factor() const { return scale_backoff_factor_; }

  int growth_interval() const { return growth_interval_; }

 private:
  double scale_growth_factor_;
  double scale_backoff_factor_;
  int growth_interval_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
