#pragma once

#include "caffe2/core/predictor.h"
#include "caffe2/proto/caffe2.pb.h"

#include <memory>

namespace onnx_caffe2 {
class Caffe2BackendRep {
 public:
  void Run(
      const caffe2::Predictor::TensorVector& inputs,
      caffe2::Predictor::TensorVector* outputs);
  void RunMap(
      const caffe2::Predictor::TensorMap& inputs,
      caffe2::Predictor::TensorVector* outputs);

  caffe2::NetDef& init_net() {
    return init_net_;
  }
  caffe2::NetDef& pred_net() {
    return pred_net_;
  }

  const caffe2::NetDef& init_net() const {
    return init_net_;
  }
  const caffe2::NetDef& pred_net() const {
    return pred_net_;
  }

 private:
  void CheckInit();

  caffe2::NetDef init_net_;
  caffe2::NetDef pred_net_;
  std::unique_ptr<caffe2::Predictor> predictor_{nullptr};
};
} // namespace onnx_caffe2
