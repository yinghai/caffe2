#include "backend_rep.h"
#include "caffe2/core/common.h"

namespace onnx_caffe2 {

void Caffe2BackendRep::Run(const caffe2::Predictor::TensorVector &inputs,
                          caffe2::Predictor::TensorVector *outputs) {
  if (not predictor_) {
    predictor_ = caffe2::make_unique<caffe2::Predictor>(init_net_, pred_net_);
    init_net_.Clear();
    pred_net_.Clear();
  }

  predictor_->run(inputs, outputs);
  }
}
