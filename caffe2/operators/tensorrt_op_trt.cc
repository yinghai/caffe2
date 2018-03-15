/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/operators/tensorrt_op_trt.h"
#include "caffe2/core/logging.h"

#include <unordered_map>

namespace caffe2 {

namespace {
// Note that input of trt tensor is in CHW format, while our tensor is NCHW
bool CheckDims(
    const nvinfer1::Dims& nv_dims,
    const std::vector<TIndex>& c2_dims) {
  std::cout << "trt_dims: ";
  for (int i = 0; i < nv_dims.nbDims; ++i) {
    std::cout << nv_dims.d[i] << ",";
  }
  std::cout << std::endl;
  std::cout << "c2_dims: ";
  for (int i = 0; i < c2_dims.size(); ++i) {
    std::cout << c2_dims[i] << ",";
  }
  std::cout << std::endl;

  if (nv_dims.nbDims + 1 != c2_dims.size()) {
    return false;
  }
  for (int i = 0; i < nv_dims.nbDims; ++i) {
    if (nv_dims.d[i] != c2_dims[i + 1]) {
      return false;
    }
  }
  return true;
}

} // namespace

// Upon construction, we build the inference enigne by deserializing from
// protobuf string. And since we know the input/output blobs, we can do the
// binding here too.
TensorRTOp::TensorRTOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CUDAContext>(operator_def, ws),
      logger_((nvinfer1::ILogger::Severity)(
          OperatorBase::GetSingleArgument<int>("log_verbosity", 2))),
      batch_size_(OperatorBase::GetSingleArgument<int>("max_batch_size", 1)) {
  {
    auto engine_string =
        OperatorBase::GetSingleArgument<std::string>("serialized_engine", "");
    CAFFE_ENFORCE(!engine_string.empty(), "Empty serialized TensorRT engine!");
    auto trt_runtime = InferObject(nvinfer1::createInferRuntime(logger_));
    // TODO(support trt plugin factory)
    trt_engine_ = InferObject(trt_runtime->deserializeCudaEngine(
        engine_string.data(), engine_string.size(), nullptr));
  }

  if(!trt_engine_) {
    CAFFE_THROW("Cannot deserialize TensorRT engine!");
  }

  // match and bind the input/output
  std::unordered_map<std::string, int> inputs;
  std::unordered_map<std::string, int> outputs;
  for (int i = 0; i < operator_def.input_size(); ++i) {
    inputs.emplace(operator_def.input(i), i);
    std::cout << "Adding Input: " << operator_def.input(i) << std::endl;
  }
  for (int i = 0; i < operator_def.output_size(); ++i) {
    outputs.emplace(operator_def.output(i), i);
    std::cout << "Adding Output: " << operator_def.output(i) << std::endl;
  }


  int num_bindings = trt_engine_->getNbBindings();
  for (int b = 0; b < num_bindings; ++b) {
    const auto& name = trt_engine_->getBindingName(b);
    nv_dims_.push_back(trt_engine_->getBindingDimensions(b));
    if (trt_engine_->bindingIsInput(b)) {
      std::cout << "Checking TRT input: " << name << std::endl;
      const auto it = inputs.find(name);
      CAFFE_ENFORCE(it != inputs.end(), MakeString("Cannot find trt input: ", name));
      binding_hints_.emplace_back(it->second, true);
    } else {
      const auto it = outputs.find(name);
      CAFFE_ENFORCE(it != outputs.end());
      binding_hints_.emplace_back(it->second, false);
    }
  }

  trt_executor_ = InferObject(trt_engine_->createExecutionContext());
}

bool TensorRTOp::RunOnDevice() {
  CAFFE_ENFORCE(trt_executor_);
  // Decide input batch size
  int N = 0;
  bool first = true;
  for (int i = 0; i < InputSize(); ++i) {
    const auto& input_tensor = Input(i);
    const auto& tensor_dims = input_tensor.dims();
    if (first) {
      N = tensor_dims.front();
      first = false;
    } else {
      CAFFE_ENFORCE(
          N == tensor_dims.front(), "Mismatched batch size in input tensors");
    }
  }
  CAFFE_ENFORCE(N <= batch_size_, "Batch size is too large");
  batch_size_ = N;

  // We need to do the binding at RunOnDevice time because we only know the
  // exact shapes of the tensors now
  bindings_.clear();
  int b = 0;
  for (const auto& p : binding_hints_) {
    const auto& dims = trt_engine_->getBindingDimensions(b++); //nv_dims_[b++];
    if (p.second) {
      // input, check input dimensions
      const auto& input_tensor = Input(p.first);
      const float* input_data = input_tensor.data<float>();
      const auto& tensor_dims = input_tensor.dims();
      CAFFE_ENFORCE(CheckDims(dims, tensor_dims));
      bindings_.push_back((void*)(input_data));
    } else {
      // output, we need to allocate the output tensor
      auto* output_tensor = Output(p.first);
      std::vector<TIndex> tensor_dims;
      tensor_dims.push_back(N);
      for (int i = 0; i < dims.nbDims; ++i) {
        tensor_dims.push_back(dims.d[i]);
      }
      output_tensor->Resize(tensor_dims);
      float* output_data = output_tensor->mutable_data<float>();
      bindings_.push_back((void*)(output_data));
    }
  }

  CAFFE_ENFORCE(bindings_.size() == InputSize() + OutputSize());
  return trt_executor_->execute(batch_size_, &bindings_[0]);
}

OPERATOR_SCHEMA(TensorRT)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
The TensorRT operator is a block-box operator serialized from prebuilt TensorRT
Engine string. It will take the input, do the computation by calling TensorRT
inference engine and generate the outputs.

This is a GPU only operator.
)DOC")
    .Arg(
        "log_verbosity",
        "(int default 0) verbosity of the TensorRt engine log."
        )
    .Arg(
        "serialized_engine",
        "(string default=\"\" blob for serialized TensorRT engine."
        "Note that serialized engine is not compatible across platform and "
        "different TensorRT version."
        )
    .Arg(
        "batch_size",
        "(int default 0) Batch size set by the TensorRT engine builder."
        "It must be no larger than the max_batch_size of the engine builder so "
        "it is better not to edit this manually.");

REGISTER_CUDA_OPERATOR(TensorRT, TensorRTOp);
} // namespace caffe2
