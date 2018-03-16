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


#include "caffe2/trt/tensorrt_tranformer.h"
#include "caffe2/trt/trt_utils.h"
#include "onnx2trt.hpp"
#include "NvInfer.h"

#include <iostream>

namespace caffe2 {

OperatorDef TensorRTTransformer::BuildTrtOp(
    const std::string& onnx_model_str,
    const std::unordered_map<std::string, std::vector<int>>&
        output_size_hints) {
  OperatorDef op;
  op.set_type("TensorRT");

  TrtLogger logger;
  auto trt_builder = InferObject(nvinfer1::createInferBuilder(logger));
  auto trt_network = InferObject(trt_builder->createNetwork());
  auto importer = InferObject(onnx2trt::createImporter(trt_network.get()));
  auto status = importer->import(onnx_model_str.data(), onnx_model_str.size(), false);
  if (status.is_error()) {
    std::cerr << "TensorRTTransformer ERROR: " << status.file() << ":"
              << status.line() << " In function " << status.func() << ":\n"
              << "[" << status.code() << "] " << status.desc() << std::endl;
  }
  trt_builder->setMaxBatchSize(max_batch_size_);
  trt_builder->setMaxWorkspaceSize(max_workspace_size_);
  trt_builder->setDebugSync(debug_builder_);
  auto trt_engine =
      InferObject(trt_builder->buildCudaEngine(*trt_network.get()));

  // Set up inputs/outputs
  int num_bindings = trt_engine->getNbBindings();
  for (int b = 0; b < num_bindings; ++b) {
    const auto& name = trt_engine->getBindingName(b);
    if (trt_engine->bindingIsInput(b)) {
      op.add_input(name);
    } else {
      op.add_output(name);
    }
  }

  auto engine_plan = InferObject(trt_engine->serialize());

  auto* serialized_engine_arg = op.add_arg();
  serialized_engine_arg->set_s("");
  serialized_engine_arg->set_name("serialized_engine");
  auto* s = serialized_engine_arg->mutable_s();
  s->assign((char*)engine_plan->data(), engine_plan->size());

  auto* max_batch_size_arg = op.add_arg();
  max_batch_size_arg->set_name("max_batch_size");
  max_batch_size_arg->set_i(max_batch_size_);

  auto* verbosity_arg = op.add_arg();
  verbosity_arg->set_name("log_verbosity");
  verbosity_arg->set_i(verbosity_);

  auto* output_size_hints_arg = op.add_arg();
  output_size_hints_arg->set_name("output_size_hints");
  for(const auto& o: op.output()) {
    const auto it = output_size_hints.find(o);
    if (it != output_size_hints.end()) {
      const auto& dims = it->second;
      for (const auto& i : dims) {
        output_size_hints_arg->add_ints(i);
      }
      // Add an extra -1 to indicate the end
      output_size_hints_arg->add_ints(-1);
    }
  }

  return op;
}
}
