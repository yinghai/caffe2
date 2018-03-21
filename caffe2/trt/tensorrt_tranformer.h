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

#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "onnx/onnx_pb.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace caffe2 {
  class TensorRTTransformer {
    private:
     enum class NodeType : std::int8_t {
       OP = 0,
       TENSOR = 1,
       INPUT = 2,
       OUTPUT = 3
     };

     struct NodeT {
       NodeT(const std::string& n, NodeType t):name(n), type(t) {}
       std::string name;
       NodeType type;
       std::vector<NodeT*> out_nodes;
       const OperatorDef* op{nullptr};
     };

     struct GraphT {
       std::unordered_map<std::string, NodeT> nodes;
       std::vector<NodeT*> inputs;
       std::vector<NodeT*> outputs;
     };

    public:
     OperatorDef BuildTrtOp(
         const std::string& onnx_model_str,
         const std::unordered_map<std::string, std::vector<int>>&
             output_size_hints);

     void LoadNets(const NetDef& init_net, const NetDef& pred_net);

     void TransformSimple(
         NetDef* init_net,
         NetDef* pred_net,
         const std::unordered_map<std::string, TensorShape>& shape_hints);

    private:
      int LoadPredNet(const NetDef& net, int seq);
      void LoadInitNet(const NetDef& net);

      void ClusterToTrtOp(
          const NetDef& init_net,
          const NetDef& pred_net,
          int start,
          int end,
          const std::unordered_set<std::string>& weights,
          const std::unordered_map<std::string, TensorShape>& shape_hints,
          ::ONNX_NAMESPACE::ModelProto* model,
          std::vector<OperatorDef>* new_ops);

      size_t max_batch_size_{50};
      size_t max_workspace_size_{1024*1024*2};
      int verbosity_{2};
      bool debug_builder_{true};
      std::unique_ptr<GraphT> g_{nullptr};
  };
}
