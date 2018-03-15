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
#include "caffe2/proto/caffe2.pb.h"
#include <string>

namespace caffe2 {

  class TensorRTTransformer {
    public:
     OperatorDef BuildTrtOp(
         const std::string& onnx_model_str,
         const std::vector<std::string>& inputs,
         const std::vector<std::string>& outputs);

    private:
      size_t max_batch_size_{50};
      size_t max_workspace_size_{1024*1024*2};
      int verbosity_{2};
      bool debug_builder_{true};
  };
}
