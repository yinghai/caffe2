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

#ifndef CAFFE2_OPERATORS_TENSORRT_OP_H
#define CAFFE2_OPERATORS_TENSORRT_OP_H

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#include <NvInfer.h>
#include <iostream>
#include <memory>

namespace caffe2 {

// Logger for GIE info/warning/errors
class TrtLogger : public nvinfer1::ILogger {
  using nvinfer1::ILogger::Severity;

 public:
  TrtLogger(
      Severity verbosity = Severity::kWARNING,
      std::ostream& ostream = std::cout)
      : _verbosity(verbosity), _ostream(&ostream) {}
  void log(Severity severity, const char* msg) override {
    if (severity <= _verbosity) {
      std::string sevstr =
          (severity == Severity::kINTERNAL_ERROR
               ? "    BUG"
               : severity == Severity::kERROR ? "  ERROR"
                                              : severity == Severity::kWARNING
                       ? "WARNING"
                       : severity == Severity::kINFO ? "   INFO" : "UNKNOWN");
      (*_ostream) << "[" << sevstr << "] " << msg << std::endl;
    }
  }

 private:
  Severity _verbosity;
  std::ostream* _ostream;
};

class TensorRTOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  TensorRTOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;
  virtual ~TensorRTOp() noexcept {}

 private:
  TrtLogger logger_;
  int batch_size_;
  std::vector<void*> bindings_;
  std::shared_ptr<nvinfer1::ICudaEngine> trt_engine_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext > trt_executor_{nullptr};
};

} // namespace caffe2

#endif
