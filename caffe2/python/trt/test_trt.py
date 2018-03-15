# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import onnx
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info, make_model
from onnx.backend.base import namedtupledict
import caffe2.python.onnx.backend as c2
from caffe2.python.onnx.workspace import Workspace
import numpy as np

import caffe2.python._import_c_extension as C

# Note that ONNX-TRT enforce an NCHW input!!!!

def test_relu_graph():
    X = np.random.randn(1, 1, 3, 2).astype(np.float32)
    node_def = make_node("Relu", ["X"], ["Y"])
    Y_c2 = c2.run_node(node_def, {"X": X})
    graph_def = make_graph(
        [node_def],
        name="test",
        inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 3, 2])],
        outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1, 3, 2])])
    model_def = make_model(graph_def, producer_name='relu-test')
    #print("Onnx Model: {}".format(model_def))
    op_inputs = [x.name for x in graph_def.input]
    op_outputs = [x.name for x in graph_def.output]
    trt_str = C.onnx_to_trt_op(model_def.SerializeToString(), op_inputs, op_outputs)
    op = caffe2_pb2.OperatorDef()
    op.ParseFromString(trt_str)
    device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
    op.device_option.CopyFrom(device_option)
    #print("{}".format(op))
    Y_trt = None
    with Workspace(), core.DeviceScope(device_option):  # temporary!
        workspace.FeedBlob("X", X)
        workspace.RunOperatorsOnce([op])
        output_values = [workspace.FetchBlob(name) for name in op_outputs]
        Y_trt = namedtupledict('Outputs', op_outputs)(*output_values)
    print("{}".format(Y_c2))
    print("{}".format(Y_trt))
    np.testing.assert_almost_equal(Y_c2, Y_trt)


if __name__ == '__main__':
    test_relu_graph()
