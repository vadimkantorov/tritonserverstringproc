import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_name = args['model_name']

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        auto_complete_model_config.add_input( {"name": "text_input",  "data_type": "TYPE_STRING", "dims": [-1]})
        auto_complete_model_config.add_output({"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]})
        auto_complete_model_config.set_max_batch_size(0)
        return auto_complete_model_config

    def execute(self, requests):
        responses = []
        for request in requests:
            in_numpy = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()
            assert np.object_ == in_numpy.dtype, 'in this demo, triton passes in a numpy array of size 1 with object_ dtype'
            assert 1 == len(in_numpy), 'in this demo, only supporting a single string per inference request'
            assert bytes == type(in_numpy.tolist()[0]), 'this object encapsulates a byte-array'
            str1 = in_numpy.tolist()[0].decode('utf-8')
            str2 = self.model_name + ': ' + str1 + ' World'

            out_numpy = np.array([str2.encode('utf-8')], dtype = np.object_)
            out_pb = pb_utils.Tensor("text_output", out_numpy)
            responses.append(pb_utils.InferenceResponse(output_tensors = [out_pb]))
        return responses
