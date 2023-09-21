import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        auto_complete_model_config.add_input( {"name": "INPUT0",  "data_type": "TYPE_UINT8", "dims": [-1]})
        auto_complete_model_config.add_output({"name": "OUTPUT0", "data_type": "TYPE_UINT8", "dims": [-1]})
        auto_complete_model_config.set_max_batch_size(0)
        return auto_complete_model_config

    def execute(self, requests):
        responses = []
        for request in requests:
            in_numpy = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            in_str = str(bytes(in_numpy), 'utf8')
            out_str = 'modelB:' + in_str
            out_numpy = np.frombuffer(bytes(out_str, 'utf8'), dtype = np.uint8)
            out_pb = pb_utils.Tensor("OUTPUT0", out_numpy)
            responses.append(pb_utils.InferenceResponse(output_tensors = [out_pb]))
        return responses
