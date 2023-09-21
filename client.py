# python -m pip install tritonclient[all] --user

import sys
import json
import numpy as np
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient("localhost:8000", verbose = True)
model_name = sys.argv[1]
input_str = json.dumps(dict(hi = sys.argv[2]))

input_arr = np.frombuffer(bytes(input_str, 'utf8'), dtype = np.uint8)

inputs = [httpclient.InferInput("INPUT0", input_arr.shape, "UINT8")]
inputs[0].set_data_from_numpy(input_arr, binary_data=True)


res = triton_client.infer(model_name=model_name, inputs=inputs)

output_arr = res.as_numpy('OUTPUT0')
output_str = str(bytes(output_arr), 'utf8')

print(output_str)
