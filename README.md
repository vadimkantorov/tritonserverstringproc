##  Example of building and running NVidia Triton Inference server on a CPU-only Docker-less system

Here we have a few example Python models accepting a batch of JSON objects and returning a batch of JSON objects. These models are connected in a pipeline.

## Installation of `triton-inference-server` from source
See [buildtritoninferenceserver.yml](./.github/workflows/buildtritoninferenceserver.yml) for steps

```shell
#install or extract tritonserver into /opt/tritonserver
sudo ln -s $PWD/build/install/tritonserver /opt
export PATH=/opt/tritonserver/bin/:$PATH

tritonserver --model-repository $PWD/models --log-verbose=1

curl -i http://localhost:8000/v2/health/ready
# HTTP/1.1 200 OK

curl -i -X POST localhost:8000/v2/models/modelA/infer -H 'Inference-Header-Content-Length: 138' -H "Content-Type: application/octet-stream" --data-binary '{"inputs":[{"name":"INPUT0","shape":[5],"datatype":"UINT8","parameters":{"binary_data_size":5}}],"parameters":{"binary_data_output":true}}hello'

curl -i -X POST localhost:8000/v2/models/modelB/infer -H 'Inference-Header-Content-Length: 138' -H "Content-Type: application/octet-stream" --data-binary '{"inputs":[{"name":"INPUT0","shape":[5],"datatype":"UINT8","parameters":{"binary_data_size":5}}],"parameters":{"binary_data_output":true}}hello'

```

## References
- https://github.com/triton-inference-server/python_backend
- https://github.com/triton-inference-server/python_backend/tree/main/examples/preprocessing
- https://github.com/triton-inference-server/python_backend/tree/main/examples/auto_complete
- https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#cpu-only-build
