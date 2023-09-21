##  Example of building and running NVidia Triton Inference server on a CPU-only Docker-less system

Here we have a few example Python models accepting a batch of JSON objects and returning a batch of JSON objects. These models are connected in a pipeline.

```shell
## https://github.com/triton-inference-server/server/blob/5dd9398dd76a90a117ce6b3052e15561337fe88b/build.py#L1006-L1009
#sudo add-apt-repository ppa:mhier/libboost-latest
#sudo apt-get update
#sudo apt install cmake rapidjson-dev libssl-dev libre2-dev libb64-dev libarchive-dev libboost1.81-dev
#git clone https://github.com/triton-inference-server/server --branch r23.08 --single-branch --depth 1
#pushd server
#python3 ./build.py -v --no-container-build --enable-logging --enable-stats --enable-tracing --build-dir="$PWD/build" --backend python  --extra-core-cmake-arg=TRITON_ENABLE_GRPC=OFF --extra-core-cmake-arg=TRITON_ENABLE_HTTP=ON  --extra-core-cmake-arg=TRITON_ENABLE_ENSEMBLE=ON
#export PATH=$PWD/server/build/opt/tritonserver/bin/:$PATH
#sudo ln -s $PWD/build/install/tritonserver /opt
#popd

tritonserver --model-repository $PWD/models

curl -i http://localhost:8000/v2/health/ready
# HTTP/1.1 200 OK

curl -i -X POST localhost:8000/v2/models/modelA/infer -H 'Inference-Header-Content-Length: 138' -H "Content-Type: application/octet-stream" --data-binary '{"inputs":[{"name":"INPUT0","shape":[5],"datatype":"UINT8","parameters":{"binary_data_size":5}}],"parameters":{"binary_data_output":true}}hello'

curl -i -X POST localhost:8000/v2/models/modelB/infer -H 'Inference-Header-Content-Length: 138' -H "Content-Type: application/octet-stream" --data-binary '{"inputs":[{"name":"INPUT0","shape":[5],"datatype":"UINT8","parameters":{"binary_data_size":5}}],"parameters":{"binary_data_output":true}}hello'

```

## References
- https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#cpu-only-build
- https://github.com/triton-inference-server/python_backend
- https://github.com/triton-inference-server/python_backend/tree/main/examples/preprocessing
- https://github.com/triton-inference-server/python_backend/tree/main/examples/auto_complete
