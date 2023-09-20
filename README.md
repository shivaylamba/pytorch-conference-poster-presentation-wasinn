# pytorch-conference

## Pytorch WASI NN Mobilenet example: 
1. 
```
 cd pytorch-mobilenet-image/
```
2. Install Wasmedge with WASINN-Pytorch
```
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasi_nn-pytorch
source $HOME/.wasmedge/env
```
3. The WASI-NN plug-in with PyTorch backend depends on the libtorch C++ library to perform AI/ML computations. You need to install the PyTorch 1.8.2 LTS dependencies for it to work properly.
```
export PYTORCH_VERSION="1.8.2"
# For the Ubuntu 20.04 or above, use the libtorch with cxx11 abi.
export PYTORCH_ABI="libtorch-cxx11-abi"
# For the manylinux2014, please use the without cxx11 abi version:
#   export PYTORCH_ABI="libtorch"
curl -s -L -O --remote-name-all https://download.pytorch.org/libtorch/lts/1.8/cpu/${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip
unzip -q "${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
rm -f "${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)/libtorch/lib
```
4. Install Rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup target add wasm32-wasi
```
5. Navigate inside Rust Directory
```
cd rust
cargo build --target=wasm32-wasi --release
```

6. Come outside and run wasmedgec
```
cd .. 
wasmedgec rust/target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image.wasm wasmedge-wasinn-example-mobilenet-image-aot.wasm
wasmedgec rust/target/wasm32-wasi/release/wasmedge-wasinn-example-mobilenet-image-named-model.wasm wasmedge-wasinn-example-mobilenet-image-named-model-aot.wasm
```
7. ## Run

### Generate Model

First generate the fixture of the pre-trained mobilenet with the script:

```bash
pip3 install torch==1.8.2 torchvision==0.9.2 pillow --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
# generate the model fixture
python3 gen_mobilenet_model.py
```

(Or you can use the pre-generated one at [`mobilenet.pt`](mobilenet.pt))

### Test Image

The testing image `input.jpg` is downloaded from <https://github.com/bytecodealliance/wasi-nn/raw/main/rust/examples/images/1.jpg> with license Apache-2.0

### Generate Tensor

If you want to generate the [raw tensor](image-1x3x224x224.rgb), you can run:

```bash
pip3 install torchvision
python3 gen_tensor.py input.jpg image-1x3x224x224.rgb
```

### Execute

Users should [install the WasmEdge with WASI-NN PyTorch backend plug-in](https://wasmedge.org/docs/start/install#wasi-nn-plug-in-with-pytorch-backend).

Execute the WASM with the `wasmedge` with PyTorch supporting:

- Case 1:

```bash
wasmedge --dir .:. wasmedge-wasinn-example-mobilenet-image.wasm mobilenet.pt input.jpg
```

You will get the output:

```console
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 602112
Executed graph inference
   1.) [954](20.6681)banana
   2.) [940](12.1483)spaghetti squash
   3.) [951](11.5748)lemon
   4.) [950](10.4899)orange
   5.) [953](9.4834)pineapple, ananas
```

- Case 2: Apply named model feature
> requirement wasi-nn >= 0.5.0 and WasmEdge-plugin-wasi_nn-(*) >= 0.13.4 and  
> --nn-preload argument format follow <name>:<encoding>:<target>:<model_path>

```bash
wasmedge --dir .:. --nn-preload demo:PyTorch:CPU:mobilenet.pt wasmedge-wasinn-example-mobilenet-image-named-model.wasm demo input.jpg
```

You will get the same output:

```console
Loaded graph into wasi-nn with ID: 0
Created wasi-nn execution context with ID: 0
Read input tensor, size in bytes: 602112
Executed graph inference
   1.) [954](20.6681)banana
   2.) [940](12.1483)spaghetti squash
   3.) [951](11.5748)lemon
   4.) [950](10.4899)orange
   5.) [953](9.4834)pineapple, ananas
```

