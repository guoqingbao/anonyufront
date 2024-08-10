# UFront
Unified MLIR Frontend for Deep Learning 

## How to reproduce the results?
### Option 1 (recommended):

Experiencing UFront on Kaggle (for model compilation, performance comparison, ImageNet inference, accuracy validation, etc.)

Run the anonymous UFront online tests in Kaggle using the links below, **be sure to login** to use full functionality and free GPU (T4x2) resources.

https://www.kaggle.com/code/anomyuser/ufront-test/

https://www.kaggle.com/code/anomyuser/test-torch

https://www.kaggle.com/code/anomyuser/test-tf

**Note**: the results on Kaggle may slightly different from the paper reported because of different CPU and GPU used.

**Important:** Access GPU at no cost or turn on an internet connection. Need to **login** and **Get phone verified** in Kaggle.

**The Internet Connection** in the Kaggle notebook need to be **turned on** to allow package download.

### Option 2 (test locally with jupyter notebook):
Execute provided jupyter notebooks in the examples folder, be sure to install dependencies:
```sh
examples/ufront_test.ipynb

examples/test_tf.ipynb

examples/test_torch.ipynb
```

### Option 3 (suitable for debug):
Execute python scripts, you may install corresponding dependencies manually.

```sh
python examples/torch_test.py
python examples/bert_test.py
python examples/lstm_test.py
python examples/keras_test.py
python examples/onnx_test.py
```


### Option 4

ImageNet inference with UFront locally

1) Download ImageNet validation set (about 2GB, named "imagenet1kvalid") from kaggle.com and extract it to a local folder

2) In the example/test_accuracy.ipynb (jupyter notebook), change the root path to the parent folder of "imagenet1kvalid", execute all notebook cells (depend on your local Python version, you may install different UFront packages).
   

## Examples' dependencies
### Install pre-build ufront package
In Ubuntu 20.04 or 22.04, download corresponding ufront package in the release folder and install.

Install any of the following packages according to your default Python version.
```sh
pip install ufront-0.1.1-cp37-cp37m-manylinux_2_28_x86_64.whl #for Python3.7

pip install ufront-0.1.1-cp38-cp38-manylinux_2_28_x86_64.whl #for Python3.8

pip install ufront-0.1.1-cp39-cp39-manylinux_2_28_x86_64.whl #for Python3.9

pip install ufront-0.1.1-cp310-cp310-manylinux_2_28_x86_64.whl #for Python3.10

pip install ufront-0.1.1-cp311-cp311-manylinux_2_28_x86_64.whl #for Python3.11
```

### Install Execution Backend (IREE)
Recommended stable IREE
```sh
pip install iree-compiler==20230524.529 iree-runtime==20230524.529 
```

If you want to experience IREE-TF, install:
```sh
pip install iree-compiler==20230815.614 iree-runtime==20230815.614
```

If you want to experience new features of Torch-MLIR (under Python 3.11 and CUDA 12), install: 
```sh
pip install iree_compiler==20240129.785 iree_runtime==20240129.785
```

For older Python like Python3.7 you may install previous iree because recent iree release does not provide Python 3.7 support:
```sh
pip install iree-compiler==20230330.474 -f https://github.com/iree-org/iree/releases/download/candidate-20230330.474/iree_compiler-20230330.474-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip iree-runtime==20230330.474 -f https://github.com/iree-org/iree/releases/download/candidate-20230330.474/iree_runtime-20230330.474-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## How to build UFront from the source code?

### Code structure
1. `python folder` (written in Python): model tracing
2. `src folder` (written in Rust): unified interface, type inference, high-level IR generation
3. `cpp folder` (written in C++): high-level IR to low-level IR conversion with MLIR pipeline (i.e., UFront2TOSA, or TOSA Convertor)

### Build TOSA Convertor first
Going to folder `cpp/UFront2TOSA`, build the TOSA Convertor according to the subproject's README.

### Build UFront release package (assume you have maturin installed)
In the `main` folder, execute the following command (you may change `python3.9` to your python version)

```
maturin build --release -i python3.9
```

Maturin will build UFront package into folder `target/wheels/`

**Note:** assume you have Rust compiler and Maturin installed, refer to:

https://www.rust-lang.org/tools/install

https://www.maturin.rs/installation 

### Install UFront package on the target machine (with GPU support)
In the `main` folder, execute the following command (you may change `.whl` to the package you built)

```
pip install target/wheels/ufront-0.1.1-cp39-cp39-manylinux_2_28_x86_64.whl
```

### Perform the tests
`Option 1:` upload the ufront package you built to online Kaggle environment (previously given), install the package and perform the online test using free GPU resources;

`Option 2:` install the ufront package you built on a target machine (with GPU support), perform the offline test using the given Jupyter Notebooks.


## Trouble shootings
### 0. UFront build error
 ```
 = note: /usr/bin/ld: cannot find -lUfrontCAPI
          collect2: error: ld returned 1 exit status
 ``` 

You need to `first build TOSA convertor` (folder cpp/UFront2TOSA) then build UFront package because the package is relying on the convertor.

### 1. CUDA RuntimeError

If you experiencing the following error, you need to upgrade NVidia Driver and CUDA; or you can lower IREE to a lower version, e.g., 20230330.474 (howover, the results may slightly different from reported). The PTX code generated by recent IREE is not compatible with old NVidia Driver.

RuntimeError: Error creating vm context with modules:main_checkout/runtime/src/iree/hal/drivers/cuda/native_executable.c:99: INTERNAL；CUDA driver error 'CUDA_ERROR_UNSUPPORTED_PTX_VERSION' (222):the provided PTX was compiled with an unsupported toolchain.;while invoking native function hal.executable.create; while calling import;

### 2. Mismatched IREE and CUDA

For **CUDA 11**, the stable IREE is:
```sh
pip install iree-compiler==20230524.529 iree-runtime==20230524.529 
pip install iree-tools-tf==20230524.529  iree-tools-tflite==20230524.529
```

For **CUDA 12**, the compatible IREE are:

Python 3.10
```sh
pip install iree-compiler==20230815.614 iree-runtime==20230815.614
```

Python 3.11
```sh
pip install iree_compiler==20240129.785 iree_runtime==20240129.785
```

However, some IREE runtime version like v20230815.614 has problem of 'missing iree-benchmark-module', the resolution is:

1) Find if iree-benchmark-module available in _runtime_libs folder
```sh
ls /opt/conda/lib/python3.10/site-packages/iree/_runtime_libs/
```
2) Copy it to the runtime folder
```sh
cp /opt/conda/lib/python3.10/site-packages/iree/_runtime_libs/iree-benchmark-module /opt/conda/lib/python3.10/site-packages/iree/runtime/
```

### Issues related to Torch-MLIR, IREE-TF and ONNX-MLIR
Torch-MLIR is bunded with torch (dev), to compile torch models using Torch-MLIR, you need to download both torch-mlir and torch package from their release website, for example https://github.com/llvm/torch-mlir/releases/tag/snapshot-20240127.1096 The installed torch-mlir also need to be compatible with IREE. Recent release of torch-mlir requires torchvision 0.18.0
```sh
!pip install https://github.com/llvm/torch-mlir/releases/download/snapshot-20240127.1096/torch_mlir-20240127.1096-cp311-cp311-linux_x86_64.whl --no-dependencies
!pip install https://github.com/llvm/torch-mlir/releases/download/snapshot-20240127.1096/torch-2.3.0.dev20240122+cpu-cp311-cp311-linux_x86_64.whl --no-dependencies
!pip install https://download.pytorch.org/whl/cpu/torchvision-0.18.0%2Bcpu-cp311-cp311-linux_x86_64.whl --no-dependencies
```

The compatible IREE backend for recent torch-mlir release (Python 3.11 environment) is:
```sh
!pip install iree_compiler==20240129.785 iree_runtime==20240129.785
```

If you are using Python 3.9, Python 3.10 or previous torch-mlir release, you may use previous IREE backend:
```sh
!pip install iree-compiler==20230815.614 iree-runtime==20230815.614
```
```sh
!pip install iree-compiler==20230524.529 iree-runtime==20230524.529
```

IREE-TF and some Tensorflow models have the following dependencies
```sh
pip uninstall tensorflow -y
pip install tensorflow-cpu==2.13.0
pip install tensorflow-addons==0.21.0
pip install validators
pip install scipy
pip install opencv-python
apt install libgl1 -y
```

The compatible IREE backend for tensorflow 2.13, 2.14 and 2.15 is:
```sh
pip install iree-compiler==20230815.614 iree-runtime==20230815.614
```
```sh
pip install iree-compiler==20230524.529 iree-runtime==20230524.529
```

If you encouter problems for compiling models using IREE-TF, you may install the IREE-TF tools and other versions of iree-runtime:
```sh
pip install iree-tools-tf==20230524.529  iree-tools-tflite==20230524.529

# Python 3.9
pip install https://github.com/iree-org/iree/releases/download/candidate-20230816.615/iree_runtime-20230816.615-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Python 3.10
pip https://github.com/iree-org/iree/releases/download/candidate-20230816.615/iree_runtime-20230816.615-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Python 3.11
pip install https://github.com/iree-org/iree/releases/download/candidate-20230816.615/iree_runtime-20230816.615-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

ONNX-MLIR on GPU? No official support yet.





