# ufront
Unified Computing Frontend for Deep Learning 

## How it works?
Convert Pytorch, Tensorflow, Keras, ONNX models to UFront IR and then lower them into standard MLIR dialect (TOSA IR)

## For Conference Reproducibility
#### Install pre-build ufront package
In Ubuntu 20.04 or 22.04, download corresponding ufront package in the release folder and install.

```sh
pip install ufront-0.1.1-cp38-cp38-manylinux_2_28_x86_64.whl #for Python3.8

pip install ufront-0.1.1-cp39-cp39-manylinux_2_28_x86_64.whl #for Python3.9

pip install ufront-0.1.1-cp310-cp310-manylinux_2_28_x86_64.whl #for Python3.10
```

#### Install Execution Backend (IREE)
```sh
pip install iree-compiler==20230326.470 iree-runtime==20230326.470 -f https://openxla.github.io/iree/pip-release-links.html
```

#### Install Pytorch-cpu, Tensorflow-cpu (optional, for compiling pytorch and tensorflow models)
For pytorch, any version above 1.13.0 and torchvision above 0.14.0

```sh
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

```sh
pip install tensorflow-cpu
```
#### Execute end-to-end demo
```sh
python examples/torch_e2e_demo.py
```
The following results should be reproduced:

Pytorch:  [('n02099712', 'Labrador_retriever', 0.7244003), ('n02091831', 'Saluki', 0.13146894), ('n02099601', 'golden_retriever', 0.04318187), ('n02087394', 'Rhodesian_ridgeback', 0.020887861), ('n02092339', 'Weimaraner', 0.013975109)]

Compiling TOSA model...

UFront:  [('n02099712', 'Labrador_retriever', 0.72440577), ('n02091831', 'Saluki', 0.13146713), ('n02099601', 'golden_retriever', 0.043180563), ('n02087394', 'Rhodesian_ridgeback', 0.020887945), ('n02092339', 'Weimaraner', 0.01397502)]

Model:  MobileNetV3 , MAE:  9.8586455e-09

#### Commet and uncommet code for other models
in examples/torch_e2e_demo.py, change to other models, e.g., vision_transformer,
note: weight download will take some time
``` python
    # net = resnet18(pretrained=True)
    # net = resnet50(pretrained=True)
    # net = densenet121(pretrained=True)
    # net = inception_v3(pretrained=True) 
    # net = squeezenet1_1(pretrained=True)
    # net = shufflenet_v2_x1_5(pretrained=True)
    net = mobilenet_v3_small(pretrained=True, dropout=0.0)
    # net = models.vision_transformer.vit_b_16(weights=True) 
```

#### Change to GPU execution
Note: you need Nvidia GPU, driver and CUDA installed
``` python
GPU = False #change this to True
```
