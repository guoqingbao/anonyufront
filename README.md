# ufront
Unified Computing Frontend for Deep Learning 

## How it works?
Convert Pytorch, Tensorflow, Keras, ONNX models to UFront IR and then lower them into standard MLIR dialect (e.g., TOSA IR)

## For Conference Reproducibility
#### Install pre-build ufront package
In Ubuntu 20.04 or 22.04, download corresponding ufront package in the release folder and install.

```sh
pip install ufront-0.1.1-cp37-cp37m-manylinux_2_28_x86_64.whl #for Python3.7

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

ONNX Runtime:  [[('n02099712', 'Labrador_retriever', 0.61181027), ('n02091831', 'Saluki', 0.22184308), ('n02087394', 'Rhodesian_ridgeback', 0.03912668), ('n02092339', 'Weimaraner', 0.026891246), ('n02109047', 'Great_Dane', 0.019160295)], [('n02099712', 'Labrador_retriever', 0.61181027), ('n02091831', 'Saluki', 0.22184308), ('n02087394', 'Rhodesian_ridgeback', 0.03912668), ('n02092339', 'Weimaraner', 0.026891246), ('n02109047', 'Great_Dane', 0.019160295)]]

Pytorch:  [[('n02099712', 'Labrador_retriever', 0.61181074), ('n02091831', 'Saluki', 0.22184515), ('n02087394', 'Rhodesian_ridgeback', 0.039125554), ('n02092339', 'Weimaraner', 0.02689132), ('n02109047', 'Great_Dane', 0.0191598)], [('n02099712', 'Labrador_retriever', 0.61181074), ('n02091831', 'Saluki', 0.22184515), ('n02087394', 'Rhodesian_ridgeback', 0.039125554), ('n02092339', 'Weimaraner', 0.02689132), ('n02109047', 'Great_Dane', 0.0191598)]]

Compiling TOSA model...
UFront:  [[('n02099712', 'Labrador_retriever', 0.61180955), ('n02091831', 'Saluki', 0.22184528), ('n02087394', 'Rhodesian_ridgeback', 0.03912663), ('n02092339', 'Weimaraner', 0.02689206), ('n02109047', 'Great_Dane', 0.019160429)], [('n02099712', 'Labrador_retriever', 0.61180955), ('n02091831', 'Saluki', 0.22184528), ('n02087394', 'Rhodesian_ridgeback', 0.03912663), ('n02092339', 'Weimaraner', 0.02689206), ('n02109047', 'Great_Dane', 0.019160429)]]

Model:  MobileNetV3 , MAE with Pytorch:  5.4389266e-09

Model:  MobileNetV3 , MAE with ONNXRuntime:  4.557022e-09

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

#### Switch execution between CPU and GPU
Note: you need Nvidia GPU, driver and CUDA installed for GPU execution
``` python
GPU = False #change this to True for GPU execution
```
