# ufront
Unified Computing Frontend for Deep Learning 

## How it works?
Convert Pytorch, Tensorflow, Keras, ONNX models to UFront IR and then lower them into standard MLIR dialect (e.g., TOSA IR)

## For Conference Reproducibility
#### Install pre-build ufront package
In Ubuntu 20.04 or 22.04, download corresponding ufront package in the release folder and install.

Install any of the following packages according to your default Python version.
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

Pytorch:  [[('n02099712', 'Labrador_retriever', 0.60195434), ('n02091831', 'Saluki', 0.22342004), ('n02099601', 'golden_retriever', 0.11305459)], [('n02099712', 'Labrador_retriever', 0.60195434), ('n02091831', 'Saluki', 0.22342004), ('n02099601', 'golden_retriever', 0.11305459)]]

Compiling TOSA model...

UFront:  [[('n02099712', 'Labrador_retriever', 0.6019566), ('n02091831', 'Saluki', 0.22341833), ('n02099601', 'golden_retriever', 0.113055184)], [('n02099712', 'Labrador_retriever', 0.6019566), ('n02091831', 'Saluki', 0.22341833), ('n02099601', 'golden_retriever', 0.113055184)]]

Model:  MobileNetV3 , MAE with Pytorch:  4.8838333e-09

#### Commet and uncommet code for other models
in examples/torch_e2e_demo.py, change to other models, e.g., vision_transformer,
note: weight download will take some time
``` python
    # net = resnet18(weights="DEFAULT")
    # net = resnet50(weights="DEFAULT")
    # net = densenet121(weights="DEFAULT")
    # net = inception_v3(weights="DEFAULT", dropout=0.0) 
    # net = squeezenet1_1(weights="DEFAULT")
    # net = shufflenet_v2_x1_5(weights="DEFAULT")
    net = mobilenet_v3_small(weights="DEFAULT", dropout=0.0)
    # net = models.vision_transformer.vit_b_16(weights="DEFAULT") 
```

#### Switch execution between CPU and GPU
Note: you need Nvidia GPU, driver and CUDA installed for GPU execution
``` python
GPU = False #change this to True for GPU execution
```

### ImageNet inference with UFront on Kaggle
Run the ImageNet-1k inference online with UFront, be sure to login to use free GPU resources.

https://www.kaggle.com/code/bobking/imagenet-inference

### ImageNet inference with UFront locally

1) Download ImageNet validation set (about 2GB) from kaggle.com and extract it to a local folder
   https://www.kaggle.com/datasets/sautkin/imagenet1kvalid

2) In the example/imagenet_inference.ipynb (jupyter notebook), change the root path to the parent folder of "imagenet1kvalid", execute notebook cells (assume you have install ufront package and IREE backend).





