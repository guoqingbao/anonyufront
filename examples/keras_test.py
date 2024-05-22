from tensorflow.keras import backend
from ufront.keras.model import UFrontKeras
from keras_def import SequentialCNN, ConcatenatedCNN, NestedCNN, ShuffleNet, SqueezeNet_11, ResNet18
from tensorflow.keras.applications import ResNet50, ResNet50V2, EfficientNetB0, Xception, MobileNetV2, MobileNetV3Small, DenseNet121, InceptionV3, VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from vit_keras import vit
import numpy as np
import iree.compiler as ireec
from iree import runtime
import tensorflow as tf
import torch
from torch_def import mae, mse, r_square, rmse, mpe, load_sample_image

def decode_result(result):
  return tf.keras.applications.resnet50.decode_predictions(result, top=5)[0]
    

if __name__ == "__main__":
    GPU = True
    input_last, input = load_sample_image()
    backend.set_image_data_format('channels_first')
    tf.keras.backend.set_floatx('float32')
    # backend.set_image_data_format('channels_last')
    # inputs, outputs, model_name = SequentialCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = ConcatenatedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # inputs, outputs, model_name = NestedCNN(shape=(3, 32, 32), dtype="float32", num_classes=10)
    # input = np.ones((1, 3, 224, 224), dtype=np.float32)
    # keras_input = layers.Input(shape=input.shape[1:])

    # base_model = ResNet18(classes=1000, input_shape=(3, 224, 224))
    # base_model = vit.vit_b16(image_size=224, activation='relu', pretrained=False, include_top=True, pretrained_top=False, channel_first=True)
    # base_model = ResNet50(weights=None, include_top=True) # no batch norm

    # base_model = ResNet50V2(weights=None, include_top=True) # with batch norm

    # base_model = EfficientNetB0(weights=None, include_top=True, input_shape=input.shape[1:]) 

    # base_model = Xception(weights=None, include_top=True, input_shape=input.shape[1:])
    # base_model = MobileNetV2(weights=None, include_top=True, input_shape=input.shape[1:])
    # base_model = MobileNetV3Small(weights=None, include_top=True, input_shape=input.shape[1:])
    base_model = DenseNet121(weights=None, include_top=True, input_shape=input.shape[1:])
    # base_model = InceptionV3(weights=None, include_top=True, input_shape=input.shape[1:])
    # base_model = VGG16(weights=None, include_top=True, input_shape=input.shape[1:])

    # base_model = SqueezeNet_11(input_shape=input.shape[1:], nb_classes=1000, channel_first=True)
    # base_model = ShuffleNet(include_top=True, pooling='avg', input_shape=input.shape[1:])

    weights_channel_first = base_model.get_weights()

    model_name = base_model.name

    transformer = True if model_name.find("Transformer") > 0 or model_name.find("vit") >= 0 else False

    model = UFrontKeras(base_model, inputs = [input], batch_size = 1, transformer=transformer, pass_weights=True)

    if transformer:
      last_op = model.get_output_operator()
      output = model.umodel().softmax(input=last_op.get_output(0), name="softmax_out")

    model.compile(optimizer={"type":"sgd", "lr":"0.01", "momentum":"0", "nesterov":"False", "weight_decay":"0"},
                        loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    print("\r\n\r\nIR for ", model_name)

    # for operator in operators:
    #   print(operator.ir) #show ir for each operator
    modelir= model.dump_ir()
    # print(modelir)
    # with open("vit_keras.mlir", "w") as f:
    #     f.write(modelir)
    tosa_ir= model.dump_tosa_ir()

    # import pathlib
    # # path = str(pathlib.Path(__file__).parent.resolve()) + "/output_ir/keras_" + model_name + ".ir"
    # f = open("resnet18_ufront.mlir", "w")
    # f.write(tosa_ir)
    # f.close()
    


    print("Compiling TOSA model...")
    if GPU:
        binary = ireec.compile_str(tosa_ir,
                        target_backends=["cuda"], 
                        input_type=ireec.InputType.TOSA)
        module = runtime.load_vm_flatbuffer(binary, driver="cuda")
    else:
        binary = ireec.compile_str(tosa_ir,
                        target_backends=["llvm-cpu"], 
                        input_type=ireec.InputType.TOSA)
        module = runtime.load_vm_flatbuffer(binary,backend="llvm-cpu") 

    ufront_ret = module.forward(input).to_host()
    print("\nUFront: ", decode_result(ufront_ret))
    
    # make sure weight consistent for comparison
    backend.set_image_data_format('channels_last')
    sp = tuple(list(input.shape[2:]) + [3])
    # base_model = ResNet50(weights=None, include_top=True)
    # base_model = ResNet50V2(weights=None, include_top=True) # with batch norm
    # base_model = MobileNetV3Small(weights=None, include_top=True, input_shape=sp)
    # base_model = MobileNetV2(weights=None, include_top=True, input_shape=sp)
    base_model = DenseNet121(weights=None, include_top=True, input_shape=sp)
    # base_model = ResNet18(classes=1000, input_shape=sp)
    # base_model = ShuffleNet(include_top=True, input_shape=sp)
    # base_model = SqueezeNet_11(input_shape=sp, nb_classes=1000, channel_first=False)
    # base_model = InceptionV3(weights=None, include_top=True, input_shape=sp)
    # base_model = vit.vit_b16(image_size=224, activation='relu', pretrained=False, include_top=True, pretrained_top=False, channel_first=False)
    base_model.set_weights(weights_channel_first)
    ret = base_model(input_last).numpy()
    print("\nTensorflow-cpu: ", decode_result(ret))

    dif = ufront_ret - ret
    mae = np.mean(abs(dif))
    print("Model: ", model_name, ", MAE with Tensorflow: ", mae)
    print("RMSE:", rmse(torch.Tensor(ret), torch.Tensor(ufront_ret)).numpy())
    print("COD:", r_square(torch.Tensor(ret), torch.Tensor(ufront_ret)).numpy())
    print("MPE:", mpe(torch.Tensor(ret), torch.Tensor(ufront_ret)).numpy(), "%")