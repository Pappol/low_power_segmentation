import tensorflow as tf
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import torch.onnx


def standard_quantization(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path+"tf_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open(model_path + "model.tflite" , 'wb') as f:
        f.write(tflite_quant_model)

def main(argument):
    #arguments
    model_path = "cv/low_power_segmentation/quantization/"
    standard_quantization(model_path)


if __name__ == "__main__":
    argument = "c"
    main(argument)