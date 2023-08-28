import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import torch.onnx 
import onnx
from transformers import MobileNetV2ForSemanticSegmentation


def convert_model():
    dummy_input = torch.randn(1, 3, 512, 512, device="cuda")
    model = MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513", 
                                                           num_labels=14, 
                                                           ignore_mismatched_sizes=True, 
                                                           image_size=(512, 512)).cuda()

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "cv/low_power_segmentation/quantization/model.onnx", verbose=True, input_names=input_names, output_names=output_names)



convert_model()
