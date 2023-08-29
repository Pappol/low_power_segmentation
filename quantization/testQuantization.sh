#!/bin/bash          
python convert_onnx.py 
python cv/low_power_segmentation/onnx2tflite/converter.py
python quantization.py 
