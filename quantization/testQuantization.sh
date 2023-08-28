#!/bin/bash          
python convert_onnx.py 
cd onnx-tensorflow/
python onnx_2_TFLite.py 
cd ..
python quantization.py --name final --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path crop_224/hzxc.jpg --pic_b_path crop_224/mouth_open.jpg --output_path output/ --which_epoc 800000

python tf_inference.py 