from http.client import SWITCHING_PROTOCOLS
from re import T
import tensorflow as tf
import os
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
import torch.onnx
import argparse
from os import listdir
from os.path import isfile, join



def quantization(model_path, model):
    #load model 
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    #setup for data preparation
    def representative_dataset_gen():
        #list of images in the folder
        images = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
        for img_1 in images:
            for img_2 in images:
                if img_1 != img_2:
                    img_a = Image.open(images_folder+img_1).convert('RGB')
                    img_a = transformer_Arcface(img_a)
                    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

                    pic_b = opt.pic_b_path

                    img_b = Image.open(images_folder+img_2).convert('RGB')
                    img_b = transformer(img_b)
                    img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

                    # convert numpy to tensor
                    img_id = img_id.cuda()
                    img_att = img_att.cuda()

                    #create latent id
                    img_id_downsample = F.interpolate(img_id, size=(112,112))
                    latend_id = model.netArc(img_id_downsample)
                    latend_id = latend_id.detach().to('cpu')
                    latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
                    latend_id = latend_id.to('cuda')

                    yield (img_id, img_att, latend_id, latend_id, True)
        
    #converter setup
    converter.representative_dataset = representative_dataset_gen

    #convert the model
    tflite_quant_model = converter.convert()
    
    #save the model
    with open(model_path + "model.tflite", 'wb') as f:
        f.write(tflite_quant_model)

def standard_quantization(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path+"tf_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open(model_path + "model.tflite" , 'wb') as f:
        f.write(tflite_quant_model)

def main(argument):
    #arguments
    model_path = "output/"
    standard_quantization(model_path)


if __name__ == "__main__":
    argument = "c"
    main(argument)