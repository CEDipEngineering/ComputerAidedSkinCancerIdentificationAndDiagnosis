import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cascid import image_preprocessing
import os


HED_DIR = "../../../../HED/"
PROTOTXT = HED_DIR + "deploy.prototxt"
CAFFEMODEL = HED_DIR + "hed_pretrained_bsds.caffemodel"
INPUT = HED_DIR + "test_imgs/"
WIDTH = 300
HEIGHT = 300
OUTPUT_DIR = HED_DIR + "HED_RESULTS/"

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0


    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def apply_HED(image, out_name, preprocessing=None):
    inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(WIDTH, HEIGHT),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
    net.setInput(inp)
    # edges = cv.Canny(image,image.shape[1],image.shape[0])
    out = net.forward()

    out = out[0, 0]
    out = cv.resize(out, (image.shape[1], image.shape[0]))

    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)

    if preprocessing is not None:
        out = preprocessing(out)
    cv.imwrite(out_name,out)


def apply_HED_with_all_preprocessings(image, img_name):
    
    for preprocessing, preprocessing_name in zip(gray_preprocessing, gray_preprocessing_names):
        apply_HED(image, OUTPUT_DIR+preprocessing_name+img_name, preprocessing)


    for preprocessing, preprocessing_name in zip(color_preprocessing, color_preprocessing_names):
        image = preprocessing(image)
        apply_HED(image, OUTPUT_DIR+preprocessing_name+img_name)

    for preprocessing, preprocessing_name in zip(color_quant, color_quant_names):
        image = preprocessing(image)
        apply_HED(image, OUTPUT_DIR+preprocessing_name+img_name)




### ------------------ Preparing image folders ------------------ ###
gray_preprocessing = [image_preprocessing.simple_processing_clahe,
image_preprocessing.preprocessing_article,
image_preprocessing.preprocessing_article_histeq,
image_preprocessing.preprocessing_lab_histeq_grey,
image_preprocessing.unsharp_masking,
image_preprocessing.red_band_unsharp]

gray_preprocessing_names = ["gray_clahe/",
"red_band_normalization/",
"red_band_normalization_clahe/",
"lab_clahe/",
"unsharp_masking/",
"red_band_normalization_unsharp/"]

color_preprocessing = [
image_preprocessing.enhance_contrast_ab,
image_preprocessing.preprocessing_lab_histeq,
image_preprocessing.color_quantization]

color_preprocessing_names = [
"enhance_contrast_abs/",
"color_lab_clahe/",
"color_quantization/"]

color_quant_names = [
"quant_enhance_contrast_abs/",
"color_quant_lab_clahe/"]

color_quant = color_preprocessing[:2]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for name in (gray_preprocessing_names + color_preprocessing_names + color_quant_names):
    if not os.path.exists(OUTPUT_DIR+name):
        os.makedirs(OUTPUT_DIR+name)



### ------------------ Loading the model ------------------ ###
net = cv.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
cv.dnn_registerLayer('Crop', CropLayer)


### ------------------ applying HED to images ------------------ ###
for f in os.listdir(INPUT):
    if f.endswith(".png") or f.endswith(".jpg"):
        image=cv.imread(INPUT + f)
        image=cv.resize(image,(WIDTH,HEIGHT))
        apply_HED_with_all_preprocessings(image,f)