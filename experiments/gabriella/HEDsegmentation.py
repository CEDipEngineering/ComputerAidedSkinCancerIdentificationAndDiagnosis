import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cascid import image_preprocessing
import os


HED_DIR = "../../../HED/"
PROTOTXT = HED_DIR + "deploy.prototxt"
CAFFEMODEL = HED_DIR + "hed_pretrained_bsds.caffemodel"
INPUT = HED_DIR + "image.png"
WIDTH = 300
HEIGHT = 300
OUTPUT_DIR = HED_DIR + "HED_RESULTS/"

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
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

    print(out.shape)
    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)

    if preprocessing is not None:
        out = preprocessing(out)

    print(type(out))
    print(np.max(out))
    print(np.min(out))
    print(out.shape)
    print(image.shape)
   # con=np.concatenate((image,out),axis=1)
    cv.imwrite(out_name,out)



if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the model.
net = cv.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
cv.dnn_registerLayer('Crop', CropLayer)

image=cv.imread(INPUT)
image=cv.resize(image,(WIDTH,HEIGHT))

#apply_HED(image, OUTPUT_DIR+"out.jpg")

gray_preprocessing = [image_preprocessing.simple_processing_clahe,
image_preprocessing.preprocessing_article,
image_preprocessing.preprocessing_article_histeq,
image_preprocessing.preprocessing_lab_histeq_grey,
image_preprocessing.unsharp_masking,
image_preprocessing.red_band_unsharp]

gray_preprocessing_names = ["gray_clahe.jpg",
"red_band_normalization.jpg",
"red_band_normalization_clahe.jpg",
"lab_clahe.jpg",
"unsharp_masking.jpg",
"red_band_normalization_unsharp.jpg"]

color_preprocessing = [
image_preprocessing.enhance_contrast_ab,
image_preprocessing.preprocessing_lab_histeq,
image_preprocessing.color_quantization]

color_preprocessing_names = [
"enhance_contrast_abs.jpg",
"color_lab_clahe.jpg",
"color_quantization.jpg"]


# for preprocessing, preprocessing_name in zip(gray_preprocessing, gray_preprocessing_names):
#     apply_HED(image, OUTPUT_DIR+preprocessing_name, preprocessing)


# for preprocessing, preprocessing_name in zip(color_preprocessing, color_preprocessing_names):
#     image = preprocessing(image)
#     apply_HED(image, OUTPUT_DIR+preprocessing_name)


# for preprocessing, preprocessing_name in zip(color_preprocessing, color_preprocessing_names):
image = image_preprocessing.color_quantization(image)
image1 = image_preprocessing.preprocessing_lab_histeq(image)
image2 = image_preprocessing.enhance_contrast_ab(image)

apply_HED(image1, OUTPUT_DIR+"color_quantization_enhance_contrast_abs.jpg")
apply_HED(image2, OUTPUT_DIR+"color_quantization_lab_clahe.jpg")