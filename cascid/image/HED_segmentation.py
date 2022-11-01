import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from cascid.configs import hed_cnf
from skimage.morphology import skeletonize


HED_DIR = hed_cnf.HED_DIR
OUTPUT_DIR = hed_cnf.HED_RESULTS

PROTOTXT = str(HED_DIR / "deploy.prototxt")
CAFFEMODEL = str(HED_DIR / "hed_pretrained_bsds.caffemodel")
WIDTH = 300
HEIGHT = 300


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

def apply_HED(image, preprocessing=None):
    inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(WIDTH, HEIGHT),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    out = out[0, 0]
    out = cv.resize(out, (image.shape[1], image.shape[0]))

    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)

    if preprocessing is not None:
        out = preprocessing(out)

    return out



def HED_segmentation_borders(original):

    kernel_erosion = np.ones((3,3),np.uint8)
    kernel_dilation = np.ones((5,5),np.uint8)

    hed = apply_HED(original)

    mask_erosion = cv.erode(hed,kernel_erosion,iterations = 4)

    skeleton = skeletonize(mask_erosion)

    gray_skeleton = cv.cvtColor(skeleton, cv.COLOR_BGR2GRAY)

    dilation = cv.dilate(gray_skeleton.astype('uint8'),kernel_dilation,iterations = 1)
    borders = original.copy()
    borders[dilation != 0] = [100,0,255]


    return borders


net = cv.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
cv.dnn_registerLayer('Crop', CropLayer)