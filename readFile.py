# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:37:23 2022

@author: Blackpolar
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import rawpy
from skimage.feature import local_binary_pattern
from skimage.util import img_as_float
from skimage import exposure

def showImage(img,imageLabel,imageSize=500):
    #print(img.dtype)
    if (img.dtype != np.float_):
        img = img_as_float(img)
        #img = cv2.normalize(img,None,0,65535,cv2.NORM_MINMAX,cv2.CV_16U)
        
        # img = img.astype(np.uint16)
    #image_size = img.shape
    imageName = imageLabel
    #Show images at 500x500, feel free to change if necessary
    cv2.namedWindow(imageName,cv2.WINDOW_KEEPRATIO)
    cv2.imshow(imageName,img)
    cv2.resizeWindow(imageName,imageSize,imageSize)
    
    return None

def readRawFile(filename):
    raw = rawpy.imread(filename)
    rgb = convertRGB(raw)
    raw_image = convertRaw(raw)
    #rgb = raw.raw_image
    
    # print(rgb.shape)
    # print(raw.color_desc)
    
    #Show images at 500x500, feel free to change if necessary
    # cv2.namedWindow('Original Image',cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('Original Image',rgb)
    # cv2.resizeWindow('Original Image',500,500)
    # cv2.waitKey(0)
    print("Image successfully loaded")
    return raw_image,rgb

def convertRGB(raw):
    rgb = raw.postprocess(no_auto_bright = False ,use_camera_wb = True, output_bps=16)
    rgb = img_as_float(rgb)
    rgb = exposure.rescale_intensity(rgb)
    return rgb
    
def convertRaw(raw):
    rawImage = raw.raw_image;
    rawImage = img_as_float(rawImage)
    rawImage = exposure.rescale_intensity(rawImage)
    return rawImage
    
def convertLBP(img,npoints=32,radius=2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, npoints, radius)
    return lbp

