# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:33:47 2022

@author: Blackpolar
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import rawpy

def showImage(img,imageLabel):
    #Converts images to uint8 for cv2 imshow.
    print(img.dtype)
    if (img.dtype != np.uint16):
        img = cv2.normalize(img,None,0,65535,cv2.NORM_MINMAX,cv2.CV_16U)
        # img = img.astype(np.uint16)
    
    imageName = 'Original Image' + imageLabel
    #Show images at 500x500, feel free to change if necessary
    cv2.namedWindow(imageName,cv2.WINDOW_KEEPRATIO)
    cv2.imshow(imageName,img)
    cv2.resizeWindow(imageName,800,800)
    
    return None


def readRawFile(filename,y1,y2,x1,x2,strink_thresh):
    raw = rawpy.imread(filename)
    #rgb = raw.raw_image
    rgb = raw.postprocess(gamma=(1,1), output_bps=16)
    size = raw.sizes
    y1 = y1 + strink_thresh
    y2 = y2 - strink_thresh
    x1 = x1 + strink_thresh
    x2 = x2 - strink_thresh
    #rgb = rgb[800:2300,2700:4200]
    rgb = rgb[y1:y2,x1:x2]
    print(size)
    # print(rgb.shape)
    # print(raw.color_desc)
    
    #Show images at 500x500, feel free to change if necessary
    # cv2.namedWindow('Original Image',cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('Original Image',rgb)
    # cv2.resizeWindow('Original Image',500,500)
    # cv2.waitKey(0)
    
    return rgb

def hsvConverter(img):
    if (img.dtype != np.float32) :
        img = img.astype(np.float32)
    #showImage(img)
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0]
    #Not necessary, but we need to make sure that sat and val are not influencing 
    #the result.
    sat = hsv[:,:,1]
    val = hsv[:,:,2]
    #showImage(hue)
    return hue
    


def weightedAverageFiltering(img):
   
    #Creates a weighted average kernel 
    weightAvgKernel = np.array([[1/16,1/16,1/16],
                                [1/16,1/2,1/16],
                                [1/16,1/16,1/16]])
    if (np.sum(weightAvgKernel) != 1):
        print("Please check kernel as sum of kernel: ",np.sum(weightAvgKernel))
    imgWeightedAvg = cv2.filter2D(src=img, ddepth=-1, kernel=weightAvgKernel)
    #showImage(imgWeightedAvg)
    maxi = np.max(imgWeightedAvg)
    mini = np.min(imgWeightedAvg)
    avg  = np.mean(imgWeightedAvg)
    
    return imgWeightedAvg,maxi,mini,avg
    

def thresholding(img):
    #Original Number 90,360
    img[img<90] = 100000
    img[img>360] = 100000
    # for y,row in enumerate(img):
    #     for x, pixel in enumerate(row):
    #         if pixel<90 or pixel>350:
    #             pixel = 100000000000
    # showImage(img,'After Thresholding')
    return img


def consistencyCheck(img):
    for y,row in enumerate(img):
        for x, pixel in enumerate(row):
            if x == 0 or x== 1500:
                continue
            if y == 0 or y ==1500:
                continue
    return None

def convertGreyScale(img):
    if (img.dtype != np.float32):
        img = img.astype(np.float32)
    
    # All the methods below will not work and will result in an all black scenario
    # if (img.dtype != np.uint8) :
    #     img = img.astype(np.uint8)
    #     img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    #Note that cvtColor requires float32 as input 
    grey = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    return grey
    

def cannyDetector(img,thresh1,thresh2):
    if (img.dtype != np.uint8):
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        print(img.dtype)
    showImage(img, 'Before Canny, Only for Proving')
    #CV2 Canny can only work with uint8 images. 
    #However when you try to turn float64 to uint8, all the images disappeared.
    #This is due to numbers like 32 in float64 when normalise to uint8 will be 0.
    edges = cv2.Canny(img,thresh1,thresh2)
    showImage(edges, 'Canny Detector')
    return edges


if __name__ == '__main__':
    #Definition of all files names
    #White as W, Black as B
    #Extreme Light as EL
    #Light as L
    #Ideal as Id
    #Heavy as H
    #Extreme Heavy as EH
    #Inconsistent as IC
    directory = 'images/'
    imageBELDir = 'Black_1_Tripod.cr2' #Good crop at 830,2330,2700,4200
    imageBLDir  = 'Black_2_Tripod.cr2' #Good crop at 800,2300,2700,4200
    imageBIdDir = 'Black_3_Tripod.cr2'
    imageBHDir = 'Black_4_Tripod.cr2'
    imageBEHDir = 'Black_5_Tripod.cr2' #Good crop at 800,2300,2650,4150
    imageBICDir = 'Black_6_Tripod.cr2' #Good crop at 750,2250,2700,4200
    
    #For Analysis, use 250
    shrink_thresh = 0
    
    # imageBEL = readRawFile(directory+imageBELDir, 830,2330,2700,4200,shrink_thresh)
    # imageBELHue = hsvConverter(imageBEL)
    # imageBELHueWA,BELHueWAmax,BELHueWAmin,BELHueWAavg = weightedAverageFiltering(imageBELHue)
    
    # imageBL = readRawFile(directory+imageBLDir, 800,2300,2700,4200,shrink_thresh)
    # imageBLHue = hsvConverter(imageBL)
    # imageBLHueWA,BLHueWAmax,BLHueWAmin,BLHueWAavg = weightedAverageFiltering(imageBLHue)
    
    imageBId = readRawFile(directory+imageBIdDir, 800,2300,2700,4200,shrink_thresh)
    imageBIdHue = hsvConverter(imageBId)
    imageBIdHueWA,BIdHueWAmax,BIdHueWAmin,BIdHueWAavg = weightedAverageFiltering(imageBIdHue)
    
    # imageBH = readRawFile(directory+imageBHDir, 800,2300,2700,4200,shrink_thresh)
    # imageBHHue = hsvConverter(imageBH)
    # imageBHHueWA,BHHueWAmax,BHHueWAmin,BHHueWAavg = weightedAverageFiltering(imageBHHue)
    
    # imageBEH = readRawFile(directory+imageBEHDir, 800,2300,2650,4150,shrink_thresh)
    # imageBEHHue = hsvConverter(imageBEH)
    # imageBEHHueWA,BEHHueWAmax,BEHHueWAmin,BEHHueWAavg = weightedAverageFiltering(imageBEHHue)
    
    imageBIC = readRawFile(directory+imageBICDir, 750,2250,2700,4200,shrink_thresh)
    imageBICHue = hsvConverter(imageBIC)
    imageBICHueWA,BICHueWAmax,BICHueWAmin,BICHueWAavg = weightedAverageFiltering(imageBICHue)
    
    
    # showImage(imageBEHHueWA,'After Averaging')
    showImage(imageBIC,'Original')
    showImage(imageBICHue,'Hue')
    showImage(imageBICHueWA,'Weighted Average')
    #plt.hist(imageBIdHue)
    
    
    #Canny if we want to remove background, but not really needed
    grey = convertGreyScale(imageBIC)
    showImage(grey, 'grey')
    edges =cannyDetector(grey, 85, 255)
    
    
    # imageBHueMax = [BELHueWAmax,BLHueWAmax,BIdHueWAmax,BHHueWAmax,BEHHueWAmax]
    # imageBHueMin = [BELHueWAmin,BLHueWAmin,BIdHueWAmin,BHHueWAmin,BEHHueWAmin]
    # imageBHueAvg = [BELHueWAavg,BLHueWAavg,BIdHueWAavg,BHHueWAavg,BEHHueWAavg]
    #Deduce that a good range will be from 90 to 357
    result = thresholding(imageBICHueWA)
    showImage(result, 'After Modification')
    if cv2.waitKey(0):
        cv2.destroyAllWindows()


# imageBIC = readRawFile(directory+imageBICDir, 750,2250,2700,4200)
# imageBICHue = hsvConverter(imageBIC)


#readRawFile(directory+imageBIC,750,2250,2700,4200)


