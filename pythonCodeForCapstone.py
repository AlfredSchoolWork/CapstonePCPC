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
    #print(img.dtype)
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
    showImage(rgb, 'Original Without Shrinking')
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
    #To extract Data, we need to specify it hue[row,col]  
    
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
    #Selecting the colour that I want to look at... 
    #Thresholds of acceptable values
    
    
    
    img[img<90] = 0
    img[img>360] = 0
    
    #The THRESHOLDING ALGORITHM IS NOT CORRECT
    
    # for y,row in enumerate(img):
    #     for x, pixel in enumerate(row):
    #         if pixel<90 or pixel>350:
    #             pixel = 100000000000
    # showImage(img,'After Thresholding')
    return img



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

def pixelCounting(img,window_size,start_window_x,start_window_y):
    counts = 0
    for col in range(window_size):
        for row in range(window_size):
            if (img[row+start_window_x-1,col+start_window_y-1])>0:
                counts = counts +1
    return counts

def pixelCounter(img):
    print("Current Image Dimension: ",img.shape)
    (row,col) = img.shape
    
    window_size = 100
    top_pos = 10
    bottom_pos = col - window_size - top_pos
    left_pos = 10
    right_pos = row - window_size - left_pos
    #Middle Not doing well, ignore for now
    middle_frm_top = col//4 
    middle_frm_left = row//4
    
    #Ignore Middle results for now
    middle_count = pixelCounting(img, window_size, 250, 250)
    print("Middle: ",middle_count)
    top_left_count = pixelCounting(img, window_size, top_pos, left_pos)
    print("Top Left: ",top_left_count)
    top_right_count = pixelCounting(img, window_size, top_pos, right_pos)
    print("Top Right: ",top_right_count)
    bottom_left_count = pixelCounting(img, window_size, bottom_pos, left_pos)
    print("Bottom left: ", bottom_left_count)
    bottom_right_count = pixelCounting(img, window_size, bottom_pos, right_pos)
    print("Bottom Right: ",bottom_right_count)
    
    
    return None



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
    imageBIdRearDir = 'Black_3_Rear_Tripod.cr2'
    
    #For Analysis, use 250
    shrink_thresh = 250
    
    #imageBIdRear = readRawFile(directory+imageBIdRearDir, 530,3130,1700,4200,shrink_thresh)
    #imageBIdRearHue = hsvConverter(imageBIdRear)
    #imageBIdRearHueWA,BIdRearHueWAmax,BIdRearHueWAmin,BIdRearHueWAavg = weightedAverageFiltering(imageBIdRearHue)
    
    
    # imageBEL = readRawFile(directory+imageBELDir, 830,2330,2700,4200,shrink_thresh)
    # imageBELHue = hsvConverter(imageBEL)
    # imageBELHueWA,BELHueWAmax,BELHueWAmin,BELHueWAavg = weightedAverageFiltering(imageBELHue)
    
    # imageBL = readRawFile(directory+imageBLDir, 800,2300,2700,4200,shrink_thresh)
    # imageBLHue = hsvConverter(imageBL)
    # imageBLHueWA,BLHueWAmax,BLHueWAmin,BLHueWAavg = weightedAverageFiltering(imageBLHue)
    
    imageBId = readRawFile(directory+imageBIdDir, 800,2300,2700,4200,shrink_thresh)
    imageBIdHue = hsvConverter(imageBId)
    imageBIdHueWA,BIdHueWAmax,BIdHueWAmin,BIdHueWAavg = weightedAverageFiltering(imageBIdHue)
    
    imageBH = readRawFile(directory+imageBHDir, 800,2300,2700,4200,shrink_thresh)
    imageBHHue = hsvConverter(imageBH)
    imageBHHueWA,BHHueWAmax,BHHueWAmin,BHHueWAavg = weightedAverageFiltering(imageBHHue)
    
    imageBEH = readRawFile(directory+imageBEHDir, 800,2300,2650,4150,shrink_thresh)
    imageBEHHue = hsvConverter(imageBEH)
    imageBEHHueWA,BEHHueWAmax,BEHHueWAmin,BEHHueWAavg = weightedAverageFiltering(imageBEHHue)
    
    # imageBIC = readRawFile(directory+imageBICDir, 750,2250,2700,4200,shrink_thresh)
    # imageBICHue = hsvConverter(imageBIC)
    # imageBICHueWA,BICHueWAmax,BICHueWAmin,BICHueWAavg = weightedAverageFiltering(imageBICHue)
    
    
    # showImage(imageBEHHueWA,'After Averaging')
    #showImage(imageBIdRear,'Original')
    #showImage(imageBIdRearHue,'Hue')
    #showImage(imageBIdRearHueWA,'Weighted Average')
    #plt.hist(imageBIdHue)
    
    
    #Canny if we want to remove background, but not really needed
    #grey = convertGreyScale(imageBIdRear)
    #showImage(grey, 'grey')
    #edges =cannyDetector(grey, 85, 255)
    
    
    # imageBHueMax = [BELHueWAmax,BLHueWAmax,BIdHueWAmax,BHHueWAmax,BEHHueWAmax]
    # imageBHueMin = [BELHueWAmin,BLHueWAmin,BIdHueWAmin,BHHueWAmin,BEHHueWAmin]
    # imageBHueAvg = [BELHueWAavg,BLHueWAavg,BIdHueWAavg,BHHueWAavg,BEHHueWAavg]
    #Deduce that a good range will be from 90 to 357
    showImage(imageBId, 'Original Image')
    result2 = thresholding(imageBIdHue)
    showImage(result2,'Without Weights')
    #print(imageBIdHue[1])
    pixelCounter(imageBIdHue)
    #pixelCounter(result2)
    pixelCounter(imageBHHue)
    pixelCounter(imageBEHHue)
    
    
    result = thresholding(imageBIdHueWA)
    showImage(result, 'After Modification')
    #pixelCounting(imageBIdHueWA, 10, 10)
    
    
    if cv2.waitKey(0):
        cv2.destroyAllWindows()


# imageBIC = readRawFile(directory+imageBICDir, 750,2250,2700,4200)
# imageBICHue = hsvConverter(imageBIC)


#readRawFile(directory+imageBIC,750,2250,2700,4200)

















