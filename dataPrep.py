# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:49:52 2022

@author: Blackpolar
"""

import readFile as rf
import numpy as np
import cv2
import random

#modify files based on whichever function called
#Allows the use of different detection methods, LBP and cropping 
#Set up for NPZ file 

def mousePoints(event,x,y,flags,params):
    global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
	# check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        cropping = False
		
def cropImage(img,measurement,size=700):
    

    cropped = img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    image_size = cropped.shape
    row = image_size[0]
    col = image_size[1]
    #We didnt use a list because different size arrays cannot be stacked togfether
    cropped_A = cropped[0:(row//4*2),0:(col//4*2)]
    cropped_B = cropped[0:(row//4*2),(col//4):(col//4*3)]
    cropped_C = cropped[0:(row//4*2),(col//4*2):(col//4*4)]
    cropped_D = cropped[(row//4):(row//4*3),0:(col//4*2)]
    cropped_E = cropped[(row//4):(row//4*3),(col//4):(col//4*3)]
    cropped_F = cropped[(row//4):(row//4*3),(col//4*2):(col//4*4)]
    cropped_G = cropped[(row//4*2):(row//4*4),0:(col//4*2)]
    cropped_H = cropped[(row//4*2):(row//4*4),(col//4):(col//4*3)]
    cropped_I = cropped[(row//4*2):(row//4*4),(col//4*2):(col//4*4)]
    #cv2.destroyWindow("Original")
    rf.showImage(cropped, "Cropped")
    # rf.showImage(cropped_A, "cropped_A")
    # rf.showImage(cropped_B, "cropped_B")
    # rf.showImage(cropped_C, "cropped_C")
    
    # rf.showImage(cropped_D, "cropped_D")
    # rf.showImage(cropped_E, "cropped_E")
    # rf.showImage(cropped_F, "cropped_F")
       
    # rf.showImage(cropped_G, "cropped_G")
    # rf.showImage(cropped_H, "cropped_H")
    # rf.showImage(cropped_I, "cropped_I")
    print(cropped_A[0:size,0:size].shape)
    if ((cropped_A[0:size,0:size].shape) != (size,size,3)):
        print("Please recrop or choose a smaller crop size")
    
    
    data = np.array([(measurement[0],cropped_A[0:size,0:size]),
                     (measurement[1],cropped_B[0:size,0:size]),
                     (measurement[2],cropped_C[0:size,0:size]),
                     (measurement[3],cropped_D[0:size,0:size]),
                     (measurement[4],cropped_E[0:size,0:size]),
                     (measurement[5],cropped_F[0:size,0:size]),
                     (measurement[6],cropped_G[0:size,0:size]),
                     (measurement[7],cropped_H[0:size,0:size]),
                     (measurement[8],cropped_I[0:size,0:size]),],
                     dtype = [('measurement',float),('data',cropped_A[0][0].dtype,cropped_A[0:size,0:size].shape)])
    return data

def imageDataSetup(filename,measurement):
    global refPt 
    refPt = []
    raw_image,rgb = rf.readRawFile(filename)
    rf.showImage(rgb, "Original",1000)
    #rgb =rf.convertLBP(rgb)
    cv2.setMouseCallback("Original", mousePoints)
    
    while True:
	# display the image and wait for a keypress
        key = cv2.waitKey(1) & 0xFF
	# if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            print(refPt)
            break
        
    if (len(refPt) == 2):
        #Change this value such that all the values suit the data point
        data_new = cropImage(rgb,measurement,650)
    else:
        print("Not enough reference points")
    
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
        
    return data_new

def crop_save(save = False):
    directory = 'images/'
    image1 = 'Black_1_Tripod.cr2' #Good crop at 830,2330,2700,4200
    image2  = 'Black_2_Tripod.cr2' #Good crop at 800,2300,2700,4200
    image3 = 'Black_3_Tripod.cr2'
    image4 = 'Black_4_Tripod.cr2'
    image5 = 'Black_5_Tripod.cr2' #Good crop at 800,2300,2650,4150
    image6 = 'Black_6_Tripod.cr2' #Good crop at 750,2250,2700,4200
    image7 = 'White_1_Tripod.cr2' 
    image8  = 'White_2_Tripod.cr2' 
    image9 = 'White_3_Tripod.cr2'
    image10 = 'White_4_Tripod.cr2'
    image11 = 'White_5_Tripod.cr2' 
    image12 = 'White_6_Tripod.cr2' 
    
    measure1 = [45,43,48,49,44,47,38,47,34]
    #print(len(measure1))
    measure2 = [54,65,58,56,61,60,56,59,53]
    #print(len(measure2))
    measure3 = [75,77,76,87,83,85,94,86,85]
    #print(len(measure3))
    measure4 = [100,95,80,100,92,94,88,82,96]
    #print(len(measure4))
    measure5 = [100,105,107,104,94,113,98,99,104]
    #print(len(measure5))
    measure6 = [153,127,151,147,140,128,125,113,120]
    #print(len(measure6))
    measure7 = [44,42,36,39,33,46,37,34,33]
    #print(len(measure7))
    measure8 = [58,57,40,59,51,48,49,42,51]
    #print(len(measure8))
    measure9 = [75,69,82,65,75,76,79,90,85]
    #print(len(measure9))
    measure10 = [175,143,160,185,120,127,151,140,157]
    #print(len(measure10))
    measure11 = [240,195,185,190,150,183,184,182,207]
    #print(len(measure11))
    measure12 = [180,156,170,224,149,155,168,132,170]    
    #print(len(measure12))
    
    filename = directory+image1
    data_new1 = imageDataSetup(filename,measure1)
    filename = directory+image2
    data_new2 = imageDataSetup(filename,measure2)
    filename = directory+image3
    data_new3 = imageDataSetup(filename,measure3)
    filename = directory+image4
    data_new4 = imageDataSetup(filename,measure4)
    filename = directory+image5
    data_new5 = imageDataSetup(filename,measure5)
    filename = directory+image6
    data_new6 = imageDataSetup(filename,measure6)
    filename = directory+image7
    data_new7 = imageDataSetup(filename,measure7)
    filename = directory+image8
    data_new8 = imageDataSetup(filename,measure8)
    filename = directory+image9
    data_new9 = imageDataSetup(filename,measure9)
    filename = directory+image10
    data_new10 = imageDataSetup(filename,measure10)
    filename = directory+image11
    data_new11 = imageDataSetup(filename,measure11)
    filename = directory+image12
    data_new12 = imageDataSetup(filename,measure12)
    
    data = data_new1
    
    data = np.append(data,data_new2)
    data = np.append(data,data_new3)
    data = np.append(data,data_new4)
    data = np.append(data,data_new5)
    data = np.append(data,data_new6)
    data = np.append(data,data_new7)
    data = np.append(data,data_new8)
    data = np.append(data,data_new9)
    data = np.append(data,data_new10)
    data = np.append(data,data_new11)
    data = np.append(data,data_new12)
    print(len(data))
    
    if (save):
        np.savez('data/traindata.npz',data)

def split_data():
    testData = np.load('data/traindata.npz')
    data = testData['arr_0']
    print(len(data))
    #print(data[5][0]) #This gives me the point
    validation_sample_size = 18
    random_samples = list(set([random.randrange(0,len(data)-1) for i in range(validation_sample_size)]))
    print(random_samples)
    valid_data = data[random_samples[0]]
    for r in random_samples[1:]:
        valid_data = np.append(valid_data,data[r])
    data = np.delete(testData['arr_0'],random_samples)
    print(len(data))
    print(len(valid_data))
    
    test_sample_size = 10
    random_samples = list(set([random.randrange(0,len(data)-1) for i in range(test_sample_size)]))
    print(random_samples)
    test_data = data[random_samples[0]]
    for r in random_samples[1:]:
        test_data = np.append(test_data,data[r])
    data = np.delete(data,random_samples)
    print(len(data))
    print(len(test_data))
    
    np.savez('data/train_data.npz',data)
    np.savez('data/test_data.npz',test_data)
    np.savez('data/valid_data.npz',valid_data)

if __name__ == '__main__':
    
    #crop_save()
    #split_data()
