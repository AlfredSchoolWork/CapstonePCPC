# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:40:23 2022

@author: Blackpolar
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
import torchvision.models
from torch.utils.data import Dataset , DataLoader

import dataPrep as dp
import readFile as rf
import Machine_learning_classifier as ml

def setup_new_data(filename,savedirectory='data/demo_data.npz'):
    measurement = [0,0,0,0,0,0,0,0,0]
    data = dp.imageDataSetup(filename, measurement)
    np.savez(savedirectory,data)
    return data


if __name__ == '__main__':
    #Example 1
    directory = 'images/'
    image = 'Black_5_Handheld_Stable.cr2'
    filename = directory + image
    setup_new_data(filename,savedirectory='data/demo_data.npz')
    ml.decide(demo_data_file='data/demo_data.npz',epoch = 79,TOL=0.05)
    
    
    #Example 2
    directory = 'images/'
    image = 'White_6_Handheld.cr2'
    filename = directory + image
    setup_new_data(filename,savedirectory='data/demo_data.npz')
    ml.decide(demo_data_file='data/demo_data.npz',epoch = 79,TOL=0.05)