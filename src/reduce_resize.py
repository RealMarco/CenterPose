#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:20:33 2022

@author: marco
"""
import os
import cv2
import sys
import numpy as np
 
path = "/home/marco/CenterPose/images/CenterPose/shoe1/"
print(path)

for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.jpg':
        # print(filename)
        img = cv2.imread(path + filename)
        img = cv2.resize(img,(960, 720))

        cv2.imwrite(path + filename,img)