#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:51:20 2022

@author: marco
"""

import os
import cv2
import sys
import numpy as np
 
path = "/home/marco/CenterPose/images/CenterPose/shoe/"
print(path)

for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.jpg':
        # print(filename)
        img = cv2.imread(path + filename)
        img = cv2.resize(img,(720, 960))
        print(filename.replace(".jpg",".png"))
        newfilename = filename.replace(".jpg",".png")
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)
        cv2.imwrite(path + newfilename,img)