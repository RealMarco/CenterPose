#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:38:05 2022

@author: marco
"""

import numpy as np

# axis        x, y, z
# rotation    roll, pitch, yaw
# orientation γ, β, α

# right-hand coordinate system 1
def img_to_object_r_coor(w,h,x,y):
    '''
    convert image coordinate system to object right-hand coordinate system
    return np.array
    '''
    x1= h - y
    y1= x    
    return np.array([x1, y1])
# calculate yaw
def yaw_cal(HT0, HT):
    HT_cross = np.cross(HT0,HT)
    HT_dot   = np.dot(HT0,HT)
    yaw = np.arctan2(HT_cross, HT_dot)*180/np.pi # arctan2 is identical to the atan2 function of the underlying C library.
    return yaw
