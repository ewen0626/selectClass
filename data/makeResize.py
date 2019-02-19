# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 01:57:50 2019

@author: USER
"""

import cv2

mypath = "crop/Train/227/X/"
from os import listdir
from os.path import isfile, isdir, join
files = listdir(mypath)

print("分類開始")
# Load from a file
for f in files: 
    fullpath = join(mypath, f)
    img = cv2.imread(fullpath)
    img = cv2.resize(img,(227,227))
    cv2.imwrite(fullpath,img)
    