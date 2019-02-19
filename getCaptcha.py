# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:51:54 2019

@author: Tang
"""
import requests
import random
import cv2
import time
import numpy as np
i = 0


dirpath = "data/"



while i<600:
    i = i+1
    html = requests.get("http://course.uch.edu.tw/stdsel/System_D/dvcode.asp?")
    filename = dirpath+"captcha/"+str(random.randrange(0,999999))+".jpg"
    print(filename)
        
    with open(filename, 'wb') as file:
        file.write(html.content)
        
    
    img = cv2.imread(filename)
    ret, thresh = cv2.threshold(img, 127, 255,  #　門檻值轉黑白影像
                                cv2.THRESH_BINARY_INV)   
    
    denoise=cv2.fastNlMeansDenoising(thresh, h=60)    # 去除 Noise
    
    ret, thresh = cv2.threshold(denoise, 127, 255,  #　門檻值轉黑白影像
                               cv2.THRESH_BINARY_INV) 
    
    #kernel = np.ones((2 , 3) , np.uint8)
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    img = thresh
        #cv2.line(img, (33, 0), (33, 24), (0, 0, 255), 1)
        #cv2.line(img, (62, 0), (62, 24), (0, 0, 255), 1)
        #cv2.line(img, (94, 0), (94, 24), (0, 0, 255), 1)
    
    filename = dirpath+"crop/"+str(random.randrange(0,999999))+".jpg"
    crop_img = img[0:24, 0:33]
    crop_img = cv2.resize(crop_img,(227,227))
    cv2.imwrite(filename,crop_img)
    
    
    crop_img = img[0:24, 33:62]
    filename = dirpath+"crop/"+str(random.randrange(0,999999))+".jpg"
    crop_img = cv2.resize(crop_img,(227,227))
    cv2.imwrite(filename,crop_img)
    
    
    crop_img = img[0:24, 62:94]
    filename = dirpath+"crop/"+str(random.randrange(0,999999))+".jpg"
    crop_img = cv2.resize(crop_img,(227,227))
    cv2.imwrite(filename,crop_img)
    
    
    crop_img = img[0:24, 94:124]
    filename = dirpath+"crop/"+str(random.randrange(0,999999))+".jpg"
    crop_img = cv2.resize(crop_img,(227,227))
    cv2.imwrite(filename,crop_img)
    #cv2.imshow("123",crop_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


