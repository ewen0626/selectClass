# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:25:31 2019

@author: USER
"""

import requests
import cv2
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image



graph_def = tf.GraphDef()
labels = []
filename = "data/model.pb"
labels_filename = "data/labels.txt"
# Import the TF graph
with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())





def convert_to_opencv(image):
        # RGB -> BGR conversion is performed as well.
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image
    
def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]
    
def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image
    
    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)
    
def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (227, 227), interpolation = cv2.INTER_LINEAR)
    
def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image





def captcha():
    img = requests.get("http://course.uch.edu.tw/stdsel/System_D/dvcode.asp?")
    
    with open("captcha.png", 'wb') as file:
        file.write(img.content)
        
    img = cv2.imread("captcha.png")
    ret, thresh = cv2.threshold(img, 127, 255,  #　門檻值轉黑白影像
                                cv2.THRESH_BINARY_INV)   
        
    denoise=cv2.fastNlMeansDenoising(thresh, h=60)    # 去除 Noise
        
    ret, thresh = cv2.threshold(denoise, 127, 255,  #　門檻值轉黑白影像
                                cv2.THRESH_BINARY_INV) 
    
    crop_img = thresh[0:24, 0:33]
    crop_img = cv2.resize(crop_img,(227,227))
    cv2.imwrite("1.png",crop_img)
        
        
    crop_img = thresh[0:24, 33:62]
    crop_img = cv2.resize(crop_img,(227,227))
    cv2.imwrite("2.png",crop_img)
        
    
    crop_img = thresh[0:24, 62:94]
    crop_img = cv2.resize(crop_img,(227,227))
    cv2.imwrite("3.png",crop_img)
        
        
    crop_img = thresh[0:24, 94:124]
    crop_img = cv2.resize(crop_img,(227,227))
    cv2.imwrite("4.png",crop_img)
    
    
    for i in range(1,5):
                
        imageFile = str(i)+".png"
        image = Image.open(imageFile)
        image = update_orientation(image)
        image = convert_to_opencv(image)
              
        image = resize_down_to_1600_max_dim(image)
        h, w = image.shape[:2]
        min_dim = min(w,h)
        max_square_image = crop_center(image, min_dim, min_dim)
        augmented_image = resize_to_256_square(max_square_image)
        #network_input_size = 227
        output_layer = 'loss:0'
        input_node = 'Placeholder:0'
        #cv2.imwrite ("output1.jpg",augmented_image )
        with tf.Session() as sess:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })
            #predictions, = sess.run(prob_tensor, {input_node: [image] })
            
            # Print the highest probability label
            highest_probability_index = np.argmax(predictions)
            #print('Classified as: ' + labels[highest_probability_index])
            result = str(labels[highest_probability_index])
            print(imageFile + "辨識結果:"+ result)
    print("辨識完成")
    cv2.imshow("123",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

while True:
    captcha()


