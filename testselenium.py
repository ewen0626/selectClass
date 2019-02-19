# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from selenium import webdriver
import selenium.webdriver.support.ui as ui
import cv2
import time
#browser = webdriver.Chrome()
browser = webdriver.PhantomJS()
browser.get('http://course.uch.edu.tw/stdsel/System_D/signform.asp')
browser.maximize_window()

element = browser.find_elements_by_class_name("stu")[1].click() #進入選課系統

wait = ui.WebDriverWait(browser,10)
wait.until(lambda browser: browser.find_elements_by_name("fm1"))
time.sleep(5)
browser.save_screenshot("screenshot.png")

element = browser.find_elements_by_name("fm1")

element = browser.find_element_by_name("txtaccount")

element.send_keys("M10711004")

element = browser.find_element_by_name("txtpassword")
element.send_keys("yuda39429")


element = browser.find_element_by_xpath("/html/body/form/center[2]/table/tbody/tr/td/p[2]/img")


left = element.location['x']
right = element.location['x'] + element.size['width']
top = element.location['y']
bottom = element.location['y'] + element.size['height']
"""
left = 1000
right = 1155
top = 266
bottom = 293
"""
"""
from PIL import Image
img = Image.open("screenshot.png")
img = img.crop((left,top,right,bottom))
img.show()
img.save("test.png")

"""
img = cv2.imread("screenshot.png")
img = img[top:bottom,left:right]
#cv2.resize(img,(124,24))
cv2.imwrite("captcha.png",img)

img = cv2.imread("captcha.png")
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
    
crop_img = img[0:24, 0:33]
cv2.imwrite("1.png",crop_img)
    
    
crop_img = img[0:24, 33:62]
cv2.imwrite("2.png",crop_img)
    
    
crop_img = img[0:24, 62:94]
cv2.imwrite("3.png",crop_img)
    
    
crop_img = img[0:24, 94:124]
cv2.imwrite("4.png",crop_img)






import tensorflow as tf
import os
from os import listdir
from os.path import isfile, isdir, join
import shutil
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

from PIL import Image
import numpy as np


for i in range(1,5):
    
    imageFile = str(i)+".png"
    image = Image.open(imageFile)
    
    # Update orientation based on EXIF tags, if the file has orientation info.
    image = update_orientation(image)
    
    # Convert to OpenCV format
    image = convert_to_opencv(image)
    
    
    # If the image has either w or h greater than 1600 we resize it down respecting
    # aspect ratio such that the largest dimension is 1600
    image = resize_down_to_1600_max_dim(image)
    
    # We next get the largest center square
    h, w = image.shape[:2]
    min_dim = min(w,h)
    max_square_image = crop_center(image, min_dim, min_dim)
    
    # Resize that square down to 256x256
    augmented_image = resize_to_256_square(max_square_image)
    #augmented_image = image
    
    
    
    # The compact models have a network size of 227x227, the model requires this size.
    network_input_size = 227
    
    # Crop the center for the specified network_input_Size
            #augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
    
            #cv2.namedWindow("Image")
            #cv2.imshow("Image", augmented_image)
            #cv2.imwrite ("output1.jpg",augmented_image )
            #cv2.waitKey (0)
      
    # These names are part of the model and cannot be changed.
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






"""

element = browser.find_element_by_name("code")
element.send_keys("1234")

#browser.find_elements_by_name("txtaccount").send_keys("M10711004")

#browser.find_elements_by_name("txtpassword")[1].send_keys("yuda39429")
"""
