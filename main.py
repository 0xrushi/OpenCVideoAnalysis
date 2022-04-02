import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, '/mnt/hdd2/gender_detect/yoloface/')

from face_detector import YoloDetector
import cv2
from outerutils.constants import *


# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)    

def predict_gender(face_img):
    """
    Predict the gender of the face shown in the image
    Input: face_img, numpy array
    Return: gender label
    """
    print("face_img.shape", face_img.shape)
    if face_img.shape[1] > 105:
        face_img = image_resize(face_img, width=105)

    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()

    i = gender_preds[0].argmax()
    gender = GENDER_LIST[i]
    gender_confidence_score = gender_preds[0][i]
    label = f"{gender}-{gender_confidence_score*100:.1f}%"
    print(label)
    return label

def predict_age(face_img):
    """
    Predict the age of the face shown in the image
    Input: face_img, numpy array
    Return: gender label
    """
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    age_preds =  age_net.forward()
    i = age_preds[0].argmax()
    age = AGE_INTERVALS[i]
    age_confidence_score = age_preds[0][i]
    # Draw the box
    label = f"Age: {age}-{age_confidence_score*100:.1f}%"
    return label

model = YoloDetector(target_size=720,gpu=0,min_face=90)
orgimg = np.array(Image.open('frames_saved/frame240.jpg')).astype(np.uint8)
print(orgimg.shape)
bboxes, points = model.predict(orgimg)
print(bboxes)
JUST_SAVE_BOUNDING_BOXES = True

# iterate through all the faces
for c in bboxes[0]:
    print(c)
    rect = c
    x1,y1,x2,y2 = rect
    h = y2-y1
    w = x2-x1
    print(f"h is {h} \n w is {w}")
    cv2.rectangle(orgimg,(x1,y1),(x2,y2),(0,255,0),2)
    # keep in mind index become reversed during cropping
    face = orgimg[y1:y2, x1:x2]
    
    custom_plot(face)

    if not JUST_SAVE_BOUNDING_BOXES:
        custom_plot(face)
        # apply gender prediction
        gender_label = predict_gender(face)
        # apply age prediction
        age_label = predict_age(face)
        labeltext = f"Person  {gender_label} \n {age_label}"
        y0, dy = y2+20, 22
        # The loop below is to put text one below other, we cannot use \n directly| change y0 and dy as per your screen size
        for i, line in enumerate(labeltext.split('\n')):
            y = y0 + i*dy
            cv2.putText(orgimg, line, (x1+10, y), 1, 1.8, (0,255,0))

        custom_plot(orgimg)

