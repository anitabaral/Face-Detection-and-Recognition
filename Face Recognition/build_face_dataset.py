#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import imutils
import time
import cv2
import os
from imutils.video import VideoStream
import numpy as np


ap = argparse.ArgumentParser()
##--cascade :The path to the Haar cascade file on disk
##--output :The path to the output directory. Images of faces will be stored in this directory
ap.add_argument("-c",
                "--cascade",
                required=True,
                help="path to where the face cascade resides")
ap.add_argument("-o",
                "--output",
                required=True,
                help="path to output directory")
args = vars(ap.parse_args())

face_classifier = cv2.CascadeClassifier(args["cascade"])
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)
total = 0

def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (96, 96), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((96,96), np.uint8), img
    return (x,w,y,h), roi_gray, img


while True:

    ret, frame = cap.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=400)
    rect, face, image = face_detector(frame)
   
    cv2.imshow('All', image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("k"):
        p = os.path.sep.join([args["output"], "{}.png".format(
              str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total += 1
        # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")

cap.release()
cv2.destroyAllWindows()


