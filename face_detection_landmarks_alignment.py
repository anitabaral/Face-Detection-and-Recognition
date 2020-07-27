#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
import dlib
import cv2
import openface
from skimage import io


# In[39]:


def align_image(img):

    predictor_model = "shape_predictor_68_face_landmarks.dat"
    face_aligner = openface.AlignDlib(predictor_model)
    alignedFace = face_aligner.align(
        96,
        img,
        face_aligner.getLargestFaceBoundingBox(img),
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    return alignedFace

