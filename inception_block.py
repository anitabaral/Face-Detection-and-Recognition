#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.layers.core import Layer, Lambda, Flatten, Dense
import keras.backend as K
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,      Dropout, Dense, Input, concatenate,          GlobalAveragePooling2D, AveragePooling2D,    Flatten, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

import cv2
import numpy as np
from keras import backend as K
from keras.utils import np_utils

import math
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler


# In[2]:


def inception_module(filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    conv_1x1 = Conv2D(filters_1x1, (1, 1),
                      padding='same',
                      activation='relu',
                      name='inception_1x1')(X)

    conv_3x3 = conv2D(filters_3x3_reduce, (1, 1),
                      padding="same",
                      activation='relu',
                      name='inception_3x3_reduce')(X)
    conv_3x3 = conv2D(filters_3x3, (1, 1),
                      padding="same",
                      activation='relu',
                      name='inception_3x3')(conv_3x3)

    conv_5x5 = conv2D(filters_5x5_reduce, (1, 1),
                      padding='same',
                      activation='relu',
                      name='inception_5x5_reduce')(X)
    conv_5x5 = conv2D(filters_5x5, (1, 1),
                      padding='same',
                      activation='relu',
                      name='inception_5x5')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(X)
    pool_proj = MaxPool2D(filters_pool_proj,
                          strides=(1, 1),
                          padding='same',
                          activation='relu')(pool_proj)

    inception = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

    return inception


# In[3]:


def faceRecoModel(input_shape):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    #first block
    X = Conv2D(64, (7, 7), padding='same', strides=(2, 2), name='conv_1')(X)
    X = BatchNormalization(name='bn_1')(X)
    X = Activation('relu')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1')(X)

    #second_block
    X = Conv2D(64, (1, 1), strides=(1, 1), name='conv_2')(X)
    X = BatchNormalization(name='bn_2')(X)
    X = Activation('relu')(X)

    X = ZeroPadding2D((1, 1))(X)

    X = Conv2D(192, (3, 3), strides=(1, 1), name='conv_3')(X)
    X = BatchNormalization(name='bn_3')(X)
    X = Activation('relu')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D(pool_size=3, strides=2, name='max_pool_2')(X)

    X = inception_module(X,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')

    X = inception_module(X,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')

    X = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3')(X)

    X = inception_module(X,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')

    X = inception_module(X,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')

    X = inception_module(X,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')

    X = inception_module(X,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d')

    X = inception_module(X,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')

    X = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4')(X)

    X = inception_module(X,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')

    X = inception_module(X,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')

    X = GlobalAveragePooling2D(pool_size=(3, 3),
                               strides=(1, 1),
                               name='avg_pool')(X)
    X = Dropout(0.4)(X)
    X = Flatten()(X)
    X = Dense(128, name='dense_layer')(X)

    X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

    # Create model instance
    model = Model(inputs=X_input, outputs=X, name='FaceRecoModel')

    return model


# In[ ]:




