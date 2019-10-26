import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras import backend as K
from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Lambda, Dropout, SeparableConv2D, add
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def AlexNet(input_shape=(64, 64, 3), include_top=True, load_weight=''):
    '''
    # Function: Alexnet model for keras
    
    # Arguments:
        input_shape: image shape, default (64, 64, 3) 
        include_top: top layer
        load_weight: pretrained model directory

    # Returns:
        model
    '''

    img_input = Input(shape=input_shape)

    x = Conv2D(96, 11, strides=4, padding='same', use_bias=False)(img_input) # 15
    x = Activation('relu')(x)

    x = Conv2D(256, 5, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x) # 8

    x = Conv2D(384, 3, strides=1, padding='same', use_bias=False)(x) # 15
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x) # 8

    x = Conv2D(384, 3, strides=1, padding='same', use_bias=False)(x) # 15
    x = Activation('relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', use_bias=False)(x)
    x = Activation('relu')(x)

    out = MaxPooling2D(pool_size=3, strides=2, padding='valid')(x) # 8

    if include_top:
        out = GlobalAveragePooling2D()(out)
        out = Dense(4096)(out)
        out = Activation('relu')(out)
        out = Dense(1)(out)
        out = Activation('sigmoid')(out)
    
    model = Model(img_input, out)

    if load_weight:
        model.load_weight(load_weight)

    return model