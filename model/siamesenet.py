import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Lambda, Dropout, SeparableConv2D, add
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model.alexnet import alexnet

def SiameseNet(input_shape=(64, 64, 3), pretrained_model='weights/alexnet.h5', save_weight=''):
    '''
    # Function: Siamese model for keras
    
    # Arguments:
        input_shape: image shape, default (64, 64, 3) 
        pretrained_model: alexnet pretrained model directory
        save_weight: siamese save directory

    # Returns:
        model
    '''
    # two model load for siamese
    model = load_model(pretrained_model)
    base_model = load_model(pretrained_model)

    # layer freeze
    for i in range(len(base_model.layers) - 2):
        base_model.layers[i].trainable = False