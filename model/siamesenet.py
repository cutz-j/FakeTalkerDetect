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
from model.alexnet import AlexNet
from utils.metrics import *

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

    im_in = Input(shape=input_shape)
    x1 = base_model([im_in])

    feat_x = Lambda(lambda x: l2_normalize(x, axis=1, epsilon=1e-7))(x1)
    feat_x = Dropout(0.0)(feat_x)

    model_top = Model(inputs=[im_in], outputs=feat_x)
    model_top.summary()

    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    h1 = model_top(left_input)
    h2 = model_top(right_input)
    distance = Lambda(euclidean_distance)([h1, h2])
    siam_model = Model(inputs=[left_input, right_input], outputs=distance)
    return siam_model