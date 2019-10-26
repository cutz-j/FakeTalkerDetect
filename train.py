import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn import metrics
import glob
from PIL import Image
from tqdm import tqdm, trange
from model.alexnet import AlexNet
from model.siamesenet import SiameseNet
from utils.generators import generator
from utils.metrics import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

## Dir ##
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'
test_dir = 'dataset/test'
ft_dir = 'dataset/fine-tune'
save_weights_dir = 'weights/alexnet.h5'
save_siamese_dir = 'weights/alexnet_siam.h5'

## train parameter ##
img_width, img_height = 64, 64
batch_size = 32

def alexnet_train():
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width), batch_size=batch_size, shuffle=True, class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(img_height, img_width), batch_size=batch_size, shuffle=True, class_mode='binary')     

    model = AlexNet(input_shape=(img_width, img_height, 3), include_top=True, save_weights=save_weights_dir)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])        

    callback_list = [EarlyStopping(monitor='val_acc', patience=5),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)]

    model.fit_generator(train_generator, steps_per_epoch=100, epochs=20,
                                  validation_data=validation_generator, validation_steps=len(validation_generator),
                                  callbacks=callback_list, verbose=1)
    return model

def siamese_train():
    train_gen = generator(ft_dir, 64)

    model = SiameseNet(input_shape=(img_width, img_height, 3), pretrained_model=save_weights_dir, save_weight=save_siamese_dir)
    model.compile(loss=contrastive_loss, optimizer=Adam(), metrics=[siamese_acc])

    model.fit_generator(train_gen, steps_per_epoch=50, epochs=1)
    return model

if __name__ == "__main__":
    alexnet_train()
    siamese_train()