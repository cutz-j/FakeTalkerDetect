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
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import glob
from PIL import Image
from tqdm import tqdm, trange
from utils.generators import generator_res

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

def alexnet_evaluate():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), batch_size=32, shuffle=False, class_mode='binary')
    
    model = load_model(save_weights_dir)
    score  = []
    answer = []
    for i in trange(len(test_generator)):
        y_score = model.predict_on_batch(test_generator[i][0])
        score.append(y_score)
        answer.append(test_generator[i][1])

    score = np.concatenate(score)
    answer = np.concatenate(answer)
    
    predictions = score.copy()

    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0

    predictions[np.isnan(predictions)] = 0

    fpr, tpr, thresholds = roc_curve(answer, score, pos_label=1.)
    cm = confusion_matrix(answer, predictions)
    recall = cm[0][0] / (cm[0][0] + cm[0][1])
    fallout = cm[1][0] / (cm[1][0] + cm[1][1])
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    plt.plot(fpr, tpr, 'o-')
    plt.plot([0, 1], [0, 1], 'k--', label="random guess")

    plt.xlabel('False Positive Rate (Fall-Out)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.show()

    print(metrics.classification_report(answer, predictions))
    print(confusion_matrix(answer, predictions))
    print("FPR=FAR", fallout)
    print("FNR=FRR", 1-recall)
    print('test_acc: ', len(predictions[np.equal(predictions, answer)]) / len(predictions))
    print('thresh:', thresh)
    print('eer:', eer)
    print('aou_roc: ', roc_auc_score(answer, score))
    print(cm)

def siamese_evaluate(model):
    test_generator = generator_res(ft_dir, test_dir, batch_size=1)
    score  = []
    answer = []
    for i in trange(len(test_generator)):
        y_score = model.predict_on_batch(i[0])
        score.append(y_score)
        answer.append(i[1])
    
    score = np.concatenate(score)
    answer = np.concatenate(answer)

    predictions = score.copy()
    predictions[predictions > 0.8] = 1
    predictions[predictions <= 0.8] = 0

    fpr, tpr, thresholds = roc_curve(answer, score, pos_label=1.)
    cm = confusion_matrix(answer, predictions)
    recall = cm[0][0] / (cm[0][0] + cm[0][1])
    fallout = cm[1][0] / (cm[1][0] + cm[1][1])
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    plt.plot(fpr, tpr, 'o-')
    plt.plot([0, 1], [0, 1], 'k--', label="random guess")

    plt.xlabel('False Positive Rate (Fall-Out)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.show()

    print(metrics.classification_report(answer, predictions))
    print(confusion_matrix(answer, predictions))
    print("FPR=FAR", fallout)
    print("FNR=FRR", 1-recall)
    print('test_acc: ', len(predictions[np.equal(predictions, answer)]) / len(predictions))
    print('thresh:', thresh)
    print('eer:', eer)
    print('aou_roc: ', roc_auc_score(answer, score))
    print(cm)
    
if __name__ == "__main__":
    alexnet_evaluate()