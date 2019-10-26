import numpy as np
from keras import backend as K

def manDist(x):
    result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
    return result

def euclidean_distance(inputs):
    assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v + 1e-7)), axis=1, keepdims=True))  

def contrastive_loss(y_true,y_pred):
    margin=1.4
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def siamese_acc(y_true, y_pred):
    return K.mean((K.equal(y_true, K.cast(y_pred > 0.4, K.floatx()))), axis=1)