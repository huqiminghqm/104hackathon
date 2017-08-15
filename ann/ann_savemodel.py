#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:58:06 2017

@author: liouscott
"""

from keras.utils import np_utils
import numpy as np
import pandas as pd

train_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/user_log.csv', sep='|')

X_train = train_df.values[:, 1:]
y_train = train_df.values[:, 0]

y_train = np_utils.to_categorical(y_train)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras import backend as K

def mcor(y_true, y_pred):
     #matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
 
 
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
 
 
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
 
 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
 
 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
 
     return numerator / (denominator + K.epsilon())




def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

class_weight = {0:1,1:2.5}

classifier = Sequential()
classifier.add(Dense(units=100, kernel_initializer='normal', activation='relu', input_dim=47))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=100, kernel_initializer='normal', activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=2, kernel_initializer='normal', activation='softmax'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])

classifier.fit(X_train, y_train, batch_size=512, epochs=3, validation_split=0.1, class_weight=class_weight)

classifier.save_weights('ytshen_weights.h5')

test_df = pd.read_csv('user_log_test.csv', sep='|')

pd.crosstab(y_test,prediction,rownames=['label'],colnames=['predict'])





