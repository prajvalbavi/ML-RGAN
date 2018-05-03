#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 23:30:29 2018

@author: prajvalb
"""

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


class DQN:
    def __init__(self, in_shape, out_shape, learning_rate=0.01):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.learning_rate = learning_rate
        
    def create_model(self):
        model = Sequential()
        #Same as input_shape=(784,)
        model.add(Dense(24, input_dim=self.in_shape , activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.out_shape))
        model.compile(loss="mean_squared_error",optimizer=Adam(lr=self.learning_rate))
        return model
    
    def create_conv_model(self):
        ishape = (28, 28, 1)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=ishape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.out_shape))

        model.compile(loss="mean_squared_error",
              optimizer=Adam(lr=self.learning_rate))
        return model
    