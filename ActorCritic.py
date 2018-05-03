#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:29:23 2018

@author: prajvalb
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import tensorflow as tf
import random


class ActorCritc:
    def __init__(self, in_shape, out_shape):
        self.learning_rate = 0.001
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        
    
    def actor_model(self):
        state_input = Input(shape=(self.in_shape,))
        h1 = Dense(24, activation="relu")(state_input)
        h2 = Dense(48, activation="relu")(h1)
        h3 = Dense(24, activation="relu")(h2)
        output = Dense(self.out_shape, activation="relu")(h3)
        
        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        
        return state_input, model
    
    def critic_model(self):
        state_input = Input(shape=(self.in_shape,))
        state_h1 = Dense(24, activation="relu")(state_input)
        state_h2 = Dense(48, activation="relu")(state_h1)
        
        action_input = Input(shape=(self.out_shape,))
        action_h1 = Dense(48, activation="relu")(action_input)

        merged = Add()([state_h2, action_h1])        
        merged_h1 = Dense(24, activation="relu")(merged)
        output = Dense(1, activation="relu")(merged_h1)
        model = Model(input=[state_input, action_input], output=output)
        
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model
    
    