#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:42:18 2018

@author: prajvalb
"""
import os
import numpy as np
from DQN import DQN #Import the model
from RL_Agent_ac import DrawAgent #Import the RL Agent
from env import ImageEnv #Import ImageEnv
from sklearn.datasets import fetch_mldata #Import the mnist dataset from sklearn
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from ActorCritic import ActorCritc


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch

def plot_animation(frames, repeat=False, interval=1):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[2],cmap=plt.cm.gray)
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)

if __name__ == "__main__":
    
    #Change dir to ML-RGAN if you facing problems of importing the files
    #os.chdir('./../Desktop/ML-RGAN/')

    #Start tf session
    sess = tf.Session()
    K.set_session(sess)
    
    
    #Get the mnist data set
    mnist = fetch_mldata('MNIST original') 
    
    #Pass Goal state as argument to init the ImageEnv 
    env = ImageEnv(mnist.data[20000]) 
    
    #Init the env with start state and the position of the agent in the image
    #startImage = np.copy(mnist.data[0] * 0)
    startImage = np.copy(mnist.data[20004])
    #env.init(startImage, [10,10]) 
    
    #Init the DQN model
    rl_ac = ActorCritc(786, 8)
    
    #Init the RLAgent with model and env
    rldrawAgent = DrawAgent(sess, rl_ac, env, iterations=100, debug_opt=False)
    
    train_rtrace, train_steps, train_frames, ep_val = rldrawAgent.train(startImage, [10, 10])
    np.savez('trainframes',train_frames)
    np.savez('trainframes',train_frames)
    np.savez('trainrewards', train_rtrace)
    np.savez('trainsteps', train_steps)
    
    testframes, agent_states = rldrawAgent.test(startImage, [10,10],  maxstep=1000)
    np.savez('testframes',testframes)
    np.savez('teststates', agent_states)
    
    #someframe = np.load('trainframes.npz')
    #somef = someframe['arr_0']
    #plot_animation(somef)
    sess.close()
    