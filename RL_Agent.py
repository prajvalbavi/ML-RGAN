#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:21:31 2018

@author: prajvalb
"""

from collections import deque
import numpy as np
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import requests


class DrawAgent:
    """
        Reinforcement Learning Agent Model for training/testing
        with Tabular function approximation

    """

    def __init__(self, model, t_model,  _env, iterations, debug_opt=True, test_debug_opt=True):
        self.benv = _env
        self.n_a = 8
        self.debug_opt = debug_opt
        self.test_debug_opt = test_debug_opt
        self.decay = np.exp(np.log(0.1) / iterations)
        self.ep_val = []
        self.frames = []
        self.experience = deque()
        self.memory_size = 1000
        self.update_nn = 50
        self.learning_rate = 0.01
        self.n_inputs = 786 
        self.model = model
        self.target_model = t_model
        
    def update_model(self):
        target_weights = self.target_model.get_weights()
        weigths = self.model.get_weights()
        for i in range(len(target_weights)):
            weigths[i] = target_weights[i]
        self.model.set_weights(weigths)

    def epsilon_greed(self, epsilon, s):
        actions = np.zeros(self.n_a)
        if np.random.rand() < epsilon:
            a = np.random.randint(self.n_a)
            actions[a] = 1
            if self.debug_opt:
                print('action by random', a)
        else:
            _actions = self.model.predict(s)
            a = np.argmax(_actions)
            actions[a] = 1
            if self.debug_opt:
                print('action by argmax', a)
        return actions        

    def train(self, initstate, agentStartState, **params):
        # parameters
        gamma = params.pop('gamma', 0.99)
        alpha = params.pop('alpha', 0.1)
        epsilon = params.pop('epsilon', 0.99)
        maxiter = params.pop('maxiter', 5)
        maxstep = params.pop('maxstep', 200)

        # online train
        # rewards and step trace
        rtrace = []
        steps = []
        for j in range(1, maxiter):
            print("****************************")
            print('New game')
            print('Episode Number {}'.format(j))
            
            # get init state from arg
            _initstate = np.copy(initstate)
            self.benv.init(_initstate, agentStartState)
            
            # get the current state after env init
            s = self.benv.get_curr_state()
            self.frames.append(np.copy(self.benv.state))
            
            if self.debug_opt:
                print('Start state ', s)
        
                
            rewards = []
            # run simulation for max number of steps
            for _step in range(maxstep):
                print('Episode Number: {}, Step Number {}'.format(j, _step))
                # for a given state s and selected action a, get a new state s1 and reward r
                # selection an action given an state s
                f_s = np.reshape(np.copy(self.benv.state), (784, ))
                f_s = (f_s.astype(np.float32) - 127.5) / 127.5
                s = (s.astype(np.float32) - 14) / 14
                my_state = np.concatenate((f_s, s))
                plt.imshow(self.benv.state)
                plt.show()
                my_state = np.expand_dims(my_state, axis=0)
                #print(my_state)
                a = self.epsilon_greed(epsilon, my_state)
                if self.debug_opt:
                    print('Selected action before argmax' , a)
                    print('Selected action ', np.argmax(a))
                    
                    
                s1, r = self.benv.get_next(np.argmax(a))
                f_s1 = np.reshape(np.copy(self.benv.state), (784, ))
                g_f_s1 = np.copy(f_s1)
                f_s1 = (f_s1.astype(np.float32) - 127.5) / 127.5
                s1 = (s1.astype(np.float32) - 14) / 14
                
                self.frames.append(np.copy(self.benv.state))
                rewards.append(r)
                
                if self.debug_opt:
                    print('Move for step number {} with reward {}'.format(_step,r))
                    
               
                
                # As action we take will and new states generated will be highly correlated, so we would not want 
                # to train the NN with these states, instead we would want it to store them in some buffer and 
                # get random states from the buffer 
                # if the buffer gets full, we need to pop the oldest memory from the experience
                self.experience.append([s, f_s ,a, r, s1, f_s1, g_f_s1])
                if len(self.experience) > self.memory_size:
                    self.experience.popleft() 
                    
                   
                # When there are enough memory in the experince, we will use these to start training the NN,
                # to find the Q value's for the action's given a state s1
                # Why ? We took action a when we see a state s, and we now we would need to update Q values,
                # for that we are in state s1, so we want to take action a1, now where do we get the action a1 ?
                # That is where NN comes in, so we are getting all the states s1 and get the Q values for the action
                if _step > self.update_nn:
                    print("Will update neural network with forward pass")
                    minibatch = random.sample(self.experience, 32)
                    _s = [__s[0] for __s in minibatch]
                    _fs = [__s[1] for __s in minibatch] 
                    _a = [__s[2] for __s in minibatch]
                    _r = [__s[3] for __s in minibatch]
                    _s1 = [__s[4] for __s in minibatch]
                    _fs1 = [__s[5] for __s in minibatch]
                    _g_f_s1 = [__s[6] for __s in minibatch]
                    
                    _tmp_s1 = np.vstack([np.concatenate((fs, s)) for fs, s in zip(_fs1, _s1)])
                    _tmp_s = np.vstack([np.concatenate((fs, s)) for fs, s in zip(_fs, _s)])
                    
                    
                    tmp_a1 = self.model.predict(_tmp_s1)
                    if self.debug_opt:
                        print("Predicted action", tmp_a1)
                    
                    
                # Now we want to reduce the loss for the NN, so that it will converge to a good weights
                    _y = []
                    for i in range(len(minibatch)):
                        if self.benv.checkGoal(_g_f_s1[i]):
                            _tmp_y = np.zeros(self.n_a)
                            _tmp_index = np.argmax(tmp_a1[i])
                            _tmp_y[_tmp_index] = _r[i] + self.gamma * (tmp_a1[i][_tmp_index] - _a[i])
                            _y.append(_tmp_y)
                        else:
                            _tmp_index = np.argmax(tmp_a1[i])
                            _tmp_y = np.zeros(self.n_a)
                            _tmp_y[_tmp_index] = _r[i] 
                            _y.append(_tmp_y)

                    _y = np.vstack(_y)
                    
                    #Optimize the loss
                    self.target_model.fit(_tmp_s, _y, epochs=1, verbose=0)
                if _step % 35 == 0:
                    self.update_model()
                    
                    
                if self.benv.isGoal():
                    print("Yolo: I am at goal")
                    break
                

                s = s1
                #a = a1
            epsilon *= self.decay
            self.ep_val.append(epsilon)
            rtrace.append(np.sum(rewards))
            steps.append(_step+1)
        someframe = self.frames
        return rtrace, steps, someframe, self.ep_val  # last trace of trajectory
    
    def test(self, initstate, start, maxstep=1000):
        f_frames = []
        s_states = []
        _initstate = np.copy(initstate)
        
        self.benv.init(_initstate, start)
        s = self.benv.get_curr_state()
        if self.test_debug_opt:
            print("Start state for Agent {}".format(s))
            
        s_states.append(s)
        f_frames.append(np.copy(self.benv.state))
        
        _tmp_s = np.concatenate((np.reshape(np.copy(self.benv.state), (784,)), s))
        _tmp_s[:785] = (_tmp_s[:785].astype(np.float32) - 127.5) / 127.5
        _tmp_s[785:787] = (_tmp_s[785:787].astype(np.float32) - 14) / 14
        my_state = np.expand_dims(_tmp_s, axis=0)
        # selection an action
        actions = self.model.predict(my_state)
        a = np.argmax(actions)
        if self.test_debug_opt:
            print("Start Action taken for Agent is: {} from {}".format(a, actions))
        
        # run simulation for max number of steps
        for step in range(maxstep):
            # move
            s1, r = self.benv.get_next(a)
            #s1 = self.benv.get_curr_state()
            
            if self.test_debug_opt:
                print("Reward {}".format(r))
                print("Next state {}".format(s1))
                
            _tmp_s = np.concatenate((np.reshape(np.copy(self.benv.state), (784,)), s1))
            _tmp_s[:785] = (_tmp_s[:785].astype(np.float32) - 127.5) / 127.5
            _tmp_s[785:787] = (_tmp_s[785:787].astype(np.float32) - 14) / 14
            my_state = np.expand_dims(_tmp_s, axis=0)
            s_states.append(s1)
            
            f_frames.append(np.copy(self.benv.state))
            if self.benv.isGoal():  # reached the goal
                break
            actions = self.model.predict(my_state)
            a = np.argmax(actions)
            if self.test_debug_opt:
                print("Next Action taken for Agent is: {} from {}".format(a, actions))
            
        return f_frames, s_states
    

