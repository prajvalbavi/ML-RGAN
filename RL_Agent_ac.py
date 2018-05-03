#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:21:31 2018

@author: prajvalb
"""

from collections import deque
import numpy as np
import random
import tensorflow as tf

class DrawAgent:
    """
        Reinforcement Learning Agent Model for training/testing
        with Tabular function approximation

    """

    def __init__(self, sess, rl_ac,  _env, iterations, debug_opt=True, test_debug_opt=True):
        self.rl_ac = rl_ac
        self.sess = sess
        self.benv = _env
        self.n_a = 8
        self.debug_opt = debug_opt
        self.test_debug_opt = test_debug_opt
        self.decay = np.exp(np.log(0.1) / iterations)
        self.ep_val = []
        self.frames = []
        self.experience = deque()
        self.memory_size = 1000
        self.update_nn = 100
        self.learning_rate = 0.01
        self.n_inputs = 786 
        self.sess = sess
        self.actor_state_input, self.actor_model = self.rl_ac.actor_model()
        _, self.target_actor_model = self.rl_ac.actor_model()
        
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.n_a])
        
        actor_model_weigths = self.actor_model.trainable_weights
        
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weigths,
                                        -self.actor_critic_grad)
        
        grads = zip(self.actor_grads, actor_model_weigths)
        
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        
        
        self.critic_state_input, self.critic_action_input, self.critic_model = self.rl_ac.critic_model()
        _,_,self.target_critic_model = self.rl_ac.critic_model()
        
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)
        self.sess.run(tf.initialize_all_variables())        
       
    def update_model(self):
        self.update_actor_model()
        self.update_critic_model()
        
    def update_actor_model(self):
        a_target_weights = self.target_actor_model.get_weights()
        a_model_weights = self.actor_model.get_weights()
        for i in range(len(a_model_weights)):
            a_target_weights[i] = a_model_weights[i]
        self.target_actor_model.set_weights(a_target_weights)
        
    def update_critic_model(self):
        c_target_weights = self.target_critic_model.get_weights()
        c_model_weights = self.critic_model.get_weights()
        for i in range(len(c_model_weights)):
            c_target_weights[i] = c_model_weights[i]
        self.target_critic_model.set_weights(c_target_weights)
        
    
        

    def epsilon_greed(self, epsilon, s):
        actions = np.zeros(self.n_a)
        if np.random.rand() < epsilon:
            a = np.random.randint(self.n_a)
            actions[a] = 1
            if self.debug_opt:
                print('action by random', a)
        else:
            _actions = self.actor_model.predict(s)
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
        maxiter = params.pop('maxiter', 5000)
        maxstep = params.pop('maxstep', 300)

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
                    for sample in minibatch:
                        _s, _fs, _a, _r, _s1, _fs1, _g_f_s1 = sample
                        if not self.benv.checkGoal(_g_f_s1):
                            _fs1 = (_fs1.astype(np.float32) - 127.5) / 127.5
                            _s1 = (_s1.astype(np.float32) - 14) / 14
                            new_state = np.concatenate((_fs1, _s1))
                            new_state = np.expand_dims(new_state, axis=0)
                            target_action = self.target_actor_model.predict(new_state)
                            future_reward = self.target_critic_model.predict([new_state, target_action])
                            _r += gamma * future_reward 
                        
                        cur_state = np.concatenate((_fs, _s))
                        cur_state = np.expand_dims(cur_state, axis=0)
                        _a = np.expand_dims(_a, axis=0)
                        self.critic_model.fit([cur_state, _a], _r, epochs=1, verbose=0)
                        
                    
                    for sample in minibatch:
                        _s, _fs, _a, _r, _s1, _fs1, _g_f_s1 = sample
                        _fs = (_fs.astype(np.float32) - 127.5) / 127.5
                        _s = (_s.astype(np.float32) - 14) / 14
                        curr_state = np.concatenate((_fs, _s))
                        curr_state = np.expand_dims(curr_state, axis=0)
                        predicted_action = self.actor_model.predict(curr_state)
                        grads = self.sess.run(self.critic_grads, feed_dict={
                                self.critic_state_input: curr_state,
                                self.critic_action_input: predicted_action})[0]
                        self.sess.run(self.optimize, feed_dict={
                                self.actor_state_input: curr_state, 
                                self.actor_critic_grad: grads})
                                       
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
        actions = self.target_actor_model.predict(my_state)
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
            actions = self.target_actor_model.predict(my_state)
            a = np.argmax(actions)
            if self.test_debug_opt:
                print("Next Action taken for Agent is: {} from {}".format(a, actions))
            
        return f_frames, s_states
    

