#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:41:21 2018

@author: prajvalb
"""

#import libraries
import numpy as np
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt

class ImageEnv:
    """
        ImageEnv is the environment for modifying the images
    """

    def __init__(self, goal):
        self.state = None
        self.goal = np.reshape(goal, (28, 28))  # if using ssmi
        self._actions = [[0, -1, 0], [0, -1, 1],
                         [0, 1, 0], [0, 1, 1],
                         [-1, 0, 0], [-1, 0, 1],
                         [1, 0, 0], [1, 0, 1]]
        self._s = None
        self._size = self.goal.shape
        self._orig_s = None

    def init(self, state, _state=None):
        # to generate the original noisy 28*28 image
        self.state = np.reshape(state, (28, 28))
        if _state is None:
            s = [0, 0]
        else:
            s = _state
        self._s = np.asarray(s)
        self._orig_s = np.copy(self._s)

    def get_reward(self):
        # use image compare for reward
        # can use D o/p also as reward
        res = compare_ssim(self.state, self.goal, full=True)[0]
        if res < 0.6:
            return -1
        elif res >= 0.6 and res < 0.7:
            return 5000
        else:
            return 100000

    def get_actions(self):
        # return list of actions
        return self._actions

    def get_curr_state(self):
        # will return the current image state
        return self._s

    def check_state(self, s):
        if s[0] < 0 or s[1] < 0 or \
                s[0] >= self._size[0] - 1 or s[1] >= self._size[1] - 1:
            return True

    def get_next(self, action):
        # take the given action and return the reward
        # actions can be: 1. modify a block of 2x2 image by setting it to 0 or 255
        # actions 2. generate a random image and add it to the current state
        # reward 1. image similarity index
        # reward 2. output of the discriminator
        temp_s = self._s + self._actions[action][0:2]
        if self.check_state(temp_s):
            tmp_s = self._s
            self._s = self._orig_s
            return tmp_s, -10
        self._s = self._s + self._actions[action][0:2]
        self.state[[self._s[0]], [self._s[1]]] = self._actions[action][2:3][0] * 255
        reward = self.get_reward()
        return self._s, reward

    def isGoal(self):
        # checks if the image is near to the goal
        return True if compare_ssim(self.state, self.goal, full=True)[0] >= 0.9 else False

    def checkGoal(self, stateFrame):
        return True if compare_ssim(np.reshape(stateFrame, (28, 28)), self.goal, full=True)[0] >= 0.9 else False