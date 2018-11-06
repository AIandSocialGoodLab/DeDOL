import numpy as np
from poacher_cnn import Poacher
from poacher_rule import Poacher as Poacher_h
from env import Env
from replay_buffer import ReplayBuffer
import argparse
import sys
import tensorflow as tf
import os
import warnings
import time

from tkinter import *
from tkinter import Tk, Canvas
from maps import Mountainmap
import copy
warnings.simplefilter('error', RuntimeWarning)


class RandomSweepingPatroller():
    '''
    The random sweeping patroller.
    At first time step, random chooses among four directions.
    Then heads towards that direction until it gets to the border of the map.
    Then travels along the border
    Once it finds a footprint, it then follows the footprint
    If there are multiple footprints in one grid, it randomly choose one footprint to follow
    ''' 

    def __init__(self, args, mode = None):
        self.args = args
        self.row_num = args.row_num
        self.column_num = args.column_num
        self.mode = mode

    def infer_action(self, loc, last_action, footprints):
        '''
        args:
            loc: current location of the patroller
            last_action: the last action of the patroller
            footpriints: a list containing directions of footprints like ['left', 'right', 'up', 'down']
        '''
        # there are footprints, randomly choose one to follow
        if len(footprints) > 0: 
            idx = np.random.choice(len(footprints))
            action = footprints[idx]
            return action

        # if separating modes:
        if self.mode is not None:
            if loc[0] == self.row_num // 2 and loc[1] == self.column_num // 2:
                if self.mode == 0:
                    return 'up'
                elif self.mode == 1:
                    return 'right'
                elif self.mode == 2:
                    return 'down'
                elif self.mode == 3:
                    return 'left'

            if self.mode == 0 and loc[0] == 0 and loc[1] == self.column_num // 2:
                # print('working')
                return 'left'

            elif self.mode == 1 and loc[0] == self.row_num // 2 and loc[1] == self.column_num - 1:
                # print('working')
                return 'up'

            elif self.mode == 2 and loc[0] == self.row_num - 1 and loc[1] == self.column_num // 2:
                # print('working')
                return 'right'

            elif self.mode == 3 and loc[0] == self.row_num // 2 and loc[1] == 0:
                # print('working')
                return 'down'                 

        # No footprint
        if self.leftup_corner(loc):
            if last_action == 'up':
                return 'right'
            elif last_action == 'left':
                return 'down'

        if self.rightup_corner(loc):
            if last_action == 'up':
                return 'left'
            elif last_action == 'right':
                return 'down'

        if self.leftdown_corner(loc):
            if last_action == 'down':
                return 'right'
            elif last_action == 'left':
                return 'up'
            
        if self.rightdown_corner(loc):
            if last_action == 'down':
                return 'left'
            elif last_action == 'right':
                return 'up'

        if self.verticalborder(loc):
            if last_action == 'up':
                return 'up'
            elif last_action == 'down':
                return 'down'
            
        if self.horizontalborder(loc):
            if last_action == 'left':
                return 'left'
            elif last_action == 'right':
                return 'right'

        actions = ['up', 'down', 'left', 'right']
        distances = self.distance(loc)
        actions = [(actions[i], distances[i]) for i in range(4)]

        mindis = 10000
        for a in actions:
            if a[1] > 0 and a[1] < mindis:
                mindis = a[1]
        
        ret = []
        for a in actions:
            if a[1] == mindis:
                ret.append(a[0])

        idx = np.random.choice(len(ret))
        return ret[idx]

    def get_action_probs(self, loc, last_action, footprints):
        '''
        args:
            loc: current location of the patroller
            last_action: the last action of the patroller
            footpriints: a list containing directions of footprints like ['left', 'right', 'up', 'down']
        '''
        # there are footprints, randomly choose one to follow
        if len(footprints) > 0: 
            prob = 1. / len(footprints)
            dic = {}
            for a in footprints:
                dic[a] = prob
            return dic

        # if separating modes:
        if self.mode is not None:
            if loc[0] == self.row_num // 2 and loc[1] == self.column_num // 2:
                if self.mode == 0:
                    return {'up':1}
                elif self.mode == 1:
                    return {'right':1}
                elif self.mode == 2:
                    return {'down':1}
                elif self.mode == 3:
                    return {'left':1}

            if self.mode == 0 and loc[0] == 0 and loc[1] == self.column_num // 2:
                # print('working')
                return {'left':1}

            elif self.mode == 1 and loc[0] == self.row_num // 2 and loc[1] == self.column_num - 1:
                # print('working')
                return {'up':1}

            elif self.mode == 2 and loc[0] == self.row_num - 1 and loc[1] == self.column_num // 2:
                # print('working')
                return {'right':1}

            elif self.mode == 3 and loc[0] == self.row_num // 2 and loc[1] == 0:
                # print('working')
                return {'down':1}                 

        # No footprint
        if self.leftup_corner(loc):
            if last_action == 'up':
                return {'right':1}
            elif last_action == 'left':
                return {'down':1}

        if self.rightup_corner(loc):
            if last_action == 'up':
                return {'left':1}
            elif last_action == 'right':
                return {'down':1}

        if self.leftdown_corner(loc):
            if last_action == 'down':
                return {'right':1}
            elif last_action == 'left':
                return {'up':1}
            
        if self.rightdown_corner(loc):
            if last_action == 'down':
                return {'left':1}
            elif last_action == 'right':
                return {'up':1}

        if self.verticalborder(loc):
            if last_action == 'up':
                return {'up':1}
            elif last_action == 'down':
                return {'down':1}
            
        if self.horizontalborder(loc):
            if last_action == 'left':
                return {'left':1}
            elif last_action == 'right':
                return {'right':1}

        actions = ['up', 'down', 'left', 'right']
        distances = self.distance(loc)
        actions = [(actions[i], distances[i]) for i in range(4)]

        mindis = 10000
        for a in actions:
            if a[1] > 0 and a[1] < mindis:
                mindis = a[1]
        
        ret = []
        for a in actions:
            if a[1] == mindis:
                ret.append(a[0])

        prob = 1./ len(ret)
        dic = {}
        for a in ret:
            dic[a] = prob
        
        return dic

 

    def distance(self, loc):
        '''return the distance to the up, down, left, right border'''
        up = loc[0]
        down = self.row_num - 1 - loc[0]
        left = loc[1]
        right = self.column_num - 1 - loc[1]
        return up, down, left, right

    def verticalborder(self, loc):
        if loc[1] == 0 or loc[1] == self.column_num - 1:
            return True
        return False

    def horizontalborder(self,loc):
        if loc[0] == 0 or loc[0] == self.row_num - 1:
            return True
        return False

    def leftup_corner(self, loc):
        if loc[0] == 0 and loc[1] == 0:
            return True
        return False

    def leftdown_corner(self, loc):
        if loc[0] == self.row_num - 1 and loc[1] == 0:
            return True
        return False

    def rightup_corner(self, loc):
        if loc[0] == 0 and loc[1] == self.column_num - 1:
            return True
        return False

    def rightdown_corner(self, loc):
        if loc[0] == self.row_num - 1 and loc[1] == self.column_num - 1:
            return True
        return False
