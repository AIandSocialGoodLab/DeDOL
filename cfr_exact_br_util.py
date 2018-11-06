import numpy as np
from env import Env
import sys
import argparse
import time
from patroller_cnn import Patroller_CNN as Patroller
from poacher_cnn import Poacher
from poacher_rule import Poacher as Poacher_h
from env import Env
from replay_buffer import ReplayBuffer
import os
import re
import copy
from maps import generate_map
import tensorflow as tf
from patroller_randomsweeping import RandomSweepingPatroller
import gc
from DeDOL_util import tf_copy


class InfoSet():
    '''
    The information set data structure
    '''
    def __init__(self, infokey, player, actions = None):
        self.infokey = infokey
        self.player = player
        self.RegretSum = np.zeros(len(actions)) 
        self.AvgStrategy = np.zeros(len(actions)) 
        self.action_num = len(actions)
        self.actions = actions
        self.action_values = np.zeros(self.action_num)
        self.nodes = []

    def get_strategy(self):
        '''
        get the stored strategy at the information set
        '''
        regret = np.clip(self.RegretSum, a_min = 0, a_max = None)
        total_regret = np.sum(regret)
        if total_regret > 0:
            return regret / total_regret
        else:
            return np.ones(self.action_num) / self.action_num 
        
    def get_average_strategy(self):
        '''
        compute the average strategy during the cfr iterations
        '''
        if np.sum(self.AvgStrategy) > 0:
            return self.AvgStrategy / np.sum(self.AvgStrategy)
        else:
            return np.ones(self.action_num) / self.action_num

class tree_node():
    '''
    data structure for a node in the game tree
    '''
    def __init__(self, action_history, p1infoset, p2infoset, player, chance_prob):
        self.action_history = action_history # the action histories of player1, player2, chance player, unique for each node
        self.p1infokey = p1infoset # infokey
        self.p2infokey = p2infoset # infokey
        self.player = player
        self.chance_prob = chance_prob # accmulated chance probabilities of reaching this node
        self.actions = None # the valid actions at this node
        self.best_action = None # poacher best action at this node
        self.best_utility = None # poacher best utility at this node
        self.leaf = False # whethe the node is at leaf
        self.u1 = 0 # leaf node p1 (patroller) utility
        self.u2 = 0 # leaf node p2 (poacher) utility
        self.po_no_move = False # whether poacher can still move at this node (caught or returned home)
        self.chance_probs = 0 # at the chance node, the probability for each chance action
        self.pa_state = None # the pa_state at this node, needed for DQN agent
        self.pa_loc = None # the pa loc at this node, needed for RandomSweeping agent
        self.po_trace = None
        self.RS_footprints = None # the footprints patroller observed at this node, needed for randomsweeping agent
        self.po_loc = None
        self.local_trace = None
        self.local_snare = None
        self.snare_num = None

    def store_game(self, game_states):
        self.game_states = game_states

    def store_actions(self, actions):
        self.actions = actions


class GameTree():
    '''
    Turning our GSG-I game into a game tree
    '''
    def __init__(self, game, args, sess):
        self.game = game
        self.args = args
        self.infodic = {}
        self.cfrlog = open(self.args.cfrlog, 'w')
        self.p1infoprob = {} # record the probabilities of getting into a p1infoset when computing best response of poacher
        self.tree_nodes_dic = {} # record all tree nodes
        self.sess = sess
        self.DQN_pa_action_probs = {} # record the behaviour strategy at a given node for DQN and RS agents. 
                                      # key: action_history specifying a node, value: action_probs at that node
        self.p2_bs_dic = {} # record player2's best response at a given information set. save running time.
        self.p1seqdic = {} # record all player1's action sequences
        self.p2seqdic = {}

    def solve_CFR(self, iteration):
        '''
        Run CFR iterations
        '''
        global game_state_counter, infoset_container
        game_state_number = []
        infoset_number = []

        for iter in range(iteration):
            begin = time.time()
            game_state_counter, infoset_container = 0, []
            self.run_CFR(action_history='', p1p=1, p2p=1, player='c')
            # print('game states counter is {0}'.format(game_state_counter))
            # print('info sets counter is {0}'.format(len(infoset_container)))
            # self.cfrlog.write('game states counter is {0} \n'.format(game_state_counter))
            # self.cfrlog.write('info sets counter is {0} \n'.format(len(infoset_container)))
            end = time.time()
            game_state_number.append(game_state_counter)
            infoset_number.append(len(infoset_container))
            # self.cfrlog.write('game states counter average is {0} \n'.format(np.mean(game_state_number)))
            # self.cfrlog.write('info sets counter is {0} \n'.format(np.mean(infoset_number)))
            self.cfrlog.flush()
            print('time cost {0}'.format(end - begin))
            if iter > 0 and iter % self.args.cfr_test_every_episode == 0:
                u1, u2 = self.compute_po_best_response(action_history='', player='c')
                print('iteration {0} Poacher best response utility {1}'.format(iter, u2))
                self.cfrlog.write('iteration {0} Poacher best response utility {1} \n'.format(iter, u2))
                self.cfrlog.flush()
                gc.collect()

        # self.save()

    # no usage now
    def save(self):
        save_file = open(self.args.save_path + self.args.cfr_save, 'w')
        for inforkey in self.infodic:
            save_file.write(str(inforkey) + ' ')
            inforset = self.infodic[inforkey]
            strategy = inforset.get_average_strategy()
            for x in strategy:
                save_file.write(str(x) + ' ')
            save_file.write('\n')
        save_file.close()

    def run_CFR(self, action_history, p1p, p2p, player):
        '''
        run the cfr main iterations
        params:
            action_history: specifying the node we are currently at
            p1p/p2p: player1 or 2's counterprob of reaching this node
            player: which player to act. {'1', '2' 'c'} -> {player1, player2, chance player}
        Return 
            player1 and player2's expected utilities at the current node using the current cfr strategy
        '''
        global game_state_counter, infoset_container
        game_state_counter += 1

        treenode = self.tree_nodes_dic.get(action_history)
        if treenode is None:
            print(action_history)
        assert treenode is not None

        if treenode.leaf:
            return treenode.u1, treenode.u2
        
        if player == 'c': # chance player
            u1, u2 = 0.0, 0.0
            if action_history == '': # reset game
                if self.args.po_location is None:
                    chance_actions = [0,1,2,3]
                    probs = [0.25, 0.25, 0.25, 0.25]
                    for idx in range(len(chance_actions)):
                        self.game.reset_game(chance_actions[idx])
                        action_history_ = action_history + str(chance_actions[idx])
                        u1_, u2_ = self.run_CFR(action_history_, probs[idx], probs[idx], 1)
                        u2 += probs[idx] * u2_
                        u1 += probs[idx] * u1_
                    return  u1, u2
                else:
                    self.game.reset_game()
                    action_history_ = action_history + str(self.args.po_location)
                    u1, u2 = self.run_CFR(action_history_, p1p = 1, p2p = 1, player = 1)
                    return u1, u2

            # recall chance action is a number. in binary form, each bit means wether to catch an animal at a cell with a snare. 
            chance_actions = treenode.actions
            probs = treenode.chance_probs
            u1 = 0
            u2 = 0
            actionidx = np.random.randint(low=0, high= len(chance_actions))
            # for idx in range(len(chance_actions)):
            #     action_history_ = action_history + str(chance_actions[idx])
            #     p1u, p2u = self.run_CFR(action_history_, p1p * probs[idx], p2p * probs[idx], 1)
            #     u1 += p1u * probs[idx]
            #     u2 += p2u * probs[idx]
            action_history_ = action_history + str(chance_actions[actionidx])
            p1u, p2u = self.run_CFR(action_history_, p1p * probs[actionidx], p2p * probs[actionidx], 1)
            return p1u, p2u

        elif player == 1:
            p1infokey = treenode.p1infokey
            p1infoset = self.infodic.get(p1infokey)
            assert p1infoset is not None
            if p1infokey not in infoset_container:
                infoset_container.append(p1infokey)

            U1 = np.zeros(p1infoset.action_num)
            U2 = np.zeros(p1infoset.action_num)
            U1node = 0 
            U2node = 0
            p1strategy = p1infoset.get_strategy()
            for i, a1 in enumerate(p1infoset.actions):
                action_history_ = action_history + str(a1[0])
                U1[i], U2[i] = self.run_CFR(action_history_, p1p * p1strategy[i], p2p, 2)
                U1node += p1strategy[i] * U1[i]
                U2node += p1strategy[i] * U2[i]

            for i in range(p1infoset.action_num):
                regret = U1[i] - U1node
                p1infoset.RegretSum[i] += regret * p2p
                p1infoset.AvgStrategy[i] += p1p * p1strategy[i]

            return U1node, U2node

        elif player == 2:
            p2infokey = treenode.p2infokey
            p2infoset = self.infodic.get(p2infokey) 

            if p2infokey not in infoset_container:
                infoset_container.append(p2infokey)

            p2strategy = p2infoset.get_strategy()

            U1 = np.zeros(p2infoset.action_num)
            U2 = np.zeros(p2infoset.action_num)
            U1node = 0 
            U2node = 0

            # poacher get caught or has returned home, cannot move further
            if treenode.po_no_move:
                action_history_ = action_history + 's' + '0'
                U1, U2 = self.run_CFR(action_history_, p1p, p2p, 'c')
                return U1, U2
    
            for j, a2 in enumerate(p2infoset.actions):
                action_history_ = action_history + str(a2[0][0]) + str(a2[1])
                U1[j], U2[j] = self.run_CFR(action_history_,p1p, p2p * p2strategy[j], 'c')
                U1node += p2strategy[j] * U1[j]
                U2node += p2strategy[j] * U2[j]

            for i in range(p2infoset.action_num):
                regret = U2[i] - U2node
                p2infoset.RegretSum[i] += regret * p1p
                p2infoset.AvgStrategy[i] +=  p2p * p2strategy[i]

            return U1node, U2node

    def get_p1_sequence(self, action_history):
        '''
        recover player1's action sequences from action_history (comprising of all players actions)
        '''
        idx = 1
        res = ''
        while(idx < len(action_history)):
            res += action_history[idx]
            idx += 4
        return res

    def get_p2_sequence(self, action_history):
        '''
        recover player1's action sequences from action_history (comprising of all players actions)
        '''
        idx = 1
        res = ''
        while(idx < len(action_history)):
            res += action_history[idx+1:idx+3]
            idx += 4
        return res

    def build_trees(self, action_history, p1info, p2info, player, chance_prob, p1u, p2u, mode = 'pa_best_response'):
        '''
        Build the game tree of the game
        infokey is the information a player gets along the way; includes his own action history, the observed footprints
        for patroller, also encodes the info of removing snares and catch the poacher
        for poacher, also encodes getting back home and being caught.
        params:
            action_history: the action history of all players along the way. unique to each node.
            p1info: player1's information set at the current node
            p2info: player2's information set at the current node
            player: which player to act. '1': player1 (patroller here), '2': player2 (poacher here), 
                    and 'c': chance player (environment here: whether catching an animal)
            chance_prob: the accmulated chance prob of getting at the current node
            p1u: the accumulated player1's reward
            p2u: the accumulated player2's reward
            mode: determine the purpose of builging the tree.
                could be 'pa_best_response', 'cfr', or 'po_best_response'. 
                The reason is that different modes need to store different variables at a node.
                If storing them all, the total memory would be too large and the program runs extrememtly slow. 
        return:
            none. All needed data have recorded in the class's self variables.
        '''

        p1sequence = self.get_p1_sequence(action_history)
        p2sequence = self.get_p2_sequence(action_history)
        if self.p1seqdic.get(p1sequence) is None:
            self.p1seqdic[p1sequence] = 1
        if self.p2seqdic.get(p2sequence) is None:
            self.p2seqdic[p2sequence] = 1    

        treenode = self.tree_nodes_dic.get(action_history)
        if treenode is None:
            treenode = tree_node(action_history, p1info, p2info, player, chance_prob)
            self.tree_nodes_dic[action_history] = treenode

        if self.game.end_game:
            treenode.leaf = True
            treenode.u1 = p1u
            treenode.u2 = p2u 
            return 

        if player == 'c':
            if action_history == '': # reset game
                if self.args.po_location is None:
                    chance_actions = [0,1,2,3]
                    probs = [0.25, 0.25, 0.25, 0.25]
                    for idx in range(len(chance_actions)):
                        self.game.reset_game(chance_actions[idx])
                        action_history_ = action_history + str(chance_actions[idx])
                        p2info_ = p2info + str(chance_actions[idx])
                        self.build_trees(action_history_, p1info, p2info_, 1, probs[idx], p1u, p2u, mode)
                    return
                else:
                    self.game.reset_game(self.args.po_location)
                    action_history_ = action_history + str(self.args.po_location)
                    p2info_ = p2info + str(self.args.po_location)
                    self.build_trees(action_history_, p1info, p2info_, 1, 1, p1u, p2u, mode)
                    return

            chance_actions, probs = self.game.get_chance_actions()
            # print(chance_actions)
            # print(self.game.snare_state)
            treenode.actions = chance_actions
            treenode.chance_probs = probs
            for idx in chance_actions:
                pre_game, p1u_, p2u_, pa_remove_snare_cnt = self.game.chancestep(number = chance_actions[idx])
                p1info_ = p1info
                p2info_ = p2info
                if self.game.catch_flag:  # patroller caught poacher
                    p2info_ += '*'
                    p1info_ += '*'
                if self.game.home_flag: # poacher returned home 
                    p2info_ += '&'
                if pa_remove_snare_cnt > 0: # patroller removed snares
                    p1info_ += '#' + str(pa_remove_snare_cnt)
                chance_prob_ = chance_prob * probs[idx]
                action_history_ = action_history + str(chance_actions[idx])
                # action_history_ = action_history + str(idx)
                self.build_trees(action_history_, p1info_, p2info_, 1, chance_prob_, p1u + p1u_, p2u + p2u_, mode)
                self.game.chanceundo(pre_game)
            return 

        elif player == 1:
            if mode == 'po_best_response':
                # for computin poacher best response, need to record correponding internal states to recover DQN patroller or 
                # RS patroller action probabilities
                treenode.pa_state = self.game.get_pa_state()
                treenode.pa_loc = self.game.pa_loc.copy()
                # treenode.po_trace = copy.deepcopy(self.game.po_trace)
                footprints = []
                actions_ = ['up', 'down', 'left', 'right']
                pa_loc = self.game.pa_loc
                for i in range(4,8):
                    if self.game.po_trace[pa_loc[0], pa_loc[1]][i] == 1:
                        footprints.append(actions_[i - 4])
                treenode.RS_footprints = footprints

            
            

            p1infoset = self.infodic.get(p1info)
            if p1infoset is None:
                actions = self.game.get_pa_actions()
                p1infoset = InfoSet(p1info, 1, actions)
                self.infodic[p1info] = p1infoset

            if mode == 'pa_best_response':
                # for computing patroller best response: need to record all nodes in the same information set
                p1infoset.nodes.append(action_history) 

            p1infoset.nodes.append(action_history)
            for i, a1 in enumerate(p1infoset.actions):
                pregame = self.game.patrollerstep(a1)
                p1info_ = p1info + a1[0] 
                p2info_ = p2info
                action_history_ = action_history + str(a1[0])
                self.build_trees(action_history_, p1info_, p2info_, 2, chance_prob, p1u, p2u, mode)
                self.game.patrollerundo(pregame)
            return

        elif player == 2:
            p2infoset = self.infodic.get(p2info)
            if p2infoset is None:
                actions = self.game.get_po_actions()
                p2infoset = InfoSet(p2info, 2, actions)
                self.infodic[p2info] = p2infoset

            if mode == 'po_best_response' or mode == 'cfr':
                # for computing poacher best response, needs to know all nodes in the same information set
                p2infoset.nodes.append(action_history)

            if mode == 'pa_best_response':
                # for computing patroller best response, need to recover heuristice poacher action probabilities
                treenode.po_loc = self.game.po_loc.copy()
                treenode.local_trace = self.game.get_local_pa_trace(treenode.po_loc)
                treenode.local_snare = self.game.get_local_snare(treenode.po_loc)
                treenode.snare_num = self.game.poacher_snare_num

            # poacher get caught or has returned home, cannot move further
            if self.game.home_flag or self.game.catch_flag:
                treenode.po_no_move = True
                pregame, p1footprint, p2footprint = self.game.poacherstep('still', 0)
                p1info_ = p1info + p1footprint        
                action_history_ = action_history + 's' + str(0)
                self.build_trees(action_history_, p1info_, p2info, 'c', chance_prob, p1u, p2u, mode)
                self.game.poacherundo(pregame)
                return 
    
            for j, a2 in enumerate(p2infoset.actions):
                pregame, p1footprint, p2footprint = self.game.poacherstep(a2[0], a2[1])
                p1info_ = p1info + p1footprint
                p2info_ = p2info + a2[0][0] + str(a2[1]) +  p2footprint
                action_history_ = action_history + a2[0][0] + str(a2[1])
                self.build_trees(action_history_, p1info_, p2info_, 'c', chance_prob, p1u, p2u, mode)
                self.game.poacherundo(pregame)

            return 
            

    # compute the best response for poacher
    # utility only stands for the poacher utility
    def compute_po_best_response(self, action_history, player, patrollers = None, patroller_types = None, pa_meta_strategy = None):
        '''
        compute the poacher best response against a given pa strategy: DQN or CFR
        using recursive depth first searching. 
        for more details please refre to the appendix of the orginal paper or 
        Bosansky et al. Double-oracle algorithm for computing an exact nash
        equilibrium in zero-sum extensive-form games. AAMAS’13.
        '''
        treenode = self.tree_nodes_dic.get(action_history)
        if treenode is None:
            print(action_history)
        assert treenode is not None

        if treenode.leaf:
            return treenode.u1, treenode.u2
        
        if player == 'c':
            u = 0

            if action_history == '': # reset game
                if self.args.po_location is None:
                    chance_actions = [0,1,2,3]
                    probs = [0.25, 0.25, 0.25, 0.25]
                    for idx in range(len(chance_actions)):
                        self.game.reset_game(chance_actions[idx])
                        action_history_ = action_history + str(chance_actions[idx])
                        u1, u2 = self.compute_po_best_response(action_history_, 1, 
                                                        patrollers, patroller_types, pa_meta_strategy)
                        u += probs[idx] * u2
                    return -u, u
                else:
                    self.game.reset_game()
                    action_history_ = action_history + str(self.args.po_location)
                    u1, u2 = self.compute_po_best_response(action_history_, 1, 
                                                    patrollers, patroller_types, pa_meta_strategy)
                    return -u2, u2

            chance_actions, probs = treenode.actions, treenode.chance_probs
            for idx in range(len(chance_actions)):
                action_history_ = action_history + str(chance_actions[idx])
                # action_history_ = action_history + str(idx)
                u1, u2 = self.compute_po_best_response(action_history_, 1, 
                                                    patrollers, patroller_types, pa_meta_strategy)
                u += probs[idx] * u2
            return -u, u

        elif player == 1:
            p1infokey = treenode.p1infokey
            p1infoset = self.infodic.get(p1infokey)
            assert p1infoset is not None

            u = 0
            p1strategy = p1infoset.get_average_strategy() ### CFR strategy

            ### a simple heuristic strategy for debug usage
            # p1strategy = np.zeros(len(p1infoset.actions))
            # p1strategy[0] = 1
            # time = (len(action_history) - 1) // 4
            # if time == 0:
            #     p1strategy[3] = 1
            # else:
            #     p1strategy[0] = 1

            if patrollers is not None: ### DQN strategy
                p1strategy = self.get_p1_behaviour_strategy(action_history, patrollers, patroller_types, pa_meta_strategy)
                # print(p1strategy)

            for i, a1 in enumerate(p1infoset.actions):
                if p1strategy[i] < 1e-10: # cut the branches where the patroller does not visit
                    continue
                action_history_ = action_history + a1[0]
                u1, u2 = self.compute_po_best_response(action_history_, 2, patrollers, patroller_types, pa_meta_strategy)
                u += u2 * p1strategy[i]
            
            return -u, u

        elif player == 2:

            if treenode.best_action is not None:
                return -treenode.best_utility, treenode.best_utility

            p2infokey = treenode.p2infokey
            p2infoset = self.infodic.get(p2infokey)
            assert p2infoset is not None
            same_info_nodes = p2infoset.nodes
            assert action_history in same_info_nodes

            actions = p2infoset.actions
            u = [0 for i in range(len(actions))]
            unode = [[0 for j in range(len(actions))] for i in range(len(same_info_nodes))]
            for idx1, po_action in enumerate(actions):
                for idx2, x in enumerate(same_info_nodes):
                    x = self.tree_nodes_dic.get(x)
                    assert x is not None
                    chance_prob = x.chance_prob
                    p1_prob = self.get_p1_probs(x.p1infokey) ### CFR strategy 
                    if patrollers is not None: ### DQN strategy
                        p1_prob = self.get_p1_probs_DQN(same_info_nodes[idx2], patrollers, patroller_types, pa_meta_strategy)
                    prob = chance_prob * p1_prob
                    
                    if x.po_no_move:
                        action_history_ = x.action_history + 's' + str(0) 
                        u1, u2 = self.compute_po_best_response(action_history_, 'c', patrollers, patroller_types, pa_meta_strategy)
                        u[idx1] += u2 * prob
                        unode[idx2][idx1] = u2

                    else:
                        action_history_ = x.action_history + po_action[0][0] + str(po_action[1])
                        u1, u2 = self.compute_po_best_response(action_history_, 'c', patrollers, patroller_types, pa_meta_strategy)
                        u[idx1] += prob * u2
                        unode[idx2][idx1] = u2
            
            best_action_idx = np.argmax(u)
            for idx, x in enumerate(same_info_nodes):
                x_ = self.tree_nodes_dic.get(x)
                x_.best_action = best_action_idx
                x_.best_utility = unode[idx][best_action_idx]
                if x == action_history:
                    ret = x_.best_utility
            return -ret, ret



    def get_p2_behaviour_strategy(self, action_history, poacher_h):
        '''
        transform the poacher's heuristic strategy into a behaviour strategy at the $action_history$ node.
        '''

        if self.p2_bs_dic.get(action_history) is not None:
            return self.p2_bs_dic[action_history]

        treenode = self.tree_nodes_dic.get(action_history)
        assert treenode is not None
        po_loc = treenode.po_loc
        local_trace = treenode.local_trace
        local_snare = treenode.local_snare
        po_initial_loc = self.game.po_initial_loc
        snare_num = treenode.snare_num
        # if snare_num is None:
        #     print(action_history)
        valid_actions = self.infodic.get(treenode.p2infokey).actions
        # a list like: {valid_action0: 0.5, valid_action1: 0.2, ....}
        action_probs = poacher_h.infer_action_probs(po_loc, local_trace, local_snare, 
            po_initial_loc, snare_num, valid_actions)
       
        ## a simple heuristic strategy for debug usage
        # action_probs = np.zeros(len(valid_actions))
        # if len(action_history) < 4:
        #     for idx, a in enumerate(valid_actions):
        #         if a[1] == 1 and a[0] == 'still':
        #             action_probs[idx] = 1
        # else:
        #     if valid_actions[0] != ('still', 0):
        #         print(valid_actions[0])
        #     action_probs[0] = 1
            

        assert np.abs(np.sum(action_probs) - 1) < 1e-10
        self.p2_bs_dic[action_history] = action_probs
        # print('action_prob is:', action_probs)
        return action_probs

    def get_p2_probs_h(self, action_history, poacher_h):
        '''
        get the heuristic poachers's couter probability of reaching the $action_history$ node
        '''
        idx = 1 # the inital move is the chance move, determining the poacher enter point
        p = 1.
        while(idx < len(action_history)):
            action_his = action_history[:idx + 1]
            po_action = action_history[idx+1:idx+3]
            
            # print('processing history: ', action_his, 'idx is: ', idx)
            action_probs = self.get_p2_behaviour_strategy(action_his, poacher_h)

            treenode = self.tree_nodes_dic[action_his]
            actions = self.infodic[treenode.p2infokey].actions

            for id, a in enumerate(actions):
                if a[0][0] == po_action[0] and po_action[1] == a[1]:
                    p *= action_probs[id]

            idx += 4
        return p


    # compute the best response for patroller against a heuristic poacher
    # utility only stands for the poacher utility
    def compute_pa_best_response(self, action_history, player, poacher_h):
        '''
        compute the patroller best response against a heuristic poacher
        '''
        treenode = self.tree_nodes_dic.get(action_history)
        if treenode is None:
            print(action_history)
        assert treenode is not None

        if treenode.leaf:
            return treenode.u1, treenode.u2
        
        if player == 'c':
            u1, u2 = 0,0

            if action_history == '': # reset game
                # if self.args.po_location is None:
                chance_actions = [0,1,2,3]
                probs = [0.25, 0.25, 0.25, 0.25]
                if self.args.po_location is not None:
                    for x in range(4):
                        if x == self.args.po_location:
                            probs[x] = 1
                        else:
                            probs[x] = 0.
                for ca, p in zip(chance_actions, probs):
                    self.args.po_location = ca 
                    self.game.reset_game(ca)
                    action_history_ = action_history + str(ca)
                    u1_, u2_ = self.compute_pa_best_response(action_history_, 1, poacher_h)    
                    print('chance action is {0}, u1_ is {1}'.format(ca, u1_))       
                    u2 += p * u2_
                    u1 += p * u1_

                return u1,u2
                # else:
                #     self.game.reset_game()
                #     action_history_ = action_history + str(self.args.po_location)
                #     u1_, u2_ = self.compute_pa_best_response(action_history_, 1, poacher_h)
                #     return u1_, u2_

            chance_actions, probs = treenode.actions, treenode.chance_probs
            for idx in range(len(chance_actions)):
                action_history_ = action_history + str(chance_actions[idx])
                # action_history_ = action_history + str(idx)
                u1_, u2_ = self.compute_pa_best_response(action_history_, 1, poacher_h)
                u1 += probs[idx] * u1_
                u2 += probs[idx] * u2_
            return u1, u2

        elif player == 2:
            p2infokey = treenode.p2infokey
            p2infoset = self.infodic.get(p2infokey)
            assert p2infoset is not None

            u1, u2 = 0, 0
           
            if treenode.po_no_move:
                action_history_ = treenode.action_history + 's' + str(0) 
                u1_, u2_ = self.compute_pa_best_response(action_history_, 'c', poacher_h)
                return u1_, u2_


            # get p2strategy here 
            p2strategy = self.get_p2_behaviour_strategy(action_history, poacher_h)

            for i, a2 in enumerate(p2infoset.actions):
                if p2strategy[i] < 1e-10: # cut the branches where the poacher does not visit
                    continue
                action_history_ = action_history + a2[0][0] + str(a2[1])
                u1_, u2_ = self.compute_pa_best_response(action_history_, 'c', poacher_h)
                u1 += u1_ * p2strategy[i]
                u2 += u2_ * p2strategy[i]
            return  u1, u2

        elif player == 1:

            # here treenode.best utility stands for the patroller's best utility
            if treenode.best_action is not None:
                return treenode.best_utility, -treenode.best_utility

            p1infokey = treenode.p1infokey
            p1infoset = self.infodic.get(p1infokey)
            assert p1infoset is not None
            same_info_nodes = p1infoset.nodes 
            assert action_history in same_info_nodes

            actions = p1infoset.actions
            u = [0 for i in range(len(actions))] # patroller utility for each action
            unode = [[0 for j in range(len(actions))] for i in range(len(same_info_nodes))]
            for idx1, pa_action in enumerate(actions):
                for idx2, x in enumerate(same_info_nodes):
                    x = self.tree_nodes_dic.get(x)
                    assert x is not None
                    chance_prob = x.chance_prob
                    p2_prob = self.get_p2_probs_h(same_info_nodes[idx2], poacher_h) ### poacher heuristic strategy 
                    prob = chance_prob * p2_prob
                    
                    action_history_ = x.action_history + pa_action[0]
                    u1, u2 = self.compute_pa_best_response(action_history_, 2,poacher_h)
                    u[idx1] += prob * u1
                    unode[idx2][idx1] = u1
        
            best_action_idx = np.argmax(u)
            for idx, x in enumerate(same_info_nodes):
                x_ = self.tree_nodes_dic.get(x)
                x_.best_action = best_action_idx
                x_.best_utility = unode[idx][best_action_idx]
                if x == action_history:
                    ret = x_.best_utility
            return ret, -ret


    def get_p1_probs(self, p1infosets):
        '''
        get the counter probability of reaching a given p1infoset for simple CFR strategies
        '''
        p = 1.
        action_loc = []
        for idx, c in enumerate(p1infosets):
            if c.isalpha():
                action_loc.append(idx)
        for cnt, aidx in enumerate(action_loc):
            action = p1infosets[aidx]
            preinfo = p1infosets[:aidx]
            # print(preinfo)
            infoset = self.infodic.get(preinfo)
            assert infoset is not None
            strategy = infoset.get_average_strategy() ### DQN changes this
            # strategy = np.zeros(len(infoset.actions))
            # strategy[0] = 1
            # # if cnt == 0:
            #     strategy[3] = 1
            # else:
            #     strategy[0] = 1
            for i, a in enumerate(infoset.actions):
                if a[0] == action:
                    p *= strategy[i]
                    break
        return p

    def get_p1_probs_DQN(self, action_history, patrollers, pa_types, pa_meta_strategy):
        '''
        Compute the probability of patrollers reaching a given node (denoted by action_history), for DQN/RS agents
        params:
            action_history: the node
        '''
        idx = 1 # the inital move is the chance move, determining the poacher enter point
        p = 1.
        while(idx < len(action_history)):
            action_his = action_history[:idx]
            pa_action = action_history[idx]
            
            # print('processing history: ', action_his, 'idx is: ', idx)
            action_probs = self.get_p1_behaviour_strategy(action_his, patrollers, pa_types, pa_meta_strategy)

            treenode = self.tree_nodes_dic[action_his]
            actions = self.infodic[treenode.p1infokey].actions

            for id, a in enumerate(actions):
                if a[0] == pa_action:
                    p *= action_probs[id]

            idx += 4
        return p

    def get_p1_behaviour_strategy(self, action_history, patrollers, pa_types, pa_meta_strategy):
        '''
        Compute the behaviour strategy at a given node for a given list of DQNs or RS agents
        params:
            action_history: the key specifying a node
            patrollers: a list of patroller, {DQN, RS}
            pa_types: specify the type of each patroller
            pa_meta_strategy: the mixed strategy among the patrollers
        '''
        action_probs = self.DQN_pa_action_probs.get(action_history)
        if action_probs is not None:
            return action_probs

        treenode = self.tree_nodes_dic.get(action_history)
        actions = self.infodic.get(treenode.p1infokey).actions
        action_probs = [np.zeros(len(actions)) for i in range(len(patrollers))]
        for idx, patroller in enumerate(patrollers):
            # print('idx is {0}, pa type is {1}'.format(idx, pa_types[idx]))
            # print(pa_types[idx])
            if pa_meta_strategy[idx] < 1e-10:
                continue

            if pa_types[idx] == 'DQN':
                pa_state = treenode.pa_state
                pa_action = patroller.infer_action(sess=self.sess, states=[pa_state], policy="greedy")
                in_action = False # DQN might choose invalid actions
                for id, a in enumerate(actions):
                    if a == pa_action:
                        action_probs[idx][id] = 1
                        in_action = True
                if not in_action:
                    action_probs[idx][0] = 1 # invalid action turns to stay still

            if pa_types[idx] == 'RS':
                pa_loc = treenode.pa_loc
                # footprints = []
                # actions_ = ['up', 'down', 'left', 'right']
                # for i in range(4,8):
                #     if treenode.po_trace[pa_loc[0], pa_loc[1]][i] == 1:
                #         footprints.append(actions_[i - 4])
                footprints = treenode.RS_footprints
                previous_pa_action = action_history[-4] if len(action_history) > 1 else 'still'
                pa_action = patroller.get_action_probs(pa_loc, previous_pa_action, footprints)
                for key in pa_action: # pa_action: {'up': 0.5, 'down': 0.5}
                    for id, a in enumerate(actions):
                        if a == key:
                            action_probs[idx][id] = pa_action[key]
        
        ret = np.zeros(len(actions))
        for i, x in enumerate(pa_meta_strategy):
            ret += x * action_probs[i]
        
        # print(ret)
        # time.sleep(1)
        # assert np.abs(np.sum(ret) - 1) < 1e-10
        self.DQN_pa_action_probs[action_history] = ret
        return ret

    def show_pa_best_action(self):
        # for node in self.tree_nodes_dic:
        #     # print(node)
        #     x = self.tree_nodes_dic[node]
        #     if x.player == 1:
        #         print(node)
        #         print(x.best_action) 
        pass