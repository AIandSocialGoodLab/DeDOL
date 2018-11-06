import numpy as np
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
from cfr_exact_br_util import GameTree


# compute the exact poacher best response in one local mode. 
# due to memory limit, cannot build the original game tree at one time
# build four game trees, in each the poacher has a fixed enter point
# combine the four local tree into the original game tree 
def testmode(mode, patrollers, pa_types, pa_meta_strategy):
    print('mode {0}'.format(mode))
    # print('inside testing mode, pa_types are: ', pa_types)
    args.po_location = mode

    animal_density = generate_map(args)
    print(animal_density)
    env = Env(args, animal_density, cell_length=None, canvas=None, gui=False)

    Tree = GameTree(env, args, sess)
    begin = time.time()
    Tree.build_trees(action_history='', p1info = '1', p2info = '2', player = 'c', chance_prob=1, p1u=0, p2u = 0,
       mode = 'po_best_response')
    end = time.time()
    print(end - begin)
    log.write('building tree using time {0} \n'.format(end - begin))

    begin = time.time()
    a, b = Tree.compute_po_best_response(action_history='', player='c', 
        patrollers=patrollers, patroller_types=pa_types, pa_meta_strategy=pa_meta_strategy)
    print(a,b)
    end = time.time()
    print(end -begin)

    del Tree
    gc.collect()

    return a, b

# compute the exact patroller best response against a heuristic poacher
def pabest():
    args.po_location = None

    animal_density = generate_map(args)
    poacher_h = Poacher_h(args,animal_density)
    print('animal density \n', animal_density)
    env = Env(args, animal_density, cell_length=None, canvas=None, gui=False)

    Tree = GameTree(env, args, None)
    begin = time.time()
    Tree.build_trees(action_history='', p1info = '1', p2info = '2', player = 'c', chance_prob=1, p1u=0, p2u = 0, 
            mode = 'pa_best_response')
    end = time.time()
    print('build tree using time: ', end - begin)
    log.write('building tree using time {0} \n'.format(end - begin))

    begin = time.time()
    a, b = Tree.compute_pa_best_response(action_history='', player='c', poacher_h=poacher_h)
    print('pa best response utility:', a)
    end = time.time()
    print('computation cost time: ', end -begin)

    # Tree.show_pa_best_action()

    log.write('pa best response {0} \n'.format(a))

    del Tree
    gc.collect()

    return a, b

# only for debug usage.
def testmodepa(mode):
    print('mode {0}'.format(mode))
    # print('inside testing mode, pa_types are: ', pa_types)
    args.po_location = mode

    animal_density = generate_map(args)
    poacher_h = Poacher_h(args,animal_density)
    print(animal_density)
    env = Env(args, animal_density, cell_length=None, canvas=None, gui=False)

    Tree = GameTree(env, args, None)
    begin = time.time()
    Tree.build_trees(action_history='', p1info = '1', p2info = '2', player = 'c', chance_prob=1, p1u=0, p2u = 0, 
            mode = 'pa_best_response')
    end = time.time()
    print(end - begin)
    log.write('building tree using time {0} \n'.format(end - begin))

    begin = time.time()
    a, b = Tree.compute_pa_best_response(action_history='', player='c', poacher_h=poacher_h)
    print(a,b)
    end = time.time()
    print(end -begin)

    del Tree
    gc.collect()

    return a, b



argparser = argparse.ArgumentParser()
########################################################################################
### mode: determine the function of the code.
# 'bs': compute exact poacher best response of stored patroller DQN strategies combined from local modes without further globla training 
# 'bs_global': compute exact poacher best response of stored patroller DQN strategies local + global training
# 'pa_bs': compute the exact patroller best response against a parameterized heuristic poacher
# 'cfr': run the cfr algorithm for the 3x3 games 
argparser.add_argument('--mode', type=str, default='bs', help = 'specify the aim of the code') 

# Environment
argparser.add_argument('--row_num', type=int, default=3)
argparser.add_argument('--column_num', type=int, default=3)
argparser.add_argument('--ani_den_seed', type=int, default=66)

# Patroller
#argparser.add_argument('--pa_state_size', type=int, default=11, help="patroller state dimension")
argparser.add_argument('--pa_state_size', type=int, default=20)
argparser.add_argument('--pa_num_actions', type=int, default=5)

# # Poacher Rule Base
argparser.add_argument('--po_act_den_w', type=float, default=3.)
argparser.add_argument('--po_act_enter_w', type=float, default=0.3)
argparser.add_argument('--po_act_leave_w', type=float, default=-1.0)
argparser.add_argument('--po_act_temp', type=float, default=5.0)
argparser.add_argument('--po_home_dir_w', type=float, default=3.0)

# Poacher CNN
argparser.add_argument('--snare_num', type=int, default=3)
#argparser.add_argument('--po_state_size', type=int, default=14, help="poacher state dimension")
argparser.add_argument('--po_state_size', type=int, default=22) #yf: add self footprint to poacher
argparser.add_argument('--po_num_actions', type=int, default=10)
argparser.add_argument('--advanced_training', type=bool, default=True)


argparser.add_argument('--load_path', type=str, default='./Results55random_mode') # local mode model load dir
argparser.add_argument('--load_path2', type=str, default='./Results55random_collecting/') # global mode model load dir
argparser.add_argument('--map_type', type = str, default = 'random')
argparser.add_argument('--save_path', type=str, default='./Results_cfrrandom/')
argparser.add_argument('--load_num', type=int, default=11) # local mode load number
argparser.add_argument('--iter_num', type=int, default=14) # global mode load number


argparser.add_argument('--cfrlog', type=str, default='cfrlog.txt')
argparser.add_argument('--cfriteration', type=int, default=4000)


argparser.add_argument('--max_time', type=int, default=4)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--po_location', type = int, default = None)
argparser.add_argument('--naive', type=bool, default=False)
argparser.add_argument('--cfr_save', type=str, default='cfrsave.txt')
argparser.add_argument('--cfr_test_every_episode', type=int, default=100)
argparser.add_argument('--zero_sum', type=int, default=1)
#########################################################################################
args = argparser.parse_args()

if args.save_path and (not os.path.exists(args.save_path)):
    os.makedirs(args.save_path)

log = open(args.save_path + 'log.txt', 'w')

if args.mode == 'pa_bs':
    print('computing ')
    pabest()
    exit()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

args.po_location = None

### only for debug usage
# for i in range(1):
#     # print(testmode(i, None, None, None))
#     print(testmodepa(i))

# build the environment
animal_density = generate_map(args)
env = Env(args, animal_density, cell_length=None, canvas=None, gui=False)

# build the game tree
Tree = GameTree(env, args, sess)
begin = time.time()
Tree.build_trees(action_history='', p1info = '1', p2info = '2', player = 'c', chance_prob=1, p1u=0, p2u = 0)
end = time.time()
print(end - begin)
log.write('building tree using time {0} \n'.format(end - begin))
# print('the game tree has {0} nodes'.format(len(Tree.tree_nodes_dic) * 4))
# print('the game tree has {0} p1 sequences'.format(len(Tree.p1seqdic)))
# print('the game tree has {0} p2 sequences'.format(len(Tree.p2seqdic)))
# print('the game tree has {0} information sets'.format(len(Tree.infodic)))


if args.mode == 'bs_global':
    load_num = args.load_num
    pa_number = 4 * load_num + 4 # 0: RS mode 0; 1 -- 4*load_num: DQNs; 4*load_num + 1 -- 4 * load_num + 3: RS mode 1,2,3
    patrollers = [Patroller(args, 'pa_model' + str(i)) for i in range(load_num * 4 + 5 + args.iter_num)]
    patrollers[0] = RandomSweepingPatroller(args, mode = 0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # print('reaches here')

    # load pretrained models 
    if args.load_path is not None and args.load_num > 0:
        for i in range(1, 4):
            load_path = args.load_path + str(i) + '/'  ##### load_path needs to be checked 
            for j in range(1, load_num + 1):
                print('{0} load over'.format(load_path))
                patrollers[j].load(sess, load_path + 'iteration_{0}_pa_model.ckpt'.format(j))
                pa_copy_op = tf_copy(patrollers[i * load_num + j], patrollers[j], sess)
                sess.run(pa_copy_op)
        load_path = args.load_path + '0/'
        for j in range(1, load_num + 1):
            print('{0} load over'.format(load_path))
            patrollers[j].load(sess, load_path + 'iteration_{0}_pa_model.ckpt'.format(j))

    print('-------------local model load succeed--------------')
    print('-------------local model load succeed--------------')
    print('-------------local model load succeed--------------')

    # add the rest mode patroller random sweeping models
    for i in range(3):
        patrollers[1 + 4 * load_num + i] = RandomSweepingPatroller(args, i + 1)

    # load more global training models
    for more_number in range(args.iter_num):
        print('more_number is', more_number)
        pa_types = ['RS']
        pa_types += ['DQN' for i in range(4 * load_num)]
        pa_types += ['RS' for i in range(3)]
        for idx in range(1, more_number + 1):
            patrollers[idx - 1 + pa_number].load(sess, args.load_path2 + 'iteration_{0}_pa_model.ckpt'.format(idx))

        print('-------------global trained model load succeed--------------')
        print('-------------global trained model load succeed--------------')
        print('-------------global trained model load succeed--------------')
        pa_meta_strategy = np.load(args.load_path2 + 'pa_strategy_iter_{0}.npy'.format(more_number))
        
        for i in range(more_number):
            pa_types.append('DQN')
        
        assert pa_number + more_number == len(pa_meta_strategy)
        assert pa_number + more_number == len(pa_types)

        # print('at the begining, pa_types are: ', pa_types)

        u = 0.0
        for i in range(4):
            a, b = testmode(i, patrollers[:pa_number + more_number], pa_types,  pa_meta_strategy)
            gc.collect()
            u += b

        u /= 4.
        print('patroller numbers {0}  po best utility {1} \n'.format(pa_number + more_number, u))
        log.write('patroller numbers {0}  po best utility {1} \n'.format(pa_number + more_number, u))
        log.flush()


elif args.mode == 'bs':
    patrollers = [Patroller(args, 'pa_model' + str(i)) for i in range(20)]
    patrollers[0] = RandomSweepingPatroller(args, None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for number in range(1, args.iter_num):
        for idx in range(1, number):
            patrollers[idx].load(sess, args.load_path + 'iteration_{0}_pa_model.ckpt'.format(idx))

        print('-------------local model load succeed--------------')
        print('-------------local model load succeed--------------')
        print('-------------local model load succeed--------------')
        pa_meta_strategy = np.load(args.load_path + 'pa_strategy_iter_{0}.npy'.format(number - 1))
        pa_types = ['RS']
        for i in range(1, number):
            pa_types.append('DQN')
        
        u = 0.0
        for i in range(4):
            a, b = testmode(i, patrollers[:number], pa_types,  pa_meta_strategy)
            gc.collect()
            u += b

        u /= 4.
        log.write('patroller numbers {0}  po best utility {1} \n'.format(number, u))
        log.flush()

elif args.mode == 'cfr':
    args.po_location = None
    animal_density = generate_map(args)
    env = Env(args, animal_density, None, None, None)
    Tree = GameTree(env, args, None)
    Tree.build_trees(action_history='', p1info = '1', p2info = '2', player = 'c', chance_prob=1, p1u=0, p2u = 0, mode = 'cfr')
    print('build tree over!!!!!!!!!!!!!!!!!!!!')
    # print('total game states {0}'.format(len(Tree.tree_nodes_dic)))
    # print('total information sets {0}'.format(len(Tree.infodic)))
    # log.write('total game states {0} \n'.format(len(Tree.tree_nodes_dic)))
    # log.write('total information sets {0} \n'.format(len(Tree.infodic)))
    log.flush()
    Tree.solve_CFR(args.cfriteration)
