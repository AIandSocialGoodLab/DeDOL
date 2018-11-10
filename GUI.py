import numpy as np
import argparse
import sys
import tensorflow as tf
import os
from threading import Thread
import time

from patroller_cnn import Patroller_CNN
from patroller_rule import Patroller_Rule as Patroller_h
from poacher_cnn import Poacher
from poacher_rule import Poacher as Poacher_h
from env import Env
from replay_buffer import ReplayBuffer
from DeDOL_util import simulate_payoff, test_
from DeDOL_util import calc_pa_best_response_PER as calc_pa_best_response
from DeDOL_util import extend_payoff
from DeDOL_util import calc_NE,calc_NE_zero
from DeDOL_util import calc_po_best_response_PER as calc_po_best_response
from DeDOL_util import tf_copy
from DeDOL_util import PRDsolver
from patroller_randomsweeping import RandomSweepingPatroller
from maps import Mountainmap, generate_map
from GUI_util import test_gui
from Test_rulepatrol import test_worker



global eps_pa
global eps_po
global pa_start_num
global po_strat_num 
eps_pa, eps_po = [], []
pa_strat_num, po_strat_num = 0, 0



argparser = argparse.ArgumentParser()
########################################################################################
# Test parameters
argparser.add_argument('--pa_load_path', type=str, default='./Results5x5/')
argparser.add_argument('--po_load_path', type=str, default='./Results5x5/')
argparser.add_argument('--load', type=bool, default=False)


# Environment
argparser.add_argument('--row_num', type=int, default=7)
argparser.add_argument('--column_num', type=int, default=7)
argparser.add_argument('--ani_den_seed', type=int, default=66)

# Patroller
#argparser.add_argument('--pa_state_size', type=int, default=11, help="patroller state dimension")
argparser.add_argument('--pa_state_size', type=int, default=20)
argparser.add_argument('--pa_num_actions', type=int, default=5)

# Poacher CNN
argparser.add_argument('--snare_num', type=int, default=6)
#argparser.add_argument('--po_state_size', type=int, default=14, help="poacher state dimension")
argparser.add_argument('--po_state_size', type=int, default=22) #yf: add self footprint to poacher
argparser.add_argument('--po_num_actions', type=int, default=10)

# # Poacher Rule Base
argparser.add_argument('--po_act_den_w', type=float, default=3.)
argparser.add_argument('--po_act_enter_w', type=float, default=0.3)
argparser.add_argument('--po_act_leave_w', type=float, default=-1.0)
argparser.add_argument('--po_act_temp', type=float, default=5.0)
argparser.add_argument('--po_home_dir_w', type=float, default=3.0)

# Training
argparser.add_argument('--map_type', type = str, default = 'random')
argparser.add_argument('--advanced_training', type = bool, default = True)
argparser.add_argument('--save_path', type=str, default='./Results33Parandom/')

argparser.add_argument('--naive', type = bool, default = False)
argparser.add_argument('--pa_episode_num', type=int, default=300000)
argparser.add_argument('--po_episode_num', type=int, default=300000)
argparser.add_argument('--pa_initial_lr', type=float, default=1e-4)
argparser.add_argument('--po_initial_lr', type=float, default=5e-5)
argparser.add_argument('--epi_num_incr', type=int, default=0)
argparser.add_argument('--final_incr_iter', type = int, default = 10)
argparser.add_argument('--pa_replay_buffer_size', type=int, default=200000)
argparser.add_argument('--po_replay_buffer_size', type=int, default=100000)
argparser.add_argument('--test_episode_num', type=int, default=20000)
argparser.add_argument('--iter_num', type=int, default=10) #new added
argparser.add_argument('--po_location', type = int, default = None)
argparser.add_argument('--Delta', type = float, default = 0.0)

argparser.add_argument('--print_every', type=int, default=50)
argparser.add_argument('--zero_sum', type=int, default=1)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--target_update_every', type=int, default=2000)
argparser.add_argument('--reward_gamma', type=float, default=0.95)
argparser.add_argument('--save_every_episode', type=int, default=5000)
argparser.add_argument('--test_every_episode', type=int, default=2000)
argparser.add_argument('--gui_every_episode', type=int, default=500)
argparser.add_argument('--gui_test_num', type = int, default = 20)
argparser.add_argument('--gui', type = int, default = 0)
argparser.add_argument('--mix_every_episode', type=int, default=250) #new added
argparser.add_argument('--epsilon_decrease', type=float, default=0.05) #new added
argparser.add_argument('--reward_shaping', type = bool, default = False)
argparser.add_argument('--PER', type = bool, default = False)
#########################################################################################
args = argparser.parse_args()

if args.row_num == 7:
    args.column_num = 7
    args.max_time = 75
    args.pa_initial_lr = 1e-4
    args.po_initial_lr = 5e-5
    args.pa_replay_buffer_size = 200000
    args.po_replay_buffer_size = 100000
    if args.po_location is not None:
        args.pa_episode_num = 200000
        args.po_episode_num = 200000

elif args.row_num == 5:
    args.column_num = 5
    args.max_time = 25
    args.pa_episode_num = 300000
    args.po_episode_num = 300000
    args.pa_initial_lr = 1e-4
    args.po_initial_lr = 5e-5
    args.pa_replay_buffer_size = 50000
    args.po_replay_buffer_size = 40000
    if args.po_location is not None:
        args.pa_episode_num = 200000


elif args.row_num == 3:
    args.column_num = 3
    args.max_time = 4
    args.snare_num = 3
    args.pa_episode_num = 300000
    args.po_episode_num = 300000
    args.pa_initial_lr = 5e-5
    args.po_initial_lr = 5e-5
    args.pa_replay_buffer_size = 10000
    args.po_replay_buffer_size = 8000
    if args.po_location is not None:
        args.pa_episode_num = 80000
        args.po_episode_num = 80000

if args.save_path and (not os.path.exists(args.save_path)):
    os.makedirs(args.save_path)

paralog = open(args.save_path + 'paralog.txt', 'w')
paralog.write('row_num {0} \n'.format(args.row_num))
paralog.write('snare_num {0} \n'.format(args.snare_num))
paralog.write('max_time {0} \n'.format(args.max_time))
paralog.write('animal density seed {0} \n'.format(args.ani_den_seed))
paralog.write('initial_pa_episode_num {0} \n'.format(args.pa_episode_num))
paralog.write('initial_po_episode_num {0} \n'.format(args.po_episode_num))
paralog.write('pa_initial_lr {0} \n'.format(args.pa_initial_lr))
paralog.write('po_initial_lr {0} \n'.format(args.po_initial_lr))
paralog.write('epi_num_incr {0} \n'.format(args.epi_num_incr))
paralog.write('final_incr_iter {0} \n'.format(args.final_incr_iter))
paralog.write('pa_replay_buffer_size {0} \n'.format(args.pa_replay_buffer_size))
paralog.write('po_replay_buffer_size {0} \n'.format(args.po_replay_buffer_size))
paralog.write('test_episode_num {0} \n'.format(args.test_episode_num))
paralog.write('Delta {0} \n'.format(args.Delta))
paralog.write('po_location {0} \n'.format(str(args.po_location)))
paralog.write('map_type {0} \n'.format(str(args.map_type)))
paralog.write('naive {0} \n'.format(str(args.naive)))
paralog.flush()
paralog.close()

################## for initialization ###########################
global log_file

log_file = open(args.save_path + 'log.txt', 'w')

animal_density = generate_map(args)
env_pa = Env(args, animal_density, cell_length=None, canvas=None, gui=False)
env_po = Env(args, animal_density, cell_length=None, canvas=None, gui=False)

patrollers = [Patroller_CNN(args, 'pa_model' + str(i)) for i in range(5)]
# patrollers[0] = RandomSweepingPatroller(args, args.po_location)
poachers = [Poacher(args, 'po_model' + str(i)) for i in range(5)]
# poachers[0] = Poacher_h(args, animal_density)

# the best response poacher to be trained
target_poacher = Poacher(args, 'target_poacher')
good_poacher = Poacher(args, 'good_poacher')

target_patroller = Patroller_CNN(args, 'target_patroller')
good_patroller = Patroller_CNN(args, 'good_patroller')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# load pretrained models 

args.po_location = None

# load the DQN models you have trained
if args.load:
    poachers[0] = Poacher_h(args,animal_density)
    patrollers[0] = Patroller_h(args, animal_density)

    # poachers[1].load(sess, args.po_load_path)
    # patrollers[1].load(sess, args.pa_load_path)

    test_gui(poachers[0], patrollers[1], sess, args, pa_type = 'DQN', po_type = 'PARAM')

# test the random sweeping patroller and the heuristic poacher
else:
    poacher = Poacher_h(args, animal_density)
    patroller = RandomSweepingPatroller(args)
    test_gui(poacher, patroller, sess, args, pa_type = 'RS', po_type = 'PARAM')

