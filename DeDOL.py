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
from DeDOL_util import simulate_payoff
from DeDOL_util import calc_pa_best_response_PER as calc_pa_best_response
from DeDOL_util import extend_payoff
from DeDOL_util import calc_NE,calc_NE_zero
from DeDOL_util import calc_po_best_response_PER as calc_po_best_response
from DeDOL_util import tf_copy
from DeDOL_util import PRDsolver
from patroller_randomsweeping import RandomSweepingPatroller
from maps import Mountainmap, generate_map
from GUI_util import test_gui


global eps_pa
global eps_po
global pa_start_num
global po_strat_num 
eps_pa, eps_po = [], []
pa_strat_num, po_strat_num = 0, 0



argparser = argparse.ArgumentParser()
########################################################################################
### Environment
argparser.add_argument('--row_num', type=int, default=3)
argparser.add_argument('--column_num', type=int, default=3)
argparser.add_argument('--ani_den_seed', type=int, default=66)

### Patroller
argparser.add_argument('--pa_state_size', type=int, default=20)
argparser.add_argument('--pa_num_actions', type=int, default=5)

### Poacher CNN
argparser.add_argument('--snare_num', type=int, default=6)
argparser.add_argument('--po_state_size', type=int, default=22) # add self footprint to poacher
argparser.add_argument('--po_num_actions', type=int, default=10)

### Poacher Rule Base, parameters set following advice from domain experts
argparser.add_argument('--po_act_den_w', type=float, default=3.)
argparser.add_argument('--po_act_enter_w', type=float, default=0.3)
argparser.add_argument('--po_act_leave_w', type=float, default=-1.0)
argparser.add_argument('--po_act_temp', type=float, default=5.0)
argparser.add_argument('--po_home_dir_w', type=float, default=3.0)

### Training 
argparser.add_argument('--Delta', type = float, default = 0.0, help = 'the exploration rate in the meta-strategy') 
argparser.add_argument('--naive', type = bool, default = False, help = 'whehter using naive PSRO') 
argparser.add_argument('--advanced_training', type = bool, default = True, 
            help = 'whether using dueling double DQN with graident clipping') 
argparser.add_argument('--map_type', type = str, default = 'random')
argparser.add_argument('--po_location', type = int, default = None, help = '0, 1, 2, 3 for local modes; None for global mode')
argparser.add_argument('--save_path', type=str, default='./Results_33_random/')

argparser.add_argument('--pa_episode_num', type=int, default=300000)
argparser.add_argument('--po_episode_num', type=int, default=300000)
argparser.add_argument('--epi_num_incr', type=int, default=0) # no usage now
argparser.add_argument('--final_incr_iter', type = int, default = 10) # no usage now
argparser.add_argument('--pa_replay_buffer_size', type=int, default=200000)
argparser.add_argument('--po_replay_buffer_size', type=int, default=100000)
argparser.add_argument('--test_episode_num', type=int, default=5000)
argparser.add_argument('--iter_num', type=int, default=20, help = 'DO iteraion num') 
argparser.add_argument('--load_path', type=str, default='./Results5x5/')
argparser.add_argument('--load_num', type=int, default=0)
argparser.add_argument('--pa_initial_lr', type=float, default=1e-4)
argparser.add_argument('--po_initial_lr', type=float, default=5e-5)

argparser.add_argument('--br_po_DQN_episode_num', type=int, default=500)
argparser.add_argument('--print_every', type=int, default=50)
argparser.add_argument('--zero_sum', type=int, default=1, help = 'whether to set the game zero-sum')
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--target_update_every', type=int, default=2000)
argparser.add_argument('--reward_gamma', type=float, default=0.95)
argparser.add_argument('--save_every_episode', type=int, default= 200) #10000)
argparser.add_argument('--test_every_episode', type=int, default= 10000)
argparser.add_argument('--gui_every_episode', type=int, default=500)
argparser.add_argument('--gui_test_num', type = int, default = 20)
argparser.add_argument('--gui', type = int, default = 0)
argparser.add_argument('--mix_every_episode', type=int, default=250) 
argparser.add_argument('--epsilon_decrease', type=float, default=0.05, help = 'decrease of the epsilon exploration rate in DQN') 
argparser.add_argument('--PER', type = bool, default = False, help = 'wheter to use prioterized experience replay')
argparser.add_argument('--reward_shaping', type = bool, default = False, help = 'whether to use reward shaping in training')
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
    args.pa_episode_num = 500 #100000
    args.po_episode_num = 500 #100000
    args.pa_initial_lr = 5e-5
    args.po_initial_lr = 5e-5
    args.pa_replay_buffer_size = 200 #10000
    args.po_replay_buffer_size = 200 #8000
    if args.po_location is not None:
        args.pa_episode_num = 200 # 80000
        args.po_episode_num = 200 # 80000

if args.naive:
    args.Delta = 0.0
    args.po_location = None
else:
    pass

if args.po_location is not None:
    args.save_path = './Results_' + str(args.row_num) + str(args.column_num) + '_' \
        + args.map_type + '_mode' + str(args.po_location) + '/'

if args.save_path and (not os.path.exists(args.save_path)):
    os.makedirs(args.save_path)

paralog = open(args.save_path + 'paralog.txt', 'w')
paralog.write('row_num {0} \n'.format(args.row_num))
paralog.write('snare_num {0} \n'.format(args.snare_num))
paralog.write('max_time {0} \n'.format(args.max_time))
paralog.write('animal density seed {0} \n'.format(args.ani_den_seed))
paralog.write('pa_initial_episode_num {0} \n'.format(args.pa_episode_num))
paralog.write('po_initial_episode_num {0} \n'.format(args.po_episode_num))
paralog.write('epi_num_incr {0} \n'.format(args.epi_num_incr))
paralog.write('final_incr_iter {0} \n'.format(args.final_incr_iter))
paralog.write('pa_replay_buffer_size {0} \n'.format(args.pa_replay_buffer_size))
paralog.write('po_replay_buffer_size {0} \n'.format(args.po_replay_buffer_size))
paralog.write('pa_initial_lr {0} \n'.format(args.pa_initial_lr))
paralog.write('po_initial_lr {0} \n'.format(args.po_initial_lr))
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

patrollers = [Patroller_CNN(args, 'pa_model' + str(i)) for i in range(args.iter_num + 1)]
poachers = [Poacher(args, 'po_model' + str(i)) for i in range(args.iter_num + 1)]
pa_type = ['DQN']
po_type = ['DQN']

### initialize poachers needed for training a separate best-response poacher DQN 
br_poacher = Poacher(args, 'br_poacher')
br_target_poacher = Poacher(args, 'br_target_poacher')
br_good_poacher = Poacher(args, 'br_good_poacher')
br_utility = np.zeros(2)

if not args.naive:
    patrollers[0] = RandomSweepingPatroller(args, mode = args.po_location)
    pa_type[0] = 'RS'
if not args.naive:
    poachers[0] = Poacher_h(args, animal_density)
    po_type[0] = 'PARAM'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

### copy ops needed for training a separate best-response poacher DQN 
br_po_copy_ops = tf_copy(br_target_poacher, br_poacher, sess)
br_po_good_copy_ops = tf_copy(br_good_poacher, br_poacher, sess)

pa_payoff = np.zeros((1,1))
po_payoff = np.zeros((1,1))
length= np.zeros((1,1))
  
pa_payoff[0, 0], po_payoff[0, 0], _ = simulate_payoff(patrollers, poachers, 0, 0, env_pa, sess, 
args, pa_type = pa_type[0], po_type = po_type[0])

pa_strategy, po_strategy = np.array([1]), np.array([1])

np.save(file = args.save_path + 'pa_strategy_iter_0', arr = pa_strategy)
np.save(file = args.save_path + 'po_strategy_iter_0', arr = po_strategy)

np.save(file = args.save_path + 'pa_payoff_iter_0', arr = pa_payoff)
np.save(file = args.save_path + 'po_payoff_iter_0', arr = po_payoff)

log_file.write('pa_payoff:\n' + str(pa_payoff) + '\n')
log_file.write('po_payoff:\n' + str(po_payoff) + '\n')

log_file.write('pa_strat:\n' + str(pa_strategy) + '\n')
log_file.write('po_strat:\n' + str(po_strategy) + '\n')


############## starting DO ####################
iteration = 1
pa_pointer, po_pointer = 1, 1 # the pointer counting the number of strategies for pa and po.

while(1):
    time_begin = time.time() 

    pa_payoff, po_payoff, length = extend_payoff(pa_payoff, po_payoff, length, 
            pa_pointer + 1, po_pointer + 1)
    po_type.append('DQN')
    pa_type.append('DQN')

    log_file.flush()

    print('\n' +  'NEW_ITERATION: ' + str(iteration) + '\n')
    log_file.write('\n' + 'NEW_ITERATION: ' + str(iteration) + '\n')

    ### compute the NE utility for both sides
    po_ne_utility = 0
    pa_ne_utility = 0
    for pa_strat in range(pa_pointer):
        for po_strat in range(po_pointer):
            po_ne_utility += pa_strategy[pa_strat] * po_strategy[po_strat] * po_payoff[pa_strat, po_strat] 
            pa_ne_utility += pa_strategy[pa_strat] * po_strategy[po_strat] * pa_payoff[pa_strat, po_strat]

    log_file.write('last_pa_ne_utility:' + str(pa_ne_utility) + '\n')
    log_file.write('last_po_ne_utility:' + str(po_ne_utility) + '\n')
    pre_pa_strategy = pa_strategy
    pre_po_strategy = po_strategy

    ### compute the best response poacher utility
    ### 1. train a best response poacher DQN against the current pa strategy
    calc_po_best_response(br_poacher, br_target_poacher, br_po_copy_ops, br_po_good_copy_ops, patrollers, 
            pa_strategy, pa_type, iteration, sess, env_pa, args, br_utility, 0, train_episode_num=args.br_po_DQN_episode_num)
    br_DQN_utility = br_utility[1]
    
    ### 2. test against the heuristic poacher stored in poachers[0]
    br_heuristic_utility = 0.
    for i in range(pa_pointer):
        _, po_utility, _ = simulate_payoff(patrollers, poachers, i, 0, env_pa, sess, args,
            pa_type=pa_type[i], po_type = po_type[0])
        br_heuristic_utility += po_utility * pa_strategy[i]

    ### choose the better one
    better = 'DQN' if br_DQN_utility >= br_heuristic_utility else 'heuristic'
    br_poacher_utility = max(br_DQN_utility, br_heuristic_utility)
    log_file.write('Iteration {0} poacher best response utility {1} poacher best response type {2} \n'.format(
            iteration, br_poacher_utility, better))
    print('Iteration {0} poacher best response utility {1} poacher best response type {2}'.format(
            iteration, br_poacher_utility, better))


    ### train the best response agent
    ### using threading to accelerate the training
    good_patrollers = []
    good_poachers = []
    final_utility = [0.0, 0.0]
    target_patroller = Patroller_CNN(args, 'target_patroller' + str(iteration))
    good_patroller = Patroller_CNN(args, 'good_patroller' + str(iteration))
    pa_copy_ops = tf_copy(target_patroller, patrollers[pa_pointer], sess)
    pa_good_copy_ops = tf_copy(good_patroller, patrollers[pa_pointer], sess)
    pa_inverse_ops = tf_copy(patrollers[pa_pointer], good_patroller, sess)
    
    target_poacher = Poacher(args, 'target_poacher' + str(iteration))
    good_poacher = Poacher(args, 'good_poacher' + str(iteration))
    po_copy_ops = tf_copy(target_poacher, poachers[po_pointer], sess)
    po_good_copy_ops = tf_copy(good_poacher, poachers[po_pointer], sess)
    po_inverse_ops = tf_copy(poachers[po_pointer], good_poacher, sess)

    funcs = [calc_pa_best_response, calc_po_best_response]
    params = [[patrollers[pa_pointer], target_patroller, pa_copy_ops, pa_good_copy_ops, poachers, 
                po_strategy, po_type, iteration, sess, env_pa, args, final_utility,0], 
              [poachers[po_pointer], target_poacher, po_copy_ops, po_good_copy_ops, patrollers, 
                pa_strategy, pa_type, iteration, sess, env_po, args, final_utility,0]]

    ### if the maximum iteration number is achieved
    if args.iter_num == iteration:
        log_file.write('\n DO reaches terminating iteration {0}'.format(iteration) + '\n')
        log_file.write('Final Pa-payoff: \n' + str(pa_payoff) + '\n')
        log_file.write('Final Po-payoff: \n'+ str(po_payoff) + '\n')
        log_file.write('Final pa_strat:\n' + str(pa_strategy) + '\n')
        log_file.write('Final po_strat:\n'+ str(po_strategy) + '\n')
        log_file.write('Final pa_ne_utility:' + str(pa_ne_utility) + '\n')
        log_file.write('Final po_ne_utility:' + str(po_ne_utility) + '\n')
        log_file.flush()

        threads = []
        for i in range(2):
            process = Thread(target=funcs[i], args=params[i])
            process.start()
            threads.append(process)
        ### We now pause execution on the main thread by 'joining' all of our started threads.
        for process in threads:
            process.join()

        pa_exploit = final_utility[0] - pa_ne_utility
        po_exploit = final_utility[1] - po_ne_utility
        log_file.write('Final pa_best_response_utility:' + str(final_utility[0]) + '\n')
        log_file.write('Final po_best_response_utility:' + str(final_utility[1]) + '\n')
        log_file.write('Final pa exploitibility:' + str(pa_exploit) + '\n')
        log_file.write('Final po exploitibility:' + str(po_exploit) + '\n')
        break
   
    ### not the final iteration
    threads = []

    for i in range(2):
        process = Thread(target=funcs[i], args=params[i])
        process.start()
        threads.append(process)
    for process in threads:
        process.join()

    # calc_pa_best_response(patrollers[pa_pointer], target_patroller, pa_copy_ops, pa_good_copy_ops, poachers, 
    #         po_strategy, iteration, sess, env_pa, args, final_utility,0)

    sess.run(pa_inverse_ops)
    sess.run(po_inverse_ops)

    for pa_strat in range(pa_pointer):
        pa_payoff[pa_strat, po_pointer ],po_payoff[pa_strat, po_pointer], _  = \
            simulate_payoff(patrollers, poachers, pa_strat, po_pointer, env_pa, sess, args,
                pa_type=pa_type[pa_strat], po_type=po_type[po_pointer]) 

    for po_strat in range(po_pointer):
        pa_payoff[pa_pointer, po_strat],po_payoff[pa_pointer, po_strat],_  = \
            simulate_payoff(patrollers, poachers, pa_pointer, po_strat, env_pa, sess, args, 
            pa_type=pa_type[pa_pointer], po_type = po_type[po_strat])

    pa_payoff[pa_pointer, po_pointer],po_payoff[pa_pointer, po_pointer],_  = \
        simulate_payoff(patrollers, poachers, pa_pointer, po_pointer, env_pa, sess, args,
        pa_type=pa_type[pa_pointer], po_type = po_type[po_pointer])
    
    pa_strategy, po_strategy = calc_NE_zero(pa_payoff, po_payoff, args.Delta)
    # pa_strategy, po_strategy = np.ones(iteration + 1) / (iteration + 1), np.ones(iteration + 1) / (iteration + 1)

    params[0][5] = po_strategy
    params[1][5] = pa_strategy 

    po_best_response = final_utility[1]
    pa_best_response = final_utility[0]
    # for pa_strat in range(pa_pointer):
    #     po_best_response += pre_pa_strategy[pa_strat] * po_payoff[pa_strat, po_pointer] 
    # for po_strat in range(po_pointer):
    #     pa_best_response += pre_po_strategy[po_strat] * pa_payoff[pa_pointer, po_strat]

    # eps_po.append(po_best_response - po_ne_utility)
    # eps_pa.append(pa_best_response - pa_ne_utility)

    log_file.write('In DO pa_best_utility:' + str(pa_best_response) + '\n')
    log_file.write('In DO po_best_utility:' + str(po_best_response) + '\n')
    # log_file.write('eps_pa: ' + str(eps_pa) + '\n')
    # log_file.write('eps_po: ' + str(eps_po) + '\n')

    ######### save models for this iteration #############
    save_name = args.save_path + 'iteration_' + str(iteration) + '_pa_model.ckpt'
    patrollers[pa_pointer].save(sess =sess, filename = save_name)
    save_name = args.save_path + 'iteration_' + str(iteration) + '_po_model.ckpt'
    poachers[po_pointer].save(sess =sess, filename = save_name)

    ### save payoff matrix and ne strategies
    np.save(file = args.save_path + 'pa_payoff_iter_' + str(iteration), arr = pa_payoff)
    np.save(file = args.save_path + 'po_payoff_iter_' + str(iteration), arr = po_payoff)
    np.save(file = args.save_path + 'pa_strategy_iter_' + str(iteration), arr = pa_strategy)
    np.save(file = args.save_path + 'po_strategy_iter_' + str(iteration), arr = po_strategy)

    log_file.write('pa_payoff:\n' + str(pa_payoff) + '\n')
    log_file.write('po_payoff:\n' + str(po_payoff) + '\n')
    log_file.write('pa_strategy:\n' + str(pa_strategy) + '\n')
    log_file.write('po_strategy:\n' + str(po_strategy) + '\n')

    iteration += 1
    pa_pointer += 1
    po_pointer += 1

    time_end = time.time()

    log_file.write('Using time: \n' + str(time_end - time_begin) + '\n')
    log_file.flush()

log_file.close()
