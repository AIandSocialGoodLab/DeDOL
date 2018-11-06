from patroller_rule import Patroller_Rule
from poacher_rule import Poacher
from env import Env
import numpy as np
import argparse
import sys
import multiprocessing
# import tensorflow as tf
# from tqdm import tqdm
import os
import pickle
from maps import generate_map
from DeDOL_util import test_


def test_worker(inputs):
    
    parameter, env, poacher, patroller, args = inputs
    w_ani, w_enter, w_leave = parameter
    print(parameter)
    pa_total_reward = []
    po_total_reward = []

    for _ in range(args.episode_num):
        # reset game
        poacher.reset_snare_num()
        _, _ = env.reset_game()
        pa_episode_reward, po_episode_reward = 0., 0.

        for t in range(args.max_time):
            po_loc = env.po_loc
            if not env.catch_flag and not env.home_flag:
                snare_flag, po_action = poacher.infer_action(loc=po_loc,
                                                             local_trace=env.get_local_pa_trace(po_loc),
                                                             local_snare=env.get_local_snare(po_loc),
                                                             initial_loc=env.po_initial_loc)
            else:
                snare_flag = 0
                po_action = 'still'

            pa_loc = env.pa_loc
            pa_action = patroller.infer_action(pa_loc, env.get_local_po_trace(pa_loc), w_ani, w_enter, w_leave)

            # the env moves on a step
            _, pa_reward, _, po_reward, end_game = \
              env.step(pa_action, po_action, snare_flag, train = False)

            # accmulate the reward
            pa_episode_reward += pa_reward
            po_episode_reward += po_reward

            # the game ends if the end_game condition is true, or the maximum time step is achieved
            if end_game or (t == args.max_time - 1):
                pa_total_reward.append(pa_episode_reward)
                po_total_reward.append(po_episode_reward) 
                break 
           
    print(len(pa_total_reward))
    print(np.mean(pa_total_reward))
    return np.mean(pa_total_reward), np.mean(po_total_reward)


def main():
    argparser = argparse.ArgumentParser(sys.argv[0])
    #########################################################################################
    # Environment
    argparser.add_argument('--row_num', type=int, default=7)
    argparser.add_argument('--column_num', type=int, default=7)
    argparser.add_argument('--ani_den_seed', type=int, default=66)

    # Patroller
    argparser.add_argument('--pa_state_size', type=int, default=20, help="patroller state dimension")
    argparser.add_argument('--pa_num_actions', type=int, default=5, help="still, up, down, left, right")

    # Poacher
    argparser.add_argument('--snare_num', type=int, default=6)
    argparser.add_argument('--po_state_size', type=int, default=22) #yf: add self footprint to poacher
    argparser.add_argument('--po_num_actions', type=int, default=10)
    argparser.add_argument('--po_act_den_w', type=float, default=3.)
    argparser.add_argument('--po_act_enter_w', type=float, default=0.3)
    argparser.add_argument('--po_act_leave_w', type=float, default=-1.0)
    argparser.add_argument('--po_act_temp', type=float, default=5.0, help="softmax temperature")
    argparser.add_argument('--po_home_dir_w', type=float, default=3.0)

    # Training
    argparser.add_argument('--save_path', type=str, default='33random')
    argparser.add_argument('--map_type', type = str, default = 'gauss')
    argparser.add_argument('--naive', type = bool, default = False)

    argparser.add_argument('--po_location', type = int, default = None)
    argparser.add_argument('--zero_sum', type=int, default=1)
    argparser.add_argument('--reward_shaping', type = bool, default = False)

    argparser.add_argument('--pa_save_dir', type=str, default='models_pa/', help='models_pa/')
    argparser.add_argument('--pa_load_dir', type=str, default='None', help='models_pa/model.ckpt')
    argparser.add_argument('--episode_num', type=int, default=8000)
    argparser.add_argument('--max_time', type=int, default=75, help='maximum time step per episode')
    #########################################################################################

    args = argparser.parse_args()

    if args.row_num == 3:
        args.snare_num = 3
        args.max_time = 4
    elif args.row_num == 5:
        args.max_time = 25
    elif args.row_num == 7:
        args.max_time = 75

    if args.pa_load_dir == 'None':
        args.pa_load_dir = None
    if args.pa_save_dir == 'None':
        args.pa_save_dir = None

    print(args.row_num)
    print(args.max_time)
    print(args.map_type)

    # get animal density
    animal_density = generate_map(args)

    env = Env(args, animal_density, cell_length=None, canvas=None, gui=False)
    poacher = Poacher(args, animal_density)
    patroller = Patroller_Rule(args, animal_density)

    parameters = []
    for w_ani in np.arange(1.5, 2, 0.5):
        for w_enter in np.arange(-2, -1.5, 0.5):
            for w_leave in np.arange(5.5, 6, 0.5):
                parameters.append([w_ani, w_enter, w_leave])

    env_list = [env] * len(parameters)
    poacher_list = [poacher] * len(parameters)
    patroller_list = [patroller] * len(parameters)
    args_list = [args for i in parameters]
    inputs = zip(parameters, env_list, poacher_list, patroller_list, args_list)

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    results = pool.map(test_worker, inputs)
    pool.close()


    param_result = []

    for result, param in zip(results, parameters):
        param_result.append((param, result[0]))

    log = open(args.save_path + '_grid_search_result.txt', 'w')
    param_result = sorted(param_result, key=lambda x: x[1], reverse=True)
    for line in param_result[:50]:
        log.write(str(line) + '\n')
        print(line)

    # args.test_episode_num = args.episode_num
    # pa_reward, po_reward, _ = test_(patroller, poacher, env, None, args, poacher_type='PARAM', patroller_type='PARAM')
    # print('pa_reward {0}    po_reward {1}'.format(pa_reward, po_reward))
    # pickle.dump([results, parameters], open(args.save_path + '_grid_search_results.pkl', 'wb'))


if __name__ == '__main__':
    main()
