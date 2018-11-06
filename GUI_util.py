# this tests the poacher dqn with gui


from patroller_cnn import Patroller_CNN
from patroller_rule import Patroller_Rule
from poacher_rule import Poacher
from env import Env
import sys
import time
import numpy as np
import argparse
import tensorflow as tf
from tkinter import *
from tkinter import Tk, Canvas
from maps import Mountainmap, generate_map
from patroller_randomsweeping import RandomSweepingPatroller
import copy


def test_gui(poacher, patroller, sess, args, pa_type, po_type):
    """
    doc
    """
    #########################################################################################
    global e
    global t
    global episode_reward
    global pa_total_reward, po_total_reward, game_len, pa_episode_reward, po_episode_reward
    global pa_state, po_state, pa_action

    pa_total_reward, po_total_reward, game_len = [], [] ,[]
    pa_action = 'still'

    master = Tk()
    cell_length = 80
    canvas_width = args.column_num * cell_length
    canvas_height = args.row_num * cell_length
    canvas = Canvas(master=master, width=canvas_width, height=canvas_height)
    canvas.grid()

    # animal_density = Mountainmap(args.row_num, args.column_num)
    # np.random.seed(args.ani_den_seed)
    # animal_density = np.random.uniform(low=0.2, high=1., size=[args.row_num, args.column_num])
    animal_density = generate_map(args)
    TestEnv = Env(args, animal_density, cell_length=cell_length, canvas=canvas, gui=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    e = 0
    t = 0

    def run_step():
        global e, t, pa_total_reward, po_total_reward, game_len, pa_episode_reward, po_episode_reward
        global pa_state, po_state, pa_action

        if t == 0:
            print('reset')
            poacher.reset_snare_num()
            pa_state, po_state = TestEnv.reset_game()
            pa_episode_reward, po_episode_reward = 0., 0.

        # poacher take actions. Doing so is due to the different API provided by the DQN and heuristic agent
        if po_type == 'DQN':
            # the poacher can take actions only if he is not caught yet/has not returned home
            if not TestEnv.catch_flag and not TestEnv.home_flag: 
                po_state = np.array([po_state])
                snare_flag, po_action = poacher.infer_action(sess=sess, states=po_state, policy="greedy")
            else:
                snare_flag = 0
                po_action = 'still' 
        elif po_type == 'PARAM':
            po_loc = TestEnv.po_loc
            if not TestEnv.catch_flag and not TestEnv.home_flag:
                snare_flag, po_action = poacher.infer_action(loc=po_loc,
                                                        local_trace=TestEnv.get_local_pa_trace(po_loc),
                                                        local_snare=TestEnv.get_local_snare(po_loc),
                                                        initial_loc=TestEnv.po_initial_loc)
            else:
                snare_flag = 0
                po_action = 'still'

        # patroller take actions
        if pa_type == 'DQN':
            pa_state = np.array([pa_state])
            pa_action = patroller.infer_action(sess=sess, states=pa_state, policy="greedy")
        elif pa_type == 'PARAM':
            pa_loc = TestEnv.pa_loc
            pa_action = patroller.infer_action(pa_loc, TestEnv.get_local_po_trace(pa_loc), 1.5, -2.0, 8.0)
        elif pa_type == 'RS':
            pa_loc = TestEnv.pa_loc
            footprints = []
            actions = ['up', 'down', 'left', 'right']
            for i in range(4,8):
                if TestEnv.po_trace[pa_loc[0], pa_loc[1]][i] == 1:
                    footprints.append(actions[i - 4])
            pa_action = patroller.infer_action(pa_loc, pa_action, footprints)

        # the TestEnv moves on a step
        pa_state, pa_reward, po_state, po_reward, end_game = \
            TestEnv.step(pa_action, po_action, snare_flag, train = False)

        # print('poacher snare:', snare_flag)
        # # time.sleep(1)

        # accmulate the reward
        pa_episode_reward += pa_reward
        po_episode_reward += po_reward

        # the game ends if the end_game condition is true, or the maximum time step is achieved
        if end_game or (t == args.max_time - 1):
            info = "episode\t%s\tlength\t%s\tpatroller_total_reward\t%s\tpoacher_total_reward\t%s" % \
                   (e, t + 1, pa_episode_reward, po_episode_reward)
            print(info)
            pa_total_reward.append(pa_episode_reward)
            game_len.append(t + 1)
            po_total_reward.append(po_episode_reward) 
            t = 0
            e += 1
            if e == args.gui_test_num:
                master.destroy()
                return 
        else:
            t += 1

        master.after(500, run_step)
   

    run_step()
    master.mainloop()
    
    #print(np.mean(ret_total_reward), np.mean(ret_average_reward), np.mean(ret_length))
    return np.mean(pa_total_reward), np.mean(po_total_reward), np.mean(game_len)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    ########################################################################################
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
    argparser.add_argument('--po_state_size', type=int, default=22) # add self footprint to poacher
    argparser.add_argument('--po_num_actions', type=int, default=10)

    # # Poacher Rule Base, parameters set following advice from domain experts
    argparser.add_argument('--po_act_den_w', type=float, default=3.)
    argparser.add_argument('--po_act_enter_w', type=float, default=0.3)
    argparser.add_argument('--po_act_leave_w', type=float, default=-1.0)
    argparser.add_argument('--po_act_temp', type=float, default=5.0)
    argparser.add_argument('--po_home_dir_w', type=float, default=3.0)

    # Training 
    argparser.add_argument('--naive', type = bool, default = False, help = 'whehter using naive PSRO') 
    argparser.add_argument('--advanced_training', type = bool, default = True, 
                help = 'whether using dueling double DQN with graident clipping') 
    argparser.add_argument('--map_type', type = str, default = 'random')
    argparser.add_argument('--po_location', type = int, default = None, help = '0, 1, 2, 3 for local modes; None for global mode')
    argparser.add_argument('--max_time', type = int, default = 75)
    argparser.add_argument('--zero_sum', type=int, default=1, help = 'whether to set the game zero-sum')
    argparser.add_argument('--gui_test_num', type = int, default = 20)
    argparser.add_argument('--batch_size', type = int, default = 64)
    #########################################################################################
    args = argparser.parse_args()

    animal_density = generate_map(args)
    patroller = Patroller_CNN(args, 'model')
    poacher = Poacher(args, animal_density)
    patroller = RandomSweepingPatroller(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    test_gui(poacher, patroller, sess, args, pa_type = 'RS', po_type = 'PARAM')