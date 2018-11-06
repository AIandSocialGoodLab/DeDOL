from patroller_policy import PatrollerPolicy
from patroller_value_network import PatrollerValue
from poacher_rule import Poacher
from env import Env
import numpy as np
import argparse
import sys
import tensorflow as tf
import os
from maps import generate_map


def test(poacher, patroller, env, sess, args):
    ret_total_reward = []
    ret_length = []
    ret_average_reward = []

    for _ in range(args.test_episode_num):
        # reset game
        poacher.reset_snare_num()
        pa_state, _ = env.reset_game()
        episode_reward =  0.

        for t in range(args.max_time):
            # print('t is', t)
            po_loc = env.po_loc
            if not env.catch_flag and not env.home_flag:
                snare_flag, po_action = poacher.infer_action(loc=po_loc,
                                                             local_trace=env.get_local_pa_trace(po_loc),
                                                             local_snare=env.get_local_snare(po_loc),
                                                             initial_loc=env.po_initial_loc)
            else:
                snare_flag = 0
                po_action = 'still'


            pa_state = np.array([pa_state])
            pa_action = patroller.infer_action(sess=sess, states=pa_state)


            pa_state, pa_reward, po_state, po_reward, end_game = \
              env.step(pa_action, po_action, snare_flag, train = False)

            episode_reward += pa_reward

            if end_game or t == args.max_time - 1:
                # print(episode_reward)
                ret_total_reward.append(episode_reward)
                ret_length.append(t + 1)
                ret_average_reward.append(float(episode_reward) / (t + 1))
                break

    return np.mean(ret_total_reward), np.mean(ret_average_reward), np.mean(ret_length)


def main():
    argparser = argparse.ArgumentParser(sys.argv[0])
    #########################################################################################
    # Environment
    argparser.add_argument('--row_num', type=int, default=7)
    argparser.add_argument('--column_num', type=int, default=7)
    argparser.add_argument('--ani_den_seed', type=int, default=66)
    argparser.add_argument('--zero_sum', type=int, default=1)
    argparser.add_argument('--reward_shaping', type = bool, default = False)
    argparser.add_argument('--po_location', type=int, default=None)
    argparser.add_argument('--map_type', type=str, default='random')

    # Patroller
    argparser.add_argument('--pa_state_size', type=int, default=20, help="patroller state dimension")
    argparser.add_argument('--pa_num_actions', type=int, default=5, help="still, up, down, left, right")

    # Poacher
    argparser.add_argument('--po_state_size', type=int, default=22, help="poacher state dimension")
    argparser.add_argument('--po_num_actions', type=int, default=10, help="still, up, down, left, right x put, not put")
    argparser.add_argument('--snare_num', type=int, default=6)
    argparser.add_argument('--po_act_den_w', type=float, default=3.)
    argparser.add_argument('--po_act_enter_w', type=float, default=0.3)
    argparser.add_argument('--po_act_leave_w', type=float, default=-1.0)
    argparser.add_argument('--po_act_temp', type=float, default=5.0, help="softmax temperature")
    argparser.add_argument('--po_home_dir_w', type=float, default=3.0)

    # Training
    argparser.add_argument('--save_dir', type=str, default='./pg_models_pa/', help='models_pa/')
    argparser.add_argument('--pa_load_dir', type=str, default='None', help='models_pa/model.ckpt')
    argparser.add_argument('--log_file', type=str, default='log_train.txt')
    argparser.add_argument('--episode_num', type=int, default=300000)
    argparser.add_argument('--max_time', type=int, default=75, help='maximum time step per episode')
    argparser.add_argument('--train_every_episode', type=int, default=30)
    argparser.add_argument('--target_update_every_episode', type=int, default=100,
                           help="for state value function")
    argparser.add_argument('--reward_gamma', type=float, default=0.95)
    argparser.add_argument('--initial_lr', type=float, default=1e-5)
    argparser.add_argument('--save_every_episode', type=int, default=10000)
    argparser.add_argument('--test_every_episode', type=int, default=20000)
    argparser.add_argument('--test_episode_num', type=int, default=5000)
    #########################################################################################
    args = argparser.parse_args()
    if args.pa_load_dir == 'None':
        args.pa_load_dir = None
    if args.save_dir == 'None':
        args.save_dir = None

    if args.row_num == 3:
        args.column_num = 3
        args.max_time = 4
        args.snare_num = 3

    if args.row_num == 5:
        args.column_num = 5
        args.max_time = 25

    if args.row_num == 7:
        args.column_num = 7
        args.max_time = 75


    # get animal density
    animal_density = generate_map(args)
    env = Env(args, animal_density, cell_length=None, canvas=None, gui=False)
    poacher = Poacher(args, animal_density)
    patroller_value = PatrollerValue(args, "pa_value_model")
    # target_patroller_value = PatrollerValue(args, 'pa_value_target')
    patroller_policy = PatrollerPolicy(args, "pa_policy")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Load model if necessary
    if args.pa_load_dir:
        patroller_policy.load(sess=sess, filename=args.pa_load_dir)
        print('Load model for patroller from ' + args.pa_load_dir)

    if args.save_dir and (not os.path.exists(args.save_dir)):
        os.mkdir(args.save_dir)

    # Running Initialization
    log = open(args.save_dir + args.log_file, 'w')
    test_log = open(args.save_dir + 'test_log.txt', 'w')

    learning_rate = args.initial_lr
    action_id = {
        'still': 0,
        'up': 1,
        'down': 2,
        'left': 3,
        'right': 4
    }

    copy_ops = []
    # for target_w, model_w in zip(target_patroller_value.variables, patroller_value.variables):
    #     op = target_w.assign(model_w)
    #     copy_ops.append(op)
    # sess.run(copy_ops)
    # print("Update target value network parameter!")

    train_pre_state = []
    train_action = []
    train_reward = []
    train_post_state = []

    for e in range(args.episode_num):
        if e % 500000 == 0:
            learning_rate = max(0.0000001, learning_rate / 2.)

        
        # reset the environment
        poacher.reset_snare_num()
        pa_state, _ = env.reset_game()
        episode_reward = 0.

        for t in range(args.max_time):
            po_loc = env.po_loc
            if not env.catch_flag:
                snare_flag, po_action = poacher.infer_action(loc=po_loc,
                                                             local_trace=env.get_local_pa_trace(po_loc),
                                                             local_snare=env.get_local_snare(po_loc),
                                                             initial_loc=env.po_initial_loc)
            else:
                snare_flag = 0
                po_action = 'still'

            train_pre_state.append(pa_state)
            pa_state = np.array([pa_state])  # Make it 2-D, i.e., [batch_size(1), state_size]
            pa_action = patroller_policy.infer_action(sess=sess, states=pa_state)
            train_action.append(action_id[pa_action])

             # the game moves on a step.
            pa_state, pa_reward, po_state, _, end_game = \
              env.step(pa_action, po_action, snare_flag)

            train_reward.append(pa_reward)

            episode_reward += pa_reward

            # Get new state
            train_post_state.append(pa_state)

            if end_game:
                info = "episode\t%s\tlength\t%s\ttotal_reward\t%s\taverage_reward\t%s" % \
                       (e, t + 1, episode_reward, 1. * episode_reward / (t + 1))
                print(info)
                log.write(info + '\n')
                log.flush()
                break

        # Train
        if e > 0 and e % args.train_every_episode == 0:
            # Fit value function
            post_state_value = patroller_value.get_state_value(sess=sess, states=train_post_state)
            state_value_target = np.array(train_reward) + args.reward_gamma * np.array(post_state_value)
            feed = {
                patroller_value.input_state: train_pre_state,
                patroller_value.state_values_target: state_value_target,
                patroller_value.learning_rate: learning_rate
            }
            sess.run(patroller_value.train_op, feed_dict=feed)

            # Get advantage value
            pre_state_value = patroller_value.get_state_value(sess=sess, states=train_pre_state)
            advantage = np.array(train_reward) + args.reward_gamma * np.array(post_state_value) - \
                        np.array(pre_state_value)

            # Train policy
            feed = {
                patroller_policy.input_state: train_pre_state,
                patroller_policy.actions: train_action,
                patroller_policy.advantage: advantage,
                patroller_policy.learning_rate: learning_rate
            }
            sess.run(patroller_policy.train_op, feed_dict=feed)

            # Clear the training buffer
            train_pre_state = []
            train_action = []
            train_reward = []
            train_post_state = []

        # Test
        if e > 0 and e % args.test_every_episode == 0:
            test_total_reward, test_average_reward, test_length = test(poacher, patroller_policy, env, sess, args)
            info = [test_total_reward, test_average_reward, test_length]
            info = [str(x) for x in info]
            info = '\t'.join(info) + '\n'
            print(info)
            test_log.write(info)
            test_log.flush()

        # Update target
        # if e > 0 and e % args.target_update_every_episode == 0:
        #     sess.run(copy_ops)
        #     print("Update target value network parameter!")

        # Save model
        if e > 0 and e % args.save_every_episode == 0:
            save_name = os.path.join(args.save_dir, str(e), "model.ckpt")
            patroller_policy.save(sess=sess, filename=save_name)
            print('Save model to ' + save_name)

    test_log.close()
    log.close()


if __name__ == '__main__':
    main()
