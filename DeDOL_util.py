from patroller_cnn import Patroller_CNN
from poacher_cnn import Poacher
from env import Env
from replay_buffer import ReplayBuffer, PERMemory
import numpy as np
import argparse
import sys
import tensorflow as tf
import os
from GUI_util import test_gui
import warnings
import nashpy as nash
# from cvxopt import matrix, solvers
import time
warnings.simplefilter('error', RuntimeWarning)


def tf_copy(dst_net, src_net, sess):
    copy_ops = []
    for dst_w, src_w in zip(dst_net.variables, src_net.variables):
        op = dst_w.assign(src_w)
        copy_ops.append(op)
    return copy_ops

### extend the payoff matrix for computing NE.
def extend_payoff(pa_payoff, po_payoff, length, pa_num, po_num):
    pa_payoff_temp = pa_payoff
    po_payoff_temp = po_payoff
    ori_row_num, ori_col_num = pa_payoff.shape
    length_temp = length
    pa_payoff = np.zeros((pa_num, po_num))
    po_payoff = np.zeros((pa_num, po_num))
    length = np.zeros((pa_num, po_num))
    pa_payoff[0:ori_row_num, 0:ori_col_num] = pa_payoff_temp
    po_payoff[0:ori_row_num, 0:ori_col_num] = po_payoff_temp
    length[0:ori_row_num, 0:ori_col_num] = length_temp
    return pa_payoff, po_payoff, length

### given a poacher and patroller, doing simulations to compute the payoff for them
def simulate_payoff(patrollers, poachers, pa_index, po_index,  env, sess, args, pa_type = None, po_type = None, 
    po_location = None):
    '''
    po_location: None if global mode, else determing the poacher local mode.
    '''
    
    print('calc_payoff_patroller: ' + str(pa_index) + '  poacher: ' + str(po_index))
    patroller = patrollers[pa_index]
    poacher = poachers[po_index]

 
    pa_tot_return, po_tot_return, length = \
         test_(patroller, poacher, env, sess,  args, poacher_type=po_type, patroller_type=pa_type, 
            pa_idx=pa_index, po_idx=po_index, po_location = po_location)
    #    test(patroller, poacher, env, sess,  args, poacher_h=po_h, patroller_h=pa_h,pa_idx=pa_index, po_idx=po_index)

    print(pa_tot_return, po_tot_return, length)
    return pa_tot_return, po_tot_return, length

### not used 
def PRDsolver(A, B, x, y, _):
    A = np.array(A)
    B = np.array(B)
    num = A.shape[0]
    
    uniform = np.ones(num) / num
    delta = 0.05
    lamb = 0.05

    # if add a strategy
    if _ == 0:
        pa_strategy = np.zeros(num)
        pa_strategy[:-1] = x * (1. - 1./num)
        pa_strategy[-1] = 1. / num
        pa_strategy = (1 - lamb) * pa_strategy + lamb * uniform
        po_strategy = np.zeros(num)
        po_strategy[:-1] = y * (1. - 1./num)
        po_strategy[-1] = 1. / num
        po_strategy = (1 - lamb) * po_strategy + lamb * uniform
        print(pa_strategy)
        print(po_strategy)
        return pa_strategy, po_strategy

    x = np.array(x).reshape(1, num)
    y = np.array(y).reshape(num,1)

    dx = x * (A.dot(y) - x.dot(A.dot(y))).reshape((x.shape[0], x.shape[1]))
    x = x + delta * dx
    x = x / np.sum(x)
    print(x)
    pa_strategy = x[0,:] * (1 - lamb) + lamb * uniform
    
    dy = y * (x.dot(B) - x.dot(B.dot(y))).reshape((y.shape[0], y.shape[1]))
    y = y + delta * dy
    y = y / np.sum(y)
    print(y)
    po_strategy = y[:,0] * (1 - lamb) + lamb * uniform
    print(pa_strategy)
    print(po_strategy)
    return pa_strategy, po_strategy

### use Nashpy packege to compute nash strategy. a bit slow then the payoff matrices are huge  
def calc_NE(pa_payoff, po_payoff):
    game = nash.Game(pa_payoff, po_payoff)
    for i in range(pa_payoff.shape[0]):
        try:
            eq = game.lemke_howson(i)
        except RuntimeWarning:
            pass
        else:
            break
    return list(eq)[0], list(eq)[1]

### use CVXOPT to compute nash strategy (mixed with exploration). much faster.
def calc_NE_zero(pa_payoff, po_payoff, delta):
    matrixs = [pa_payoff, po_payoff.T] 
    strategies = [None, None]
    for i in range(2):
        A = matrixs[i]
        num_vars = A.shape[0]
        # minimize matrix c
        c = [-1] + [0 for i in range(num_vars)]
        c = np.array(c, dtype="float")
        c = matrix(c)
        # constraints G*x <= h
        G = np.matrix(A, dtype="float").T # reformat each variable is in a row
        G *= -1 # minimization constraint
        G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
        new_col = [1 for i in range(A.shape[1])] + [0 for i in range(num_vars)]
        G = np.insert(G, 0, new_col, axis=1) # insert utility column
        G = matrix(G)
        h = ([0 for i in range(A.shape[1])] + 
            [0 for i in range(num_vars)])
        h = np.array(h, dtype="float")
        h = matrix(h)
        # contraints Ax = b
        A = [0] + [1 for i in range(num_vars)]
        A = np.matrix(A, dtype="float")
        A = matrix(A)
        b = np.matrix(1, dtype="float")
        b = matrix(b)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver = 'glpk')
        strategies[i] = np.array(sol['x'][1:]).reshape(num_vars)

    num = pa_payoff.shape[0]
    uniform = np.ones(num) / num
    strategies[0] = strategies[0] * (1 - delta) + delta * uniform
    num = pa_payoff.shape[1]
    uniform = np.ones(num) / num
    strategies[1] = strategies[1] * (1 - delta) + delta * uniform
    return strategies[0], strategies[1]  # patroller strat, poacher strat


### give a poacher and patroller, run simulations to gain their rewards
def test_(patroller, poacher, env, sess, args, iteration = None, grand_episode = None, \
      poacher_type = None, patroller_type = None, pa_idx = None, po_idx = None, po_location = None):

    '''
    doc
    '''
    pa_total_reward, po_total_reward = [],[]
    ret_length = []

    if pa_idx is not None:
        log_path = args.save_path + '_pa_' + str(pa_idx) + '_po_' + str(po_idx) + '.txt' 
        test_detail_log = open(log_path, 'w')

    for e in range(args.test_episode_num):

        ### reset the environment
        poacher.reset_snare_num()
        pa_state, po_state = env.reset_game(po_location)
        pa_episode_reward, po_episode_reward = 0., 0.
        pa_action = 'still'
    
        for t in range(args.max_time):

            ### poacher take actions. Doing so is due to the different API provided by the DQN and heuristic agent
            if poacher_type == 'DQN':
                ### the poacher can take actions only if he is not caught yet/has not returned home
                if not env.catch_flag and not env.home_flag: 
                    po_state = np.array([po_state])
                    snare_flag, po_action = poacher.infer_action(sess=sess, states=po_state, policy="greedy")
                else:
                    snare_flag = 0
                    po_action = 'still' 
            elif poacher_type == 'PARAM':
                po_loc = env.po_loc
                if not env.catch_flag and not env.home_flag:
                    snare_flag, po_action = poacher.infer_action(loc=po_loc,
                                                            local_trace=env.get_local_pa_trace(po_loc),
                                                            local_snare=env.get_local_snare(po_loc),
                                                            initial_loc=env.po_initial_loc)
                else:
                    snare_flag = 0
                    po_action = 'still'

            ### patroller take actions
            if patroller_type == 'DQN':
                pa_state = np.array([pa_state])
                pa_action = patroller.infer_action(sess=sess, states=pa_state, policy="greedy")
            elif patroller_type == 'PARAM':
                pa_loc = env.pa_loc
                pa_action = patroller.infer_action(pa_loc, env.get_local_po_trace(pa_loc), 1.5, -2, 5.5)
            elif patroller_type == 'RS':
                pa_loc = env.pa_loc
                footprints = []
                actions = ['up', 'down', 'left', 'right']
                for i in range(4,8):
                    if env.po_trace[pa_loc[0], pa_loc[1]][i] == 1:
                        footprints.append(actions[i - 4])
                pa_action = patroller.infer_action(pa_loc, pa_action, footprints)

            ### the env moves on a step
            pa_state, pa_reward, po_state, po_reward, end_game = \
              env.step(pa_action, po_action, snare_flag, train = False)

            ### accmulate the reward
            pa_episode_reward += pa_reward
            po_episode_reward += po_reward

            ### the game ends if the end_game condition is true, or the maximum time step is achieved
            if end_game or (t == args.max_time - 1):
                pa_total_reward.append(pa_episode_reward)
                ret_length.append(t + 1)
                po_total_reward.append(po_episode_reward) 
                if pa_idx is not None:
                    info = '{0}  {1}  {2}  {3}\n'.format(e, pa_episode_reward, po_episode_reward, t+1)
                    test_detail_log.write(info)
                break 

    if pa_idx is not None:
        test_detail_log.flush()
        test_detail_log.close()

    # print(len(pa_total_reward))
    return np.mean(pa_total_reward), np.mean(po_total_reward), np.mean(ret_length)



### adding the PER training
### 'Prioritized Experience Replay', https://arxiv.org/abs/1511.05952 
def calc_po_best_response_PER(poacher, target_poacher, po_copy_op, po_good_copy_op, patrollers, pa_s, pa_type,
       iteration, sess, env, args, final_utility, starting_e, train_episode_num = None):
    '''
    Given a list of patrollers, and their types (DQN, PARAM, RS)
    Train a DQN poacher as the approximating best response
    Args:
        poacher: DQN poacher
        target_poacher: target DQN poacher
        po_copy_op: tensorflow copy opertaions, copy the weights from DQN to the target DQN
        po_good_copy_op: tensorflow copy operations,  save the trained ever-best poacher DQN
        patrollers: a list of patrollers
        pa_s: the patroller mixed startegy among the list of patrollers
        pa_type: a list specifying the type of each patroller, {'DQN', 'PARAM', 'RS'}
        iteration: the current DO iterations
        sess: tensorflow sess
        env: the game environment
        args: some args 
        final_utility: record the best response utility 
        starting_e: the starting of the training epoch
    Return:
        Nothing explictly returned  due to multithreading.
        The best response utility is returned in $final_utility$
        The best response DQN is copied through the $po_good_copy_op$
    '''

    #print('FIND_poacher_best_response iteration: ' + str(iteration))
    if train_episode_num is None:
        train_episode_num = args.po_episode_num

    decrease_time = 1.0 / args.epsilon_decrease
    epsilon_decrease_every = train_episode_num // decrease_time

    if not args.PER:
        replay_buffer = ReplayBuffer(args, args.po_replay_buffer_size)
    else:
        replay_buffer = PERMemory(args)
    pa_strategy = pa_s
    best_utility = -10000.0
    test_utility = []

    if starting_e == 0:
        log = open(args.save_path + 'po_log_train_iter_' + str(iteration) + '.dat', 'w')
        test_log = open(args.save_path + 'po_log_test_iter_' + str(iteration) +  '.dat', 'w')
    else:
        log = open(args.save_path + 'po_log_train_iter_' + str(iteration) + '.dat', 'a')
        test_log = open(args.save_path + 'po_log_test_iter_' + str(iteration) +  '.dat', 'a')

    epsilon = 1.0
    learning_rate = args.po_initial_lr
    global_step = 0
    action_id = {
        ('still', 0): 0,
        ('up', 0): 1,
        ('down', 0): 2,
        ('left', 0): 3,
        ('right', 0): 4,
        ('still', 1): 5,
        ('up', 1): 6,
        ('down', 1): 7,
        ('left', 1): 8,
        ('right', 1): 9
    }

    sess.run(po_copy_op)

    for e in range(starting_e, starting_e + train_episode_num):
        if e > 0 and e % epsilon_decrease_every == 0:
            epsilon = max(0.1, epsilon - args.epsilon_decrease)
        if e % args.mix_every_episode == 0 or e == starting_e:
            pa_chosen_strat = np.argmax(np.random.multinomial(1, pa_strategy))
            patroller = patrollers[pa_chosen_strat]
            type = pa_type[pa_chosen_strat]
        # if args.gui == 1 and e > 0 and e % args.gui_every_episode == 0:
        #     test_gui(poacher, patroller, sess, args, pah = heurestic_flag, poh = False)

        ### reset the environment
        poacher.reset_snare_num()
        pa_state, po_state = env.reset_game()
        episode_reward = 0.0
        pa_action = 'still'

        for t in range(args.max_time):
            global_step += 1
            transition = []

            ### transition adds current state
            transition.append(po_state)

            ### poacher chooses an action, if it has not been caught/returned home
            if not env.catch_flag and not env.home_flag: 
                po_state = np.array([po_state])
                snare_flag, po_action = poacher.infer_action(sess=sess, states=po_state, policy="epsilon_greedy",
                                                            epsilon=epsilon)
            else:
                snare_flag = False
                po_action = 'still'
            
            transition.append(action_id[(po_action, snare_flag)])

            ### patroller chooses an action
            ### Note that heuristic and DQN agent has different APIs
            if type == 'DQN':
                pa_state = np.array([pa_state])  # Make it 2-D, i.e., [batch_size(1), state_size]
                pa_action = patroller.infer_action(sess=sess, states=pa_state, policy="greedy")
            elif type == 'PARAM':
                pa_loc = env.pa_loc
                pa_action = patroller.infer_action(pa_loc, env.get_local_po_trace(pa_loc), 1.5, -2.0, 8.0)
            elif type == 'RS':
                pa_loc = env.pa_loc
                footprints = []
                actions = ['up', 'down', 'left', 'right']
                for i in range(4,8):
                    if env.po_trace[pa_loc[0], pa_loc[1]][i] == 1:
                        footprints.append(actions[i - 4])
                pa_action = patroller.infer_action(pa_loc, pa_action, footprints)


            pa_state, _, po_state, po_reward, end_game = \
              env.step(pa_action, po_action, snare_flag)

           
            ### transition adds reward, and the new state
            transition.append(po_reward)
            transition.append(po_state)
           
            episode_reward += po_reward
            
            ### Add transition to replay buffer
            replay_buffer.add_transition(transition)

            ### Start training
            ### Sample a minibatch
            if replay_buffer.size >= args.batch_size:

                if not args.PER:
                    train_state, train_action, train_reward, train_new_state = \
                        replay_buffer.sample_batch(args.batch_size)
                else:
                    train_state, train_action, train_reward,train_new_state, \
                      idx_batch, weight_batch = replay_buffer.sample_batch(args.batch_size)

                ### Double DQN get target
                max_index = poacher.get_max_q_index(sess=sess, states=train_new_state)
                max_q = target_poacher.get_q_by_index(sess=sess, states=train_new_state, index=max_index)

                q_target = train_reward + args.reward_gamma * max_q

                if args.PER:
                    q_pred = sess.run(poacher.output, {poacher.input_state: train_state})
                    q_pred = q_pred[np.arange(args.batch_size), train_action]
                    TD_error_batch = np.abs(q_target - q_pred)
                    replay_buffer.update(idx_batch, TD_error_batch)

                if not args.PER:
                    weight = np.ones(args.batch_size) 
                else:
                    weight = weight_batch 

                ### Update parameter
                feed = {
                    poacher.input_state: train_state,
                    poacher.actions: train_action,
                    poacher.q_target: q_target,
                    poacher.learning_rate: learning_rate,
                    poacher.loss_weight: weight
                }
                sess.run(poacher.train_op, feed_dict=feed)

            ### Update target network
            if global_step > 0 and global_step % args.target_update_every == 0:
                sess.run(po_copy_op)

            ### game ends: 1) the patroller catches the poacher and removes all the snares; 
            ###            2) the maximum time step is achieved
            if end_game or (t == args.max_time - 1):
                info = str(e) + "\tepisode\t%s\tlength\t%s\ttotal_reward\t%s\taverage_reward\t%s" % \
                       (e, t + 1, episode_reward, 1. * episode_reward / (t + 1))
                if  e % args.print_every == 0:
                    log.write(info + '\n')
                    print('po ' + info)
                    #log.flush()
                break

        ### save model
        if  e > 0 and e % args.save_every_episode == 0 or e == train_episode_num - 1:
            save_name = args.save_path + 'iteration_' + str(iteration) +  '_epoch_'+ str(e) +  "_po_model.ckpt"
            poacher.save(sess=sess, filename=save_name)
            #print('Save model to ' + save_name)

        ### test 
        if e == train_episode_num - 1 or ( e > 0 and e % args.test_every_episode  == 0):
            po_utility = 0.0
            test_total_reward = np.zeros(len(pa_strategy))

            ### test against each patroller strategy in the current strategy set
            for pa_strat in range(len(pa_strategy)):
                if pa_strategy[pa_strat] > 1e-10:
                    _, test_total_reward[pa_strat], _ = test_(patrollers[pa_strat], poacher, \
                        env, sess,args, iteration, e, poacher_type = 'DQN', patroller_type = pa_type[pa_strat])
                    po_utility += pa_strategy[pa_strat] * test_total_reward[pa_strat]

            test_utility.append(po_utility)

            if po_utility > best_utility and (e > min(50000, train_episode_num / 2) or args.row_num == 3):
                best_utility = po_utility
                sess.run(po_good_copy_op)
                final_utility[1] = po_utility
            
            info = [str(po_utility)] + [str(x)  for x in test_total_reward]
            info = 'test   '  + str(e) + '   ' +  '\t'.join(info) + '\n'
            #print('reward is: ', info)
            print('po ' + info)
            test_log.write(info)
            test_log.flush()

    test_log.close()
    log.close()

### adding the PER training
def calc_pa_best_response_PER(patroller, target_patroller, pa_copy_op, pa_good_copy_op, poachers, po_strategy, po_type,
        iteration, sess, env, args, final_utility, starting_e, train_episode_num = None, po_locations = None):
    
    '''
    po_locations: if is purely global mode, then po_locations is None
        else, it is the local + global retrain mode. each entry of po_locations specify the local mode of that poacher.
    Other things are basically the same as the function 'calc_po_best_response_PER'
    '''

    po_location = None

    #print('FIND_patroller_best_response iteration: ' + str(iteration))
    if train_episode_num is None:
        train_episode_num = args.pa_episode_num

    decrease_time = 1.0 / args.epsilon_decrease
    epsilon_decrease_every = train_episode_num // decrease_time

    if not args.PER:
        replay_buffer = ReplayBuffer(args, args.pa_replay_buffer_size)
    else:
        replay_buffer = PERMemory(args)
    best_utility = -10000.0
    test_utility = []

    if starting_e == 0:
        log = open(args.save_path + 'pa_log_train_iter_' + str(iteration) + '.dat', 'w')
        test_log = open(args.save_path + 'pa_log_test_iter_' + str(iteration) +  '.dat', 'w')
    else:
        log = open(args.save_path + 'pa_log_train_iter_' + str(iteration) + '.dat', 'a')
        test_log = open(args.save_path + 'pa_log_test_iter_' + str(iteration) +  '.dat', 'a')

    epsilon = 1.0
    learning_rate = args.po_initial_lr
    global_step = 0
    action_id = {
        'still': 0,
        'up': 1,
        'down': 2,
        'left': 3,
        'right': 4
    }

    sess.run(pa_copy_op)

    for e in range(starting_e, starting_e + train_episode_num):
        if e > 0 and e % epsilon_decrease_every == 0:
            epsilon = max(0.1, epsilon - args.epsilon_decrease)
        if e % args.mix_every_episode == 0 or e == starting_e:
            po_chosen_strat = np.argmax(np.random.multinomial(1, po_strategy))
            poacher = poachers[po_chosen_strat]
            type = po_type[po_chosen_strat]
            if po_locations is not None: # loacl + global mode, needs to change the poacher mode
                po_location = po_locations[po_chosen_strat]


        ### reset the environment
        poacher.reset_snare_num()
        pa_state, po_state = env.reset_game(po_location)
        episode_reward = 0.0
        pa_action = 'still'

        for t in range(args.max_time):
            global_step += 1

            ### transition records the (s,a,r,s) tuples
            transition = []

            ### poacher chooses an action
            ### doing so is because heuristic and DQN agent has different infer_action API
            if type == 'DQN':
                if not env.catch_flag and not env.home_flag: # if poacher is not caught, it can still do actions
                    po_state = np.array([po_state])
                    snare_flag, po_action = poacher.infer_action(sess=sess, states=po_state, policy="greedy")
                else: ### however, if it is caught, just make it stay still and does nothing
                    snare_flag = 0
                    po_action = 'still'
            elif type == 'PARAM':
                po_loc = env.po_loc
                if not env.catch_flag and not env.home_flag: 
                    snare_flag, po_action = poacher.infer_action(loc=po_loc,
                                                                local_trace=env.get_local_pa_trace(po_loc),
                                                                local_snare=env.get_local_snare(po_loc),
                                                                initial_loc=env.po_initial_loc)
                else:
                    snare_flag = 0
                    po_action = 'still'

            ### transition appends the current state
            transition.append(pa_state)

            ### patroller chooses an action
            pa_state = np.array([pa_state])
            pa_action = patroller.infer_action(sess=sess, states=pa_state, policy="epsilon_greedy", epsilon=epsilon)

            ### transition adds action
            transition.append(action_id[pa_action])

            ### the game moves on a step.
            pa_state, pa_reward, po_state, _, end_game = \
              env.step(pa_action, po_action, snare_flag)

            ### transition adds reward and the next state 
            episode_reward += pa_reward
            transition.append(pa_reward)
            transition.append(pa_state)

            ### Add transition to replay buffer
            replay_buffer.add_transition(transition)

            ### Start training
            ### Sample a minibatch, if the replay buffer has been full
            if replay_buffer.size >= args.batch_size:
                if not args.PER:
                    train_state, train_action, train_reward, train_new_state = \
                        replay_buffer.sample_batch(args.batch_size)
                else:
                    train_state, train_action, train_reward,train_new_state, \
                      idx_batch, weight_batch = replay_buffer.sample_batch(args.batch_size)

                ### Double DQN get target
                max_index = patroller.get_max_q_index(sess=sess, states=train_new_state)
                max_q = target_patroller.get_q_by_index(sess=sess, states=train_new_state, index=max_index)

                q_target = train_reward + args.reward_gamma * max_q

                if args.PER:
                    q_pred = sess.run(patroller.output, {patroller.input_state: train_state})
                    q_pred = q_pred[np.arange(args.batch_size), train_action]
                    TD_error_batch = np.abs(q_target - q_pred)
                    replay_buffer.update(idx_batch, TD_error_batch)

                if not args.PER:
                    weight = np.ones(args.batch_size)
                else:
                    weight = weight_batch 

                ### Update parameter
                feed = {
                    patroller.input_state: train_state,
                    patroller.actions: train_action,
                    patroller.q_target: q_target,
                    patroller.learning_rate: learning_rate,
                    patroller.weight_loss: weight
                }
                sess.run(patroller.train_op, feed_dict=feed)

            ### Update target network
            if global_step % args.target_update_every == 0:
                sess.run(pa_copy_op)

            ### game ends: 1) the patroller catches the poacher and removes all the snares; 
            ###            2) the maximum time step is achieved
            if end_game or (t == args.max_time - 1):
                info = str(e) + "\tepisode\t%s\tlength\t%s\ttotal_reward\t%s\taverage_reward\t%s" % \
                       (e, t + 1, episode_reward, 1. * episode_reward / (t + 1))
                if  e % args.print_every == 0:
                    log.write(info + '\n')
                    print('pa ' + info)
                    # log.flush()
                break


        ### save the models, and test if they are good 
        if  e > 0 and e % args.save_every_episode == 0 or e == train_episode_num - 1:
            save_name = args.save_path + 'iteration_' + str(iteration) + '_epoch_' + str(e) +  "_pa_model.ckpt"
            patroller.save(sess=sess, filename=save_name)

        ### test the agent
        if e == train_episode_num - 1 or ( e > 0 and e % args.test_every_episode  == 0):
            ### test against each strategy the poacher is using now, compute the expected utility
            pa_utility = 0.0
            test_total_reward = np.zeros(len(po_strategy))
            for po_strat in range(len(po_strategy)):
                if po_strategy[po_strat] > 1e-10:
                    if po_locations is None: ### indicates the purely global mode
                        tmp_po_location = None
                    else: ### indicates the local + global retrain mode, needs to set poacher mode
                        tmp_po_location = po_locations[po_strat]
                    test_total_reward[po_strat], _, _ = test_(patroller, poachers[po_strat], \
                            env, sess,args, iteration, e, patroller_type='DQN', poacher_type=po_type[po_strat],
                                po_location=tmp_po_location)
                    ### update the expected utility
                    pa_utility += po_strategy[po_strat] * test_total_reward[po_strat]

            test_utility.append(pa_utility)

            if pa_utility > best_utility and (e > min(50000, train_episode_num / 2) or args.row_num == 3):
                best_utility = pa_utility
                sess.run(pa_good_copy_op)
                final_utility[0] = pa_utility

            info = [str(pa_utility)] + [str(x)  for x in test_total_reward]
            info = 'test  ' + str(e) + '   ' +  '\t'.join(info) + '\n'
            #print('reward is: ', info)
            print('pa ' + info)
            test_log.write(info)
            test_log.flush()

    test_log.close()
    log.close()


