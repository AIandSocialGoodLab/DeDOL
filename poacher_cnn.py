import tensorflow as tf
import numpy as np
import random


class Poacher(object):
    def __init__(self, args, scope):
        with tf.variable_scope(scope):
            self.args = args
            self.row_num = self.args.row_num
            self.column_num = self.args.column_num
            self.in_channel = self.args.po_state_size
            self.snare_num = self.args.snare_num
            self.batch_size = self.args.batch_size
            self.id_action = {
                0: ['still', 0],
                1: ['up', 0],
                2: ['down', 0],
                3: ['left', 0],
                4: ['right', 0],
                5: ['still', 1],
                6: ['up', 1],
                7: ['down', 1],
                8: ['left', 1],
                9: ['right', 1]
            }
            self.variables = []

            # Input placeholders
            self.input_state = tf.placeholder(tf.float32, [None, self.row_num, self.column_num, self.in_channel])
            self.actions = tf.placeholder(tf.int32)  # 1-D, [batch_size]
            self.q_target = tf.placeholder(tf.float32)  # 1-D, [batch_size]
            self.learning_rate = tf.placeholder(tf.float32)
            self.loss_weight = tf.placeholder(tf.float32) # 1-D, [batch_size] the weight of the loss in PER

            # Build Graph
            with tf.variable_scope('conv-maxpool-0'):
                if self.args.row_num == 7:
                    filter_shape = [4, 4, self.in_channel, 16]
                elif self.args.row_num == 5:
                    filter_shape = [3, 3, self.in_channel, 16]
                elif self.args.row_num == 3:
                    filter_shape = [2, 2, self.in_channel, 16]
                self.W0 = tf.get_variable(name='weights',
                                          initializer=tf.truncated_normal(filter_shape, stddev=0.001))
                self.b0 = tf.get_variable(name='bias', initializer=tf.zeros([16]))
                self.conv0 = tf.nn.conv2d(self.input_state, self.W0, strides=[1, 1, 1, 1], padding="SAME")
                self.conv0 = tf.nn.relu(tf.nn.bias_add(self.conv0, self.b0), name="relu")
                # 7 rows: Batch x 7 x 7 x 16
                # 3 rows: Batch x 2 x 2 x 16

                self.variables.append(self.W0)
                self.variables.append(self.b0)

            with tf.variable_scope('conv-maxpool-1'):
                filter_shape = [2, 2, 16, 32]
                self.W1 = tf.get_variable(name='weights',
                                          initializer=tf.truncated_normal(filter_shape, stddev=0.001))
                self.b1 = tf.get_variable(name='bias', initializer=tf.zeros([32]))
                self.conv1 = tf.nn.conv2d(self.conv0, self.W1, strides=[1, 2, 2, 1], padding="SAME")
                self.conv1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.b1), name="relu")
                # 7 rows: Batch x 4 x 4 x 32
                # 3 rows: Batch x 2 x 2 x 32

                self.variables.append(self.W1)
                self.variables.append(self.b1)

            with tf.variable_scope('fc0'):
                if self.args.row_num == 7:
                    self.Wf0 = tf.get_variable(name='weights',
                                            initializer=tf.truncated_normal([4 * 4 * 32, 64], stddev=0.001))
                    self.fc0 = tf.reshape(self.conv1, [-1, 4 * 4 * 32])
                elif self.args.row_num == 5:
                    self.Wf0 = tf.get_variable(name='weights',
                                           initializer=tf.truncated_normal([3 * 3 * 32, 64], stddev=0.001))
                    self.fc0 = tf.reshape(self.conv1, [-1, 3 * 3 * 32])
                elif self.args.row_num == 3:
                    self.Wf0 = tf.get_variable(name='weights',
                                           initializer=tf.truncated_normal([2 * 2 * 32, 64], stddev=0.001))
                    self.fc0 = tf.reshape(self.conv1, [-1, 2 * 2 * 32])
                self.bf0 = tf.get_variable(name='bias', initializer=tf.zeros([64]))
                self.fc0 = tf.add(tf.matmul(self.fc0, self.Wf0), self.bf0)
                self.fc0 = tf.nn.relu(self.fc0)

                self.variables.append(self.Wf0)
                self.variables.append(self.bf0)

            with tf.variable_scope('out'):
                if not self.args.advanced_training:
                    self.Wo = tf.get_variable(name='weights',
                                            initializer=tf.truncated_normal([64, self.args.po_num_actions]))
                    self.bo = tf.get_variable(name='bias', initializer=tf.zeros([self.args.po_num_actions]))
                    self.output = tf.add(tf.matmul(self.fc0, self.Wo), self.bo)

                    self.variables.append(self.Wo)
                    self.variables.append(self.bo)
                
                else:
                    ## dueling network
                    self.Wao = tf.get_variable(name='aweights',
                                            initializer=tf.truncated_normal([64, self.args.po_num_actions]))
                    self.bao = tf.get_variable(name='abias', initializer=tf.zeros([self.args.po_num_actions]))
                    
                    self.Wvo  = tf.get_variable(name ='vweights', 
                                                initializer=tf.truncated_normal([64, 1]))
                    self.bvo = tf.get_variable(name = 'vbias', initializer=tf.zeros([1]))
                    
                    self.a_output = tf.add(tf.matmul(self.fc0, self.Wao), self.bao)
                    self.a_output -= tf.reduce_mean(self.a_output)
                    
                    self.v_output = tf.add(tf.matmul(self.fc0, self.Wvo), self.bvo)
                    
                    self.output = tf.add(self.v_output, self.a_output)
                    
                    self.variables.append(self.Wao)
                    self.variables.append(self.bao)
                    self.variables.append(self.Wvo)
                    self.variables.append(self.bvo)

            # Train operation
            self.actions_onehot = tf.one_hot(self.actions, self.args.po_num_actions, 1.0, 0.0)
            self.loss = tf.reduce_mean(
                tf.square(
                    self.q_target - tf.reduce_sum(tf.multiply(self.actions_onehot, self.output), axis=1)
                ) * self.loss_weight
            )  

            if not self.args.advanced_training:
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=0.95,
                                                        epsilon=0.01).minimize(self.loss)
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        epsilon=0.01).minimize(self.loss)

            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        epsilon=0.01)
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 2.0)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def initial_loc(self, idx = None):
        # initial_loc = [self.row_num / 2 - 1, self.column_num / 2]
        # candidate = [[self.row_num / 2, -1], [self.row_num / 2, self.column_num], [-1, self.column_num / 2],
        #              [self.row_num, self.column_num / 2]]
        # candidate = [[-1, 0], [0, self.column_num], [self.row_num, self.column_num - 1], [self.row_num - 1, - 1]]
        
        candidate = [[0, 0], [0, self.column_num - 1], [self.row_num - 1, self.column_num - 1], [self.row_num - 1, 0]]
        
        # if home_idx is not None:
        #     self.home_location = candidate[home_idx]
        # else:
        #     self.home_location = candidate[random.randint(0,3)]
        
        if idx is not None:
            return candidate[idx]
        
        index = random.randint(0, 3)
        return candidate[index]

    def step(self, snare_flag):
        if snare_flag and self.snare_num > 0:
            self.snare_num -= 1

    def infer_action(self, sess, states, policy, epsilon=0.95):
        """
        :param states: a batch of states
        :param policy: "epsilon_greedy", "greedy"
        :param epsilon: exploration parameter for epsilon_greedy
        :return: a batch of actions
        """
        q_values = sess.run(self.output, {self.input_state: states})
        # print list(q_values[0])
        argmax_actions = np.argmax(q_values, axis=1)
        assert len(argmax_actions) == 1
        argmax_action = argmax_actions[0]

        if policy == "greedy":
            action, snare_flag = self.id_action[argmax_action]
            return snare_flag, action
        elif policy == "epsilon_greedy":
            if random.random() < epsilon:
                action, snare_flag = self.id_action[random.randint(0, self.args.po_num_actions - 1)]
            else:
                action, snare_flag = self.id_action[argmax_action]
            return snare_flag, action

    def reset_snare_num(self):
        self.snare_num = self.args.snare_num

    def random_action(self):
        return np.random.choice([(0, 'still'), (0, 'up'), (0, 'down'), (0, 'left'), (0, 'right'),
                                 (1, 'still'), (1, 'up'), (1, 'down'), (1, 'left'), (1, 'right')])
    

    def get_max_q(self, sess, states):
        q_values = sess.run(self.output, {self.input_state: states})
        return np.max(q_values, axis=1)

    def get_max_q_index(self, sess, states):
        q_values = sess.run(self.output, {self.input_state: states})
        return np.argmax(q_values, axis=1)

    def get_q_by_index(self, sess, states, index):
        q_values = sess.run(self.output, {self.input_state: states})
        select_q = []
        for i, idx in enumerate(index):
            select_q.append(q_values[i][idx])
        return np.array(select_q)

    def save(self, sess, filename):
        saver = tf.train.Saver(self.variables)
        saver.save(sess, filename)

    def load(self, sess, filename):
        saver = tf.train.Saver(self.variables)
        saver.restore(sess, filename)
