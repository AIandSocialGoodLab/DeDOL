import tensorflow as tf
import numpy as np


class PatrollerPolicy(object):
    def __init__(self, args, scope):
        with tf.variable_scope(scope):
            self.args = args
            self.row_num = self.args.row_num
            self.column_num = self.args.column_num
            self.in_channel = self.args.pa_state_size
            self.id_action = {
                0: 'still',
                1: 'up',
                2: 'down',
                3: 'left',
                4: 'right'
            }
            self.action_space = np.arange(self.args.pa_num_actions)
            self.variables = []

            # Input placeholders
            self.input_state = tf.placeholder(tf.float32, [None, self.row_num, self.column_num, self.in_channel])
            # 4-D, [batch_size, row, col, state_channel] [-1, 7, 7, 10]
            self.actions = tf.placeholder(tf.int32)  # 1-D, [batch_size]
            self.advantage = tf.placeholder(tf.float32)  # 1-D, [batch_size]
            self.learning_rate = tf.placeholder(tf.float32)

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
                # Batch x 7 x 7 x 16

                self.variables.append(self.W0)
                self.variables.append(self.b0)

            with tf.variable_scope('conv-maxpool-1'):
                filter_shape = [2, 2, 16, 32]
                self.W1 = tf.get_variable(name='weights',
                                          initializer=tf.truncated_normal(filter_shape, stddev=0.001))
                self.b1 = tf.get_variable(name='bias', initializer=tf.zeros([32]))
                self.conv1 = tf.nn.conv2d(self.conv0, self.W1, strides=[1, 2, 2, 1], padding="SAME")
                self.conv1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.b1), name="relu")
                # Batch x 4 x 4 x 32

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
                # self.Wf0 = tf.get_variable(name='weights',
                #                            initializer=tf.truncated_normal([4 * 4 * 32, 64], stddev=0.001))
                self.bf0 = tf.get_variable(name='bias', initializer=tf.zeros([64]))
                # self.fc0 = tf.reshape(self.conv1, [-1, 4 * 4 * 32])
                self.fc0 = tf.add(tf.matmul(self.fc0, self.Wf0), self.bf0)
                self.fc0 = tf.nn.relu(self.fc0)

                self.variables.append(self.Wf0)
                self.variables.append(self.bf0)

            with tf.variable_scope('out'):
                self.Wo = tf.get_variable(name='weights',
                                          initializer=tf.truncated_normal([64, self.args.pa_num_actions]))
                self.bo = tf.get_variable(name='bias', initializer=tf.zeros([self.args.pa_num_actions]))
                self.action_logits = tf.add(tf.matmul(self.fc0, self.Wo), self.bo)
                self.action_prob = tf.nn.softmax(self.action_logits)

                self.variables.append(self.Wo)
                self.variables.append(self.bo)

            # Train operation
            self.actions_onehot = tf.one_hot(self.actions, self.args.pa_num_actions, 1.0, 0.0)
            self.action_log_prob = tf.reduce_sum(tf.multiply(self.actions_onehot, tf.log(self.action_prob)), axis=1)
            self.loss = - tf.reduce_mean(tf.multiply(self.action_log_prob, self.advantage))
            self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=0.95,
                                                      epsilon=0.01).minimize(self.loss)

    def infer_action(self, sess, states):
        action_prob = sess.run(self.action_prob, {self.input_state: states})
        # print(action_prob)
        assert len(action_prob) == 1
        action_prob = action_prob[0]
        action_id = np.random.choice(self.action_space, p=action_prob)
        return self.id_action[action_id]

    def random_action(self):
        return np.random.choice(['still', 'up', 'down', 'left', 'right'])

    def initial_loc(self):
        return [self.row_num / 2, self.column_num / 2]

    def save(self, sess, filename):
        saver = tf.train.Saver(self.variables)
        saver.save(sess, filename)

    def load(self, sess, filename):
        saver = tf.train.Saver(self.variables)
        saver.restore(sess, filename)
