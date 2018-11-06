import tensorflow as tf
import numpy as np


class PatrollerValue(object):
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
            self.variables = []

            # Input placeholders
            self.input_state = tf.placeholder(tf.float32, [None, self.row_num, self.column_num, self.in_channel])
            self.state_values_target = tf.placeholder(tf.float32)
            # 4-D, [batch_size, row, col, state_channel] [-1, 7, 7, 10]
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
                                          initializer=tf.truncated_normal([64, 1]))
                self.bo = tf.get_variable(name='bias', initializer=tf.zeros([1]))
                self.state_values = tf.reshape(tf.add(tf.matmul(self.fc0, self.Wo), self.bo), [-1])  # batch_size

                self.variables.append(self.Wo)
                self.variables.append(self.bo)

            # Train operation
            self.loss = tf.reduce_mean(tf.square(self.state_values - self.state_values_target))
            self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=0.95,
                                                      epsilon=0.01).minimize(self.loss)

    def get_state_value(self, sess, states):
        state_values = sess.run(self.state_values, {self.input_state: states})
        return state_values

    def save(self, sess, filename):
        saver = tf.train.Saver(self.variables)
        saver.save(sess, filename)

    def load(self, sess, filename):
        saver = tf.train.Saver(self.variables)
        saver.restore(sess, filename)
