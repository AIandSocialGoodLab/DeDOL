import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, args):
        self.pre_state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.post_state_buf = []
        self.args = args
        self.max_size = self.args.replay_buffer_size
        self.add_pointer = 0
        self.size = 0

    def add_transition(self, transition):
        pre_state = transition[0]
        action = transition[1]
        reward = transition[2]
        post_state = transition[3]

        if self.size < self.max_size:
            self.pre_state_buf.append(pre_state)
            self.action_buf.append(action)
            self.reward_buf.append(reward)
            self.post_state_buf.append(post_state)

            self.add_pointer = (self.add_pointer + 1) % self.max_size
            self.size += 1
        else:
            self.pre_state_buf[self.add_pointer] = pre_state
            self.action_buf[self.add_pointer] = action
            self.reward_buf[self.add_pointer] = reward
            self.post_state_buf[self.add_pointer] = post_state

            self.add_pointer = (self.add_pointer + 1) % self.max_size

    def sample_batch(self, batch_size):
        assert self.size >= batch_size
        ret_pre_state = []
        ret_action = []
        ret_reward = []
        ret_post_state = []
        for _ in xrange(batch_size):
            idx = random.randint(0, self.size - 1)
            ret_pre_state.append(self.pre_state_buf[idx])
            ret_action.append(self.action_buf[idx])
            ret_reward.append(self.reward_buf[idx])
            ret_post_state.append(self.post_state_buf[idx])
        return np.array(ret_pre_state), np.array(ret_action), np.array(ret_reward), np.array(ret_post_state)
