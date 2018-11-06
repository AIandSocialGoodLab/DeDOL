import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, args, size):
        self.pre_state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.post_state_buf = []
        self.args = args
        self.max_size = size
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
        for _ in range(batch_size):
            idx = random.randint(0, self.size - 1)
            ret_pre_state.append(self.pre_state_buf[idx])
            ret_action.append(self.action_buf[idx])
            ret_reward.append(self.reward_buf[idx])
            ret_post_state.append(self.post_state_buf[idx])
        return np.array(ret_pre_state), np.array(ret_action), np.array(ret_reward), np.array(ret_post_state)
    
    
    
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.number = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        
        if idx >= self.capacity - 1:
            return idx
        
        left = 2 * idx + 1
        right = left + 1

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, data, p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        self.number = min(self.number + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def num(self):
        return self.number

    

class PERMemory():
    
    def __init__(self, args, alpha = 0.6, beta = 0.5, eps = 1e-6):
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.mem_size = args.replay_buffer_size
        self.mem = SumTree(args.replay_buffer_size)
        self.size = 0
        self.max_size = args.replay_buffer_size
        self.args = args
        beta_increase_time = args.episode_num * args.max_time / 5.0
        self.beta_increase = (1. - beta) / beta_increase_time 
        
    def add_transition(self, transition):
        # here use reward for initial p, instead of maximum for initial p
        p = 1000
        self.mem.add([transition[0], transition[1], transition[2], transition[3]], p)
        self.size = min(self.size + 1, self.mem_size)
        
    def update(self, batch_idx, batch_td_error):
        for idx, error in zip(batch_idx, batch_td_error):
            p = (error + self.eps)  ** self.alpha 
            self.mem.update(idx, p)
        
    def num(self):
        return self.mem.num()
    
    def sample_batch(self, batch_size):
        
        data_batch = []
        idx_batch = []
        p_batch = []
        
        segment = self.mem.total() / batch_size
        #print(self.mem.total())
        #print(segment * batch_size)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            #print(s < self.mem.total())
            idx, p, data = self.mem.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
            p_batch.append(p)
        
        p_batch = (1.0/ np.array(p_batch) /self.mem_size) ** self.beta
        p_batch /= max(p_batch)
    
        self.beta = min(self.beta + self.beta_increase, 1)
        
        pre_state = np.array([x[0] for x in data_batch])
        action = np.array([x[1] for x in data_batch])
        reward = np.array([x[2] for x in data_batch])
        next_state = np.array([x[3] for x in data_batch])
        return (pre_state, action, reward, next_state, idx_batch, p_batch)
