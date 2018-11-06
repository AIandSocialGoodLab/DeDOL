import numpy as np
import random


class Poacher(object):
    def __init__(self, args, animal_density):
        self.args = args
        self.row_num = self.args.row_num
        self.column_num = self.args.column_num
        self.snare_num = self.args.snare_num
        self.po_act_den_w = self.args.po_act_den_w
        self.po_act_enter_w = self.args.po_act_enter_w
        self.po_act_leave_w = self.args.po_act_leave_w
        self.po_act_temp = self.args.po_act_temp
        self.po_home_dir_w = self.args.po_home_dir_w

        self.animal_density = animal_density
        self.animal_density_sum = np.mean(self.animal_density)

    def initial_loc(self, idx = None):

        candidate = [[0, 0], [0, self.column_num - 1], [self.row_num - 1, self.column_num - 1], [self.row_num - 1, 0]]

        if idx is not None:
            return candidate[idx]

        index = random.randint(0, 3)

        self.reset_snare_num()
        return candidate[index]

    def step(self, snare_flag):
        if snare_flag and self.snare_num > 0:
            self.snare_num -= 1

    def infer_action_probs(self, loc, local_trace, local_snare, initial_loc, snare_num, valid_actions):
        # First compute if place a snare at the current location
        
        snare_prob = np.zeros(2)

        if snare_num > 0:
            if local_snare == 0:  # There is no snare at this place now
                loc_ani_den = self.animal_density[loc[0], loc[1]]
                prob = np.array([10 * self.animal_density_sum, loc_ani_den])
                prob = prob / self.po_act_temp  # temperature
                prob = np.exp(prob) / np.sum(np.exp(prob))
                snare_prob[0], snare_prob[1] = prob[0], prob[1]
            else:
                snare_prob[0], snare_prob[1] = 1,0 
        else:
            snare_prob[0], snare_prob[1] = 1,0 

        # print(snare_prob)

        direction_prob = {}

        # Second decide which direction to move
        if snare_num > 0:
            # Poacher need to continue placing snares, thus animal density is considered
            if loc[0] > 0:
                up_ani_den = np.mean(self.animal_density[:loc[0], :])
            else:
                up_ani_den = 0.

            if loc[0] < self.column_num - 1:
                down_ani_den = np.mean(self.animal_density[loc[0] + 1:, :])
            else:
                down_ani_den = 0.

            if loc[1] > 0:
                left_ani_den = np.mean(self.animal_density[:, :loc[1]])
            else:
                left_ani_den = 0.

            if loc[1] < self.column_num - 1:
                right_ani_den = np.mean(self.animal_density[:, loc[1] + 1:])
            else:
                right_ani_den = 0.

            if self.in_bound(loc[0], loc[1]):
                cur_ani_den = self.animal_density[loc[0], loc[1]]
            else:
                cur_ani_den = 0.

            logits = self.po_act_den_w * np.array([up_ani_den, down_ani_den, left_ani_den, right_ani_den]) + \
                     self.po_act_enter_w * np.array(local_trace[:4]) + self.po_act_leave_w * np.array(local_trace[4:])
            logits = np.array([self.po_act_den_w * cur_ani_den] + list(logits))
            logits_temp = 1. * logits / self.po_act_temp
            prob = np.exp(logits_temp) / np.sum(np.exp(logits_temp))
            for idx, action in enumerate(['still', 'up', 'down', 'left', 'right']):
                direction_prob[action] = prob[idx]
        else:
            # If all the snare has been placed, then it's time to go back to the entry
            home_direction = np.array(initial_loc) - np.array(loc)
            if home_direction[0] == 0 and home_direction[1] == 0:
                home_direction = 0
            else:
                home_direction = home_direction / np.sqrt(np.sum(np.square(home_direction)))  # Normalize
            move_direction = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            logits = self.po_home_dir_w * np.sum(home_direction * move_direction, axis=1) + \
                     self.po_act_enter_w * np.array(local_trace[:4]) + \
                     self.po_act_leave_w * np.array(local_trace[4:])
            logits = 1. * logits / self.po_act_temp
            prob = np.exp(logits) / np.sum(np.exp(logits))
            for idx, action in enumerate(['up', 'down', 'left', 'right']):
                direction_prob[action] = prob[idx]
            direction_prob['still'] = 0

        # if move in invalid directions, this probability adds up to stay still
        valid_directions = [x[0] for x in valid_actions]

        # print('valid actions: ', valid_actions)
        # print('valid directions: ', valid_directions)
        # print('before direction probs: ', direction_prob)

        for dir in ['up','left','right','down']:
            if dir not in valid_directions:
                direction_prob['still'] += direction_prob[dir]
                direction_prob[dir] = 0

        # print('after removing invalid directions direction probs: ', direction_prob)

        dir_prob_sum = 0.0
        for x in direction_prob:
            dir_prob_sum += direction_prob[x]
        assert np.abs(dir_prob_sum - 1) < 1e-10

        # print('snare probs: ', snare_prob)

        action_probs = np.zeros(len(valid_actions))
        for idx, va in enumerate(valid_actions):
            action_probs[idx] = snare_prob[va[1]] * direction_prob[va[0]]

        # print(valid_actions)
        # print(action_probs)
        assert np.abs(np.sum(action_probs) - 1) < 1e-10
        return action_probs

    def infer_action(self, loc, local_trace, local_snare, initial_loc):
        # First compute if place a snare at the current location
        snare_flag = 0
        if self.snare_num > 0 and self.in_bound(loc[0], loc[1]):
            if local_snare == 0:  # There is no snare at this place now
                loc_ani_den = self.animal_density[loc[0], loc[1]]
                prob = np.array([10 * self.animal_density_sum, loc_ani_den])
                prob = prob / self.po_act_temp  # temperature
                prob = np.exp(prob) / np.sum(np.exp(prob))
                snare_flag = np.random.choice([0, 1], p=prob)
                if snare_flag:
                    self.snare_num -= 1

        # Second decide which direction to move
        if self.snare_num > 0:
            # Poacher need to continue placing snares, thus animal density is considered
            if loc[0] > 0:
                up_ani_den = np.mean(self.animal_density[:loc[0], :])
            else:
                up_ani_den = 0.

            if loc[0] < self.column_num - 1:
                down_ani_den = np.mean(self.animal_density[loc[0] + 1:, :])
            else:
                down_ani_den = 0.

            if loc[1] > 0:
                left_ani_den = np.mean(self.animal_density[:, :loc[1]])
            else:
                left_ani_den = 0.

            if loc[1] < self.column_num - 1:
                right_ani_den = np.mean(self.animal_density[:, loc[1] + 1:])
            else:
                right_ani_den = 0.

            if self.in_bound(loc[0], loc[1]):
                cur_ani_den = self.animal_density[loc[0], loc[1]]
            else:
                cur_ani_den = 0.

            logits = self.po_act_den_w * np.array([up_ani_den, down_ani_den, left_ani_den, right_ani_den]) + \
                     self.po_act_enter_w * np.array(local_trace[:4]) + self.po_act_leave_w * np.array(local_trace[4:])
            logits = np.array([self.po_act_den_w * cur_ani_den] + list(logits))
            logits_temp = 1. * logits / self.po_act_temp
            prob = np.exp(logits_temp) / np.sum(np.exp(logits_temp))
            action = np.random.choice(['still', 'up', 'down', 'left', 'right'], p=prob)
        else:
            # If all the snare has been placed, then it's time to go back to the entry
            home_direction = np.array(initial_loc) - np.array(loc)
            if home_direction[0] == 0 and home_direction[1] == 0:
                home_direction = 0
            else:
                home_direction = home_direction / np.sqrt(np.sum(np.square(home_direction)))  # Normalize
            move_direction = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            logits = self.po_home_dir_w * np.sum(home_direction * move_direction, axis=1) + \
                     self.po_act_enter_w * np.array(local_trace[:4]) + \
                     self.po_act_leave_w * np.array(local_trace[4:])
            logits = 1. * logits / self.po_act_temp
            prob = np.exp(logits) / np.sum(np.exp(logits))
            action = np.random.choice(['up', 'down', 'left', 'right'], p=prob)

        return snare_flag, action

    def reset_snare_num(self):
        self.snare_num = self.args.snare_num

    def in_bound(self, row, col):
        return row >= 0 and row <= (self.row_num - 1) and col >= 0 and col <= (self.column_num - 1)
