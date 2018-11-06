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

    def initial_loc(self):
        # initial_loc = [self.row_num / 2 - 1, self.column_num / 2]
        # candidate = [[self.row_num / 2, -1], [self.row_num / 2, self.column_num], [-1, self.column_num / 2],
        #              [self.row_num, self.column_num / 2]]

        candidate = [[0, 0], [0, self.column_num - 1], [self.row_num - 1, self.column_num - 1], [self.row_num - 1, 0]]

        index = random.randint(0, 3)

        self.reset_snare_num()
        return candidate[index]

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
            if tuple(initial_loc) != tuple(loc):
                home_direction = np.array(initial_loc) - np.array(loc)
                home_direction = home_direction / np.sqrt(np.sum(np.square(home_direction)))  # Normalize
                move_direction = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
                logits = self.po_home_dir_w * np.sum(home_direction * move_direction, axis=1) + \
                         self.po_act_enter_w * np.array(local_trace[:4]) + \
                         self.po_act_leave_w * np.array(local_trace[4:])
                logits = 1. * logits / self.po_act_temp
                prob = np.exp(logits) / np.sum(np.exp(logits))
                action = np.random.choice(['up', 'down', 'left', 'right'], p=prob)
            else:
                action = 'still'

        return snare_flag, action

    def reset_snare_num(self):
        self.snare_num = self.args.snare_num

    def in_bound(self, row, col):
        return row >= 0 and row <= (self.row_num - 1) and col >= 0 and col <= (self.column_num - 1)
