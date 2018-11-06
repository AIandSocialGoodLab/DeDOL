import numpy as np


class Patroller_Rule(object):
    def __init__(self, args, animal_density):
        self.args = args
        self.row_num = self.args.row_num
        self.column_num = self.args.column_num

        self.animal_density = animal_density
        self.animal_density_mean = np.mean(self.animal_density)

    def initial_loc(self):
        return [int(self.row_num / 2), int(self.column_num / 2)]

    def infer_action(self, loc, local_trace, w_ani, w_enter, w_leave):
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

        cur_ani_den = self.animal_density[loc[0], loc[1]]

        logits = w_ani * np.array([up_ani_den, down_ani_den, left_ani_den, right_ani_den]) + \
                 w_enter * np.array(local_trace[:4]) + w_leave * np.array(local_trace[4:])
        logits = np.array([w_ani * cur_ani_den] + list(logits))
        prob = np.exp(logits) / np.sum(np.exp(logits))
        action = np.random.choice(['still', 'up', 'down', 'left', 'right'], p=prob)
        return action

    def random_action(self):
        return np.random.choice(['still', 'up', 'down', 'left', 'right'])
