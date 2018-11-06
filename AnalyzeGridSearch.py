import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument('--path', type=str, default='33random')
args = argparser.parse_args()

file = open( args.path +  '_grid_search_results.pkl', 'rb')
log = open( args.path + '_grid_search_file.txt', 'w')

results, parameters = pickle.load(file, fix_imports=True, encoding='bytes')

X = {}
Y = {}
Z = {}

param_result = []

for result, param in zip(results, parameters):
    param_result.append((param, result[0]))

    if param[0] not in X:
        X[param[0]] = result[0]
    else:
        X[param[0]] += result[0]

    if param[1] not in Y:
        Y[param[1]] = result[0]
    else:
        Y[param[1]] += result[0]

    if param[2] not in Z:
        Z[param[2]] = result[0]
    else:
        Z[param[2]] += result[0]

ani = X.keys()
ani_result = [X[key] for key in ani]

enter = Y.keys()
enter_results = [Y[key] for key in enter]

leave = Z.keys()
leave_result = [Z[key] for key in leave]

# plt.figure(figsize=(8, 6))
# plt.plot(ani, ani_result, label='animal density')
# plt.savefig('ani.pdf')

# plt.figure(figsize=(8, 6))
# plt.plot(enter, enter_results, label='enter')
# plt.savefig('enter.pdf')

# plt.figure(figsize=(8, 6))
# plt.plot(leave, leave_result, label='leave')
# plt.savefig('leave.pdf')

param_result = sorted(param_result, key=lambda x: x[1], reverse=True)
for line in param_result[:50]:
    log.write(str(line) + '\n')
    print(line)
