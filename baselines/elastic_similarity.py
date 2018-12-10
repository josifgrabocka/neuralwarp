from distutils.command.config import config

from dataset import Dataset
import sys
import os
import math
import numpy as np
import time

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("warp/")

# The implementation of Time Warped Edit Distance

def init_matrix(data):
    for i in range(len(data)):
        data[i][0] = float('inf')
    for i in range(len(data[0])):
        data[0][i] = float('inf')
    data[0][0] = 0
    return data


def LpDist(time_pt_1, time_pt_2):
    if (type(time_pt_1) == int and type(time_pt_2) == int):
        return abs(time_pt_1 - time_pt_2)
    else:
        return sum(abs(time_pt_1 - time_pt_2))


def TWED(t1, t2, lam, nu):
    """"Requires: t1: multivariate time series in numpy matrix format. t2: multivariate time series in numpy matrix format. lam: penalty lambda parameter, nu: stiffness coefficient"""
    """Returns the TWED distance between the two time series. """
    t1_data = t1
    t2_data = t2
    result = [[0] * len(t2_data) for row in range(len(t1_data))]
    result = init_matrix(result)
    n = len(t1_data)
    m = len(t2_data)

    #print(len(result), len(result[0]))

    t1_time = range(1, len(t1_data)+1)
    t2_time = range(1, len(t2_data)+1)

    #print(t1_time, t2_time)

    assert (len(t1_time) == n)
    assert (len(t2_time) == m)

    for i in range(1, n):
        for j in range(1, m):
            cost = LpDist(t1_data[i], t2_data[j])
            insertion = (result[i - 1][j] + LpDist(t1_data[i - 1], t1_data[i]) +
                         nu * (t1_time[i] - t1_time[i - 1] + lam))
            deletion = (result[i][j - 1] + LpDist(t2_data[j - 1], t2_data[j]) +
                        nu * (t2_time[j] - t2_time[j - 1] + lam))
            # print i, j, n , m, t1_time[i], t2_time[j]
            match = (result[i - 1][j - 1] + LpDist(t1_data[i], t2_data[j]) +
                     nu * (abs(t1_time[i] - t2_time[j])) +
                     LpDist(t1_time[i - 1], t2_time[j - 1]) +
                     nu * (abs(t1_time[i - 1] - t2_time[j - 1])))
            result[i][j] = min(insertion, deletion, match)
    return result[n - 1][m - 1]


# create the optimizer
dataset_folder = sys.argv[1]
measure = sys.argv[2]
start_pct = float(sys.argv[3])
chunk_pct = float(sys.argv[4])

dataset = Dataset()
dataset.load_multivariate(dataset_folder)

score = 0.0

start_range = int(start_pct*dataset.num_test_instances)
end_range = int((start_pct+chunk_pct)*dataset.num_test_instances)

if end_range > dataset.num_test_instances:
    end_range = dataset.num_test_instances

print('Test indices from', start_range, end_range)

start_time= time.clock()

for i in range(start_range, end_range):

    x = dataset.X_test[i]

    dmin, jmin = math.inf, -1

    for j in range(dataset.num_train_instances):

        y = dataset.X_train[j]

        d = 0
        if measure == "dtw":
            d, _ = fastdtw(x, y, dist=euclidean)
        elif measure == "twed":
            d = TWED(x, y, 1.0, 0.001)

        if d < dmin:
            dmin = d
            jmin = j

    score += 1.0 if (np.array_equal(dataset.Y_test[i], dataset.Y_train[jmin])) else 0.0
    print(i-start_range+1, score, score/(i-start_range+1))

elapsed_time = time.clock() - start_time

print(dataset.dataset_name, elapsed_time, end_range-start_range, score)

