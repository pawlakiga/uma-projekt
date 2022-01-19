import random
import numpy as np
import cec2017.functions as functions

def init_method(init_args):
    dimensions = init_args[0]
    x_min = init_args[1]
    x_max = init_args[2]
    #precision = init_args[3]
    if dimensions == 1:
        return np.random.uniform(low=x_min, high=x_max, size=dimensions)[0]
    return np.random.uniform(low=x_min, high=x_max, size=dimensions).tolist()


# x = []
# for i in range(3):
#     x.append(init_method([10,2,100, 0.1]))
#     print(x[i])
#     f = functions.f5
#     y = f(x[i])
#     print('%s( %.1f, %.1f, ... ) = %.2f' % (f.__name__, x[i][0], x[i][1], y))
#
#
# print(np.multiply([4,5],[2,3]))