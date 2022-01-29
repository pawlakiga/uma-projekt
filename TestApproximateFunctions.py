from sympy.core import function

from cec2017.functions import f1, f2, f3, f4, f5, f10
import numpy as np
from matplotlib import pyplot as plt
# from LinearExtended import LinearExtended, step_perpendicular, divide_set
from Utils import init_method
from LinearSpline import LinearSpline
from LinearAxes import LinearAxes

def d_cube(x):
    return sum(np.multiply(x, np.multiply(x, x))) * 2 + 10


def d_sin(x):
    sin = 0
    for xx in x:
        sin += np.sin(xx) + 15
    return sin


def test_spline(f: function, training_Set, low_point, high_point, points_no, max_error):

    ls = LinearSpline()
    error, best_value, best_index= ls.approximate(training_set=training_Set, lina=None, max_error=max_error, f=f)
    return ls, error, best_value, best_index


def test_axes(f:function, training_Set, low_point, high_point, points_no, max_error):

    ls = LinearAxes()
    error, best_value, best_index = ls.approximate(training_set=training_Set, max_error=max_error, f=f)
    print("Final error: {error}".format(error = error))
    return ls, error, best_value, best_index



# test_axes(f5)
