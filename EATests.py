import pygad
import random

from sympy.core import function

from cec2017.functions import f1, f2, f3, f4, f5, f6, f7, f8, f9, f10
import numpy as np
from matplotlib import pyplot as plt
# from LinearExtended import LinearExtended, step_perpendicular, divide_set
from Utils import init_method, draw_figure
from LinearSpline import LinearSpline
from TestApproximateFunctions import test_spline, test_axes
from EA import create_run_ea

def test_ea_approximation(f : function, approximate : function):
    low_point, high_point, max_error = -10, 10, 1e-8

    training_set, best_ind = create_run_ea(f, dimensions=2, population_size=50, num_generations=40)
    ls, error, best_value, best_index = approximate(f=f,training_Set= training_set, low_point=low_point,
                                                  high_point=high_point, max_error=max_error, points_no=0)
    print("Approximation error = {error}".format(error=error))
    print("Minimum value of approximated function is {value}, for x = {x}".format(error=error, value=best_value,
                                                                                  x=training_set[best_index]))
    print("Minimum found by evolutionary algorithm is {value}, for x = {x}".format(x=best_ind[0], value=f(best_ind[0])))
    draw_figure(f=f, training_set=training_set, ls=ls, low_point=low_point, high_point=high_point, max_error=max_error)

test_ea_approximation(f5, test_axes)