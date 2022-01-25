from collections import Iterable

from sympy.core import function

# from LinearApproximator import LinearApproximator
from LinearScikit import LinearApproximator
from ModelsTree import ModelSelector, DivisionNode, step_perpendicular, surface_side
import numpy as np
from matplotlib import pyplot as plt


class LinearSpline:

    def __init__(self):
        self.model_selector = ModelSelector()

    def approximate(self, f: function, training_set, lina: LinearApproximator, max_error, step, div_node=""):
        if lina is None:
            lina = self.local_approximator(training_set,f)
        ms_errors = lina.get_errors(training_set, f)

        if np.mean(ms_errors) > max_error:
            max_ms = max(ms_errors)
            max_error_index = ms_errors.index(max_ms)

            t_set_left, t_set_right, d_node = self.divide_set(training_set, max_error_index,ms_errors ,f, div_node)
            if t_set_left == -1 :
                self.model_selector.add_node(lina, parent=d_node)
                return
            la_left = self.local_approximator(t_set_left, f)
            self.approximate(f, t_set_left, la_left, max_error, step=step, div_node=d_node)

            la_right = self.local_approximator(t_set_right, f)
            self.approximate(f, t_set_right, la_right, max_error, step, d_node)
        else:
            self.model_selector.add_node(lina, parent=div_node)

    def divide_set(self,
                   training_set,
                   point_index,
                   ms_errors,
                   f: function,
                   parent_node,
                   division_method: function = step_perpendicular,
                   assign_left: function = surface_side,
                   model_selector = None):

        if len(training_set) < 6 or len(ms_errors) < 2:
            return -1, -1, parent_node

        t_set_left = []
        t_set_right = []

        div_node = DivisionNode(division_method(training_set, point_index))
        for t_example in training_set:
            if assign_left(t_example, div_node.parameters):
                t_set_left.append(t_example)
            else:
                t_set_right.append(t_example)

        if len(t_set_left) < 3 or len(t_set_right) < 3:
            ms_errors_next = [m for m in ms_errors if m < max(ms_errors)]
            max_e = max(ms_errors_next)
            max_index = ms_errors_next.index(max_e)
            return self.divide_set(training_set, max_index, ms_errors_next, f, parent_node, division_method, assign_left)
        if model_selector is None:
            node = self.model_selector.add_node(div_node, parent=parent_node)
        else:
            node = model_selector.add_node(div_node, parent=parent_node)
        return t_set_left, t_set_right, node

    def local_approximator (self, training_set, f:function, values = None):
        if not isinstance(training_set[0], list):
            x_length = len([training_set[0]])
        else:
            x_length = len(training_set[0])
        la_parameters = np.zeros(shape=(x_length + 1)).tolist()
        la = LinearApproximator(la_parameters)
        la.update_parameters(training_set=training_set, f=f, values= values)
        return la

    def evaluate(self, x):
        la : LinearApproximator = self.model_selector.select_model(x)
        return la.evaluate(x)
