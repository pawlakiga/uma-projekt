from numpy.linalg import lstsq
from sympy.core import function
import numpy as np

from LinearApproximator import LinearApproximator


#
# def divide_set(divide_point_index, training_set, f:function):
#     t_set_left = []
#     t_set_right = []
#
#     divide_point = training_set[divide_point_index]
#     a, b = step_direction(training_set[divide_point_index-1], training_set[divide_point_index])
#     a = -1/a
#
#     for x in training_set:'


def step_perpendicular(training_set : list, point_index, f: function):
    previous_point = training_set[point_index - 1]
    break_point = training_set[point_index]

    step_vector = np.subtract(break_point, previous_point)
    surface_coordinates = []
    if not isinstance(step_vector, list):
        surface_coordinates.append(step_vector)
    else:
        for cord in step_vector:
            surface_coordinates.append(cord)
    surface_coordinates.append(-(np.sum(np.multiply(step_vector, break_point))))
    return surface_coordinates


def surface_side(t_example, div_surface_coordinates):
    return np.sum(np.multiply(div_surface_coordinates[:-1], t_example)) + div_surface_coordinates[-1] <= 0


class LinearExtended:
    def __init__(self):
        self.break_points = []
        self.linear_models = []
        self.division_parameters = []

    def approximate(self, f: function, training_set, lina: LinearApproximator, max_error, step):
        t_set = np.sort(training_set)  # .tolist()
        ms_errors = lina.get_errors(t_set, f)
        if max(ms_errors) > max_error and len(t_set) > 3:
            if isinstance(t_set[0], int):
                x_length = len([t_set[0]])
            else:
                x_length = len(t_set[0])

            parameters = np.zeros(shape=(x_length + 1)).tolist()
            max_e = max(ms_errors)
            if ms_errors.index(max_e) == 0:
                bp = t_set[2]
                bp_index = 2
            elif ms_errors.index(max_e) == len(t_set) - 1:
                bp = t_set[-3]
                bp_index = len(t_set) - 4
            else:
                bp = t_set[ms_errors.index(max_e)]
                bp_index = ms_errors.index(max_e)

            self.break_points.append(bp)
            # self.break_points.sort()
            lina1 = LinearApproximator(parameters)
            lina1.update_parameters(training_set=t_set[:bp_index], f=f, step=step, approx_method=None)
            lina2 = LinearApproximator(parameters)
            lina2.update_parameters(training_set=t_set[bp_index + 1:], f=f, step=step, approx_method=None)
            self.approximate(f, t_set[0:bp_index], lina1, max_error, step=step)
            self.approximate(f, t_set[bp_index + 1:], lina2, max_error, step=step)
        else:
            self.linear_models.append(lina)

    def evaluate(self, x):
        for bp in self.break_points:
            if bp > x:
                selected_bp = bp
                selected_model: LinearApproximator = self.linear_models[self.break_points.index(selected_bp)]
                y = selected_model.evaluate(x)
                return y
        y = self.linear_models[-1].evaluate(x)
        return y

    def divide_set(self, training_set, point_index, f: function, division_method: function = step_perpendicular,
                   assign_left: function = surface_side):
        t_set_left = []
        t_set_right = []

        self.division_parameters = division_method(training_set, point_index, f)
        for t_example in training_set:
            if assign_left(t_example, self.division_parameters):
                t_set_left.append(t_example)
            else:
                t_set_right.append(t_example)
        #
        # if t_set_left.count() <= 3:
        #     for i in range(3):
        #         right_min = min(t_set_right)
        #
        #         t_set_right.pop(right_min)

        return t_set_left, t_set_right

    # def select_model(self, x):
