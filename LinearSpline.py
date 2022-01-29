from sympy.core import function

from LinearScikit import LinearApproximator
from ModelsTree import ModelSelector, DivisionNode, step_perpendicular, surface_side
import numpy as np
from matplotlib import pyplot as plt


class LinearSpline:

    def __init__(self, model_selector : ModelSelector = None):
            if model_selector is None:
                self.model_selector = ModelSelector()
            else:
                self.model_selector = model_selector

    def approximate(self, f: function,
                    training_set,
                    lina: LinearApproximator,
                    max_error,
                    div_node="",
                    values = None):
        """
        Function to teach the spline approximator
        :param f: approximated function
        :param training_set:
        :param lina: local linear approximator currently used for training set
        :param max_error: maximum error above which the training set is divided
        :param div_node: division node, stores surface coordinates
        :param values: function values, optional
        :return:
        """
        if lina is None:
            lina = self.local_approximator(training_set,f, values)
        ms_errors = lina.get_errors(training_set, f, values)

        if np.mean(ms_errors) > max_error:
            max_ms = max(ms_errors)
            max_error_index = ms_errors.index(max_ms)

            t_set_left, t_set_right, d_node, _, _ = self.divide_set(training_set=training_set,
                                                                    point_index=max_error_index,
                                                                    ms_errors=ms_errors,
                                                                    f = f,
                                                                    parent_node=div_node)

            if t_set_left == -1 :
                # print("Training set cannot be divided - adding new leaf for set:  {len}".format(len=len(training_set)))
                self.model_selector.add_node(new_node=lina, parent=d_node)
                return
            la_left = self.local_approximator(t_set_left, f)
            self.approximate(f = f, training_set=t_set_left, lina=la_left, max_error=max_error, div_node=d_node)

            la_right = self.local_approximator(t_set_right, f)
            self.approximate(f=f, training_set=t_set_right, lina=la_right, max_error=max_error, div_node=d_node)
        else:
            # print("Error = {error} is smaller than max_error - adding new leaf for set: {len}".format(len=len(training_set), error = np.mean(ms_errors)))
            self.model_selector.add_node(new_node=lina, parent=div_node)
        final_error = np.mean(self.get_errors(training_set,f, values))
        best_value, best_index = self.get_best(training_set)
        return final_error, best_value, best_index

    def divide_set(self,
                   training_set,
                   point_index,
                   ms_errors,
                   f: function,
                   parent_node,
                   division_method: function = step_perpendicular,
                   assign_left: function = surface_side,
                   model_selector = None):
        """
        function to divide training set into subsets in a fiven break point using a given division method and an
        associated assigning function
        :param training_set:
        :param point_index: index in the training set that serves as the division point
        :param ms_errors: mean square errors for each point
        :param f: approximated function
        :param parent_node: parent for the node which will be created during division
        :param division_method: function used to calculate division parameters
        :param assign_left: function used to assign examples to subsets using division parameters
        :param model_selector: model selector objects, stores division nodes in nodes and local approximators in leaves
        :return: left subset, right subset, new node identifier, left examples indexes, right examples indexes
        """
        if len(training_set) < 6 or len(ms_errors) < 2:
            return -1, -1, parent_node, -1 ,-1

        t_set_left = []
        t_set_right = []
        ix = 0
        t_set_left_indexes = []
        t_set_right_indexes = []
        div_node = DivisionNode(division_method(training_set, point_index))
        for t_example in training_set:
            if assign_left(t_example, div_node.parameters):
                t_set_left.append(t_example)
                t_set_left_indexes.append(ix)
            else:
                t_set_right.append(t_example)
                t_set_right_indexes.append(ix)
            ix += 1

        if len(t_set_left) < 3 or len(t_set_right) < 3:
            ms_errors_next = [m for m in ms_errors if m < max(ms_errors)]
            if len(ms_errors_next) == 0 :
               return -1, -1, parent_node, -1 ,-1
            max_e = max(ms_errors_next)
            max_index = ms_errors_next.index(max_e)
            return self.divide_set(training_set, max_index, ms_errors_next, f, parent_node, division_method, assign_left, model_selector)
        if model_selector is None:
            node = self.model_selector.add_node(div_node, parent=parent_node)
        else:
            node = model_selector.add_node(div_node, parent=parent_node)
        return t_set_left, t_set_right, node, t_set_left_indexes, t_set_right_indexes

    def local_approximator (self, training_set, f:function, values = None):
        """
        function to create local approximator and fit it to the training set
        :param training_set:
        :param f: approximated function
        :param values: calculated values, optional
        :return: LinearApproximator, local approximator
        """
        if not isinstance(training_set[0], list):
            x_length = len([training_set[0]])
        else:
            x_length = len(training_set[0])
        la_parameters = np.zeros(shape=(x_length + 1)).tolist()
        la = LinearApproximator(la_parameters)
        la.update_parameters(training_set=training_set, f=f, values= values)
        return la

    def evaluate(self, x):
        """
        calculate the approximated value for example x
        :param x: training example
        :return:
        """
        la : LinearApproximator = self.model_selector.select_model(x)
        return la.evaluate(x)

    def get_errors(self, training_set, f: function,  values = None):
        mse = []
        for x in training_set:
            if values is not None:
                y_f = values[training_set.index(x)]
            else:
                y_f = f(x)
            y_a = self.evaluate(x)
            mse.append(np.square(y_f - y_a))
        return mse

    def get_best(self, training_set):
        y_a = []
        for x in training_set:
            y_a.append(self.evaluate(x))
        return min(y_a), y_a.index(min(y_a))


