from sympy.core import function
from ModelsTree import DivisionNode, ModelSelector
from LinearSpline import LinearSpline
from LinearScikit import LinearApproximator
import numpy as np

class LinearAxes (LinearSpline):
    def __init__(self):
        LinearSpline.__init__(self)
        self.ax_model_selectors =[]

    def approximate(self, f: function, training_set, la = [],  max_error = 10, parent_nodes = []):
        axes_no = len(training_set[0])
        values = self.get_func_values(training_set, f)
        model_values = np.zeros(shape = (len(training_set)))
        ax_approximators = []
        for ax in range(axes_no):
            training_set_ax = [row[ax] for row in training_set]
            ax_approximators.append(self.local_approximator(training_set_ax,f,np.divide(values,axes_no)))
            for i in range(len(training_set)):
                model_values[i] += ax_approximators[ax].evaluate(training_set_ax[i])
        ms_errors = np.square(model_values - values).tolist()
        print(ms_errors)
        if np.mean(ms_errors) > max_error:
            for ax in range(axes_no):
                training_set_ax = [row[ax] for row in training_set]
                self.approximate_axis(ax = ax, training_set_ax=training_set_ax, lina=ax_approximators[ax], max_error=max_error, values=np.multiply(values,1/axes_no).tolist(), f =f, ax_wage=1/axes_no)

    def approximate_axis(self, ax: int, training_set_ax,lina : LinearApproximator,
                         max_error, values, f : function, ax_wage, div_node = ""):

        if len(self.ax_model_selectors) < ax + 1:
            self.ax_model_selectors.insert(ax, ModelSelector())
        model_values = []
        for i in range(len(training_set_ax)):
            model_values.append(lina.evaluate(training_set_ax[i]))
        # ax_errors = np.square(np.multiply(values, ax_wage) - model_values).tolist()
        ax_errors = np.square(np.subtract(values, model_values)).tolist()
        if np.mean(ax_errors) > max_error * ax_wage:
            max_ms = max(ax_errors)
            max_error_index = ax_errors.index(max_ms)
            t_set_left, t_set_right, d_node, left_indexes, right_indexes = \
                self.divide_set(training_set_ax, max_error_index,ax_errors ,f, div_node, model_selector=self.ax_model_selectors[ax])
            if t_set_left == -1 :
                self.ax_model_selectors[ax].add_node(lina, parent=d_node)
                return
            left_values, right_values = self.divide_values(left_indexes, right_indexes, values)
            la_left = self.local_approximator(t_set_left, f, left_values)
            self.approximate_axis(ax = ax, f = f, training_set_ax=t_set_left, lina=la_left, max_error=max_error,div_node=d_node, ax_wage=ax_wage, values=left_values)

            la_right = self.local_approximator(t_set_right, f, right_values)
            self.approximate_axis(ax = ax, f = f, training_set_ax=t_set_right, lina=la_right, max_error=max_error,div_node=d_node, ax_wage=ax_wage, values=right_values)
        else:
            self.ax_model_selectors[ax].add_node(lina, parent=div_node)


    def evaluate(self, x):
        y = 0
        for ax in range(len(x)):
            model_selector : ModelSelector = self.ax_model_selectors[ax]
            la : LinearApproximator = model_selector.select_model(x[ax])
            y += la.evaluate(x[ax])
        return y

    def get_func_values(self, training_set, f: function):
        y = []
        for x in training_set:
            y.append(f(x))
        return y

    def get_errors(self, training_set, f : function):
        y_f = self.get_func_values(self,training_set,f)
        y_a = []
        for x in training_set:
            y_a.append(self.evaluate(x))

    def divide_values(self, left_indexes, right_indexes, values):
        left_values = [values[i_l] for i_l in left_indexes]
        right_values = [values[i_r] for i_r in right_indexes]
        return left_values, right_values

