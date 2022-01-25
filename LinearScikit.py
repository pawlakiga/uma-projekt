from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
import numpy as np
from sympy.core import function
from cec2017 import functions
from matplotlib import pyplot as plt


class LinearApproximator:
    parameters: list

    def __init__(self, initial_parameters):
        self.lin_reg: LinearRegression = None
        self.parameters = initial_parameters

    def evaluate(self, x):
        """
        calculate the value of the linear function for a given argument
        :param x: argument
        :return: value
        """
        X = [x]
        if not isinstance(x,list) and not isinstance(x,np.ndarray):
            X = [X]
        y = self.lin_reg.predict(X)[0]
        return y

    def update_parameters(self, training_set, f: function, values = None):
        """
        fit linear approximator to training set and the values of a given function
        :param training_set:
        :param f: approximated function
        :param values : if given we use them instead of calculating the function
        """
        if values is not None:
            y = values
        else:
            y = []
            for x in training_set:
                y.append(f(x))
        mlr = LinearRegression()
        self.lin_reg = mlr
        if len(self.parameters) == 2:
            mlr.fit(np.transpose([training_set]),y)
        else:
            mlr.fit(training_set, y)

        for i in range(len(mlr.coef_)):
            self.parameters[i] = mlr.coef_[i]
        self.parameters[-1] = mlr.intercept_

    def get_errors(self, training_set, f: function, values = None):
        """
        calculate mean square error for every point in the training set
        :param training_set:
        :param f: approximated function
        :param values : if given we use them instead of calculating the function
        :return: errors vector
        """
        mse = []
        for x in training_set:
            if values is not None:
                y_f = values[training_set.index(x)]
            else:
                y_f = f(x)
            y_a = self.evaluate(x)
            mse.append(np.square(y_f - y_a))
        return mse
