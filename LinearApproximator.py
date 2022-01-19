import numpy as np
from sympy.core import function
from cec2017 import functions
from matplotlib import pyplot as plt

class LinearApproximator:
    parameters: list

    def __init__(self, initial_parameters):
        self.parameters = initial_parameters

    def evaluate(self, x):
        y = np.sum(np.multiply(x, self.parameters[0:-1])) + self.parameters[-1]
        return y

    def update_parameters(self, approx_method: function, training_set, f: function, step):
        for x in training_set:
            self.parameters = self.parameters + self.gradient_loss(x, f, step)
        if self.parameters[0] < 1 :
            print(x)
            print("function:")
            print(f(x))
            print("approximator")
            print(self.evaluate(x))
            print("==============")

    def gradient_loss(self, x, f: function, step):
        if len(self.parameters) > 2:
            a = x.copy()#.tolist()
            a.append(1)
        else:
            a = [x, 1]
        grad = np.multiply((f(x) - self.evaluate(x)) * (-1), a)
        return - np.multiply(step, grad)

    def get_errors(self, training_set, f: function):
        mse = []
        for x in training_set:
            y_f = f(x)
            y_a = self.evaluate(x)
            mse.append(np.square(y_f - y_a))
        return mse

def square(x):
    return np.multiply(x,x)

def d_square(x):
    return sum(np.multiply(x,x))


