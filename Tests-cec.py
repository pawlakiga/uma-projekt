from sympy.core import function

from cec2017.functions import f1, f2, f3, f4, f5, f10
import numpy as np
from matplotlib import pyplot as plt
# from LinearExtended import LinearExtended, step_perpendicular, divide_set
from DataGenerator import init_method
from LinearSpline import LinearSpline
from LinearAxes import LinearAxes

def d_cube(x):
    return sum(np.multiply(x, np.multiply(x, x))) * 2 + 10


def d_sin(x):
    sin = 0
    for xx in x:
        sin += np.sin(xx) + 15
    return sin


def test_2d(f: function):
    low_point = -10
    high_point = 10
    training_set_len = 400
    points_no = 50
    max_error = 0.003
    step = 0.05
    init_params = [0, 0, 0]

    training_Set = []
    for i in range(training_set_len):
        training_Set.append(init_method([2, low_point, high_point]))

    xx = np.linspace(low_point, high_point, num=points_no)
    yy = np.linspace(low_point, high_point, num=points_no)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros(shape=(points_no, points_no))
    for i in range(points_no):
        for j in range(points_no):
            Z[i][j] = f([X[i][j], Y[i][j]])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # lina = LinearApproximator([0, 0, 0])
    # lina.update_parameters(None, training_Set, f, step)
    # ya = np.zeros(shape=(points_no,points_no))
    # for i in range(points_no):
    #     for j in range(points_no):
    #         ya[i][j] = lina.evaluate([X[i][j], Y[i][j]])
    # ax.plot_wireframe(X, Y, ya, color='pink')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('ya')

    ls = LinearSpline()
    ls.approximate(training_set=training_Set, lina=None, max_error=max_error, step=step, f=f)
    ys = np.zeros(shape=(points_no, points_no))
    for i in range(points_no):
        for j in range(points_no):
            ys[i][j] = ls.evaluate([X[i][j], Y[i][j]])
    ax.plot_wireframe(X, Y, ys, color='blue')
    plt.show()


def test_axes(f:function):
    low_point = -10
    high_point = 10
    training_set_len = 400
    points_no = 50
    max_error = 0.003
    step = 0.05
    init_params = [0, 0, 0]

    training_Set = []
    for i in range(training_set_len):
        training_Set.append(init_method([2, low_point, high_point]))

    xx = np.linspace(low_point, high_point, num=points_no)
    yy = np.linspace(low_point, high_point, num=points_no)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros(shape=(points_no, points_no))
    for i in range(points_no):
        for j in range(points_no):
            Z[i][j] = f([X[i][j], Y[i][j]])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # lina = LinearApproximator([0, 0, 0])
    # lina.update_parameters(None, training_Set, f, step)
    # ya = np.zeros(shape=(points_no,points_no))
    # for i in range(points_no):
    #     for j in range(points_no):
    #         ya[i][j] = lina.evaluate([X[i][j], Y[i][j]])
    # ax.plot_wireframe(X, Y, ya, color='pink')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('ya')

    ls = LinearSpline()
    ls.approximate(training_set=training_Set, max_error=max_error, f=f, lina = None, step = 0)
    ys = np.zeros(shape=(points_no, points_no))
    for i in range(points_no):
        for j in range(points_no):
            ys[i][j] = ls.evaluate([X[i][j], Y[i][j]])
    ax.plot_wireframe(X, Y, ys, color='blue')
    plt.show()



test_axes(f10)
