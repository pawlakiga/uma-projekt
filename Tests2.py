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


def test_2d(f: function, training_Set, low_point, high_point, points_no, max_error):
    # low_point = -100
    # high_point = 100
    # training_set_len = 400
    # points_no = 50
    # max_error = 1e-8
    # step = 0.05
    # init_params = [0, 0, 0]
    # #
    # training_Set = []
    # for i in range(training_set_len):
    #     training_Set.append(init_method([2, low_point, high_point]))

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

    ls = LinearSpline()
    ls.approximate(training_set=training_Set, lina=None, max_error=max_error, f=f)
    ys = np.zeros(shape=(points_no, points_no))
    for i in range(points_no):
        for j in range(points_no):
            ys[i][j] = ls.evaluate([X[i][j], Y[i][j]])
    ax.plot_wireframe(X, Y, ys, color='blue')

    for ix in range(0, len(training_Set), 20):
        x = training_Set[ix]
        fx = f(x)
        ax.scatter(x[0],x[1],fx,'r.')
    plt.show()


def test_axes(f:function, training_Set, low_point, high_point, points_no, max_error):

    # training_Set = []
    # for i in range(training_set_len):
    #     training_Set.append(init_method([2, low_point, high_point]))

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

    ls = LinearAxes()
    ls.approximate(training_set=training_Set, max_error=max_error, f=f)
    ys = np.zeros(shape=(points_no, points_no))
    for i in range(points_no):
        for j in range(points_no):
            ys[i][j] = ls.evaluate([X[i][j], Y[i][j]])
    ax.plot_wireframe(X, Y, ys, color='blue')
    for ix in range(0, len(training_Set), 20):
        x = training_Set[ix]
        fx = f(x)
        ax.scatter(x[0], x[1], fx, 'r.')
    plt.show()



# test_axes(f5)
