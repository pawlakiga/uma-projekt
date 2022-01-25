from LinearScikit import LinearApproximator
import numpy as np
from matplotlib import pyplot as plt
# from LinearExtended import LinearExtended, step_perpendicular, divide_set
from DataGenerator import init_method
from LinearSpline import LinearSpline


def square(x):
    return np.multiply(x, x) +100


def d_square(x):
    return sum(np.multiply(x, x)) * 200 + 10000

def spline_test_1d():
    training_Set = []
    for i in range(5000):
        training_Set.append(init_method([1, 0, 10]))

    xx = np.linspace(0, 10)
    yy = square(xx)
    plt.plot(xx, yy, label='Oryginal')
    ls = LinearSpline()
    ls.approximate(training_set=training_Set, lina=None, max_error=3, step=0.02, f=square)
    ys = []
    for x in xx:
        ys.append(ls.evaluate(x))
    plt.plot(xx, ys, label='Linear spline')
    plt.show()


def spline_test_2d():
    training_Set = []
    for i in range(500):
        training_Set.append(init_method([2, 0, 10]))

    xx = np.linspace(0, 10, num=30)
    yy = np.linspace(0, 10, num=30)
    X, Y = np.meshgrid(xx, yy)
    Z = d_square([X, Y])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # lina = LinearApproximator([0, 0, 0])
    # lina.update_parameters(None, training_Set, d_square)
    # ya = np.zeros(shape=(30, 30))
    # for i in range(30):
    #     for j in range(30):
    #         ya[i][j] = lina.evaluate([X[i][j], Y[i][j]])
    # ax.plot_wireframe(X, Y, ya, color='pink')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('ya')

    ls = LinearSpline()
    ls.approximate(training_set=training_Set, lina=None, max_error=15, step=0.011, f=d_square)
    ys = np.zeros(shape=(30, 30))
    for i in range(30):
        for j in range(30):
            ys[i][j] = ls.evaluate([X[i][j], Y[i][j]])
    ax.plot_wireframe(X, Y, ys, color='blue')
    plt.show()


spline_test_2d()