import numpy as np
from sympy.core import function
from matplotlib import pyplot as plt
from LinearSpline import LinearSpline


def init_method(init_args = [1,-10,10]):
    """
    generates a random point with a given dimension
    :param init_args: 0 - dimensions, 1 - minimum, 2 - maximum
    :return: generated point as list
    """
    dimensions = init_args[0]
    x_min = init_args[1]
    x_max = init_args[2]

    if dimensions == 1:
        return np.random.uniform(low=x_min, high=x_max, size=dimensions)[0]
    return np.random.uniform(low=x_min, high=x_max, size=dimensions).tolist()

def draw_figure(f : function, training_set, ls : LinearSpline, low_point, high_point, max_error):
    points_no = 50
    xx = np.linspace(low_point, high_point, num=points_no)
    yy = np.linspace(low_point, high_point, num=points_no)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros(shape=(points_no, points_no))
    for i in range(points_no):
        for j in range(points_no):
            Z[i][j] = f([X[i][j], Y[i][j]])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z, color='green', label = 'Funkcja oryginalna')

    ys = np.zeros(shape=(points_no, points_no))
    for i in range(points_no):
        for j in range(points_no):
            ys[i][j] = ls.evaluate([X[i][j], Y[i][j]])
    ax.plot_wireframe(X, Y, ys, color='blue', label = 'Aproksymator')

    for ix in range(0, len(training_set), 100):
        x = training_set[ix]
        fx = f(x)
        ax.scatter(x[0],x[1],fx, 'r.')
    ax.legend()
    plt.title("Wyniki działania aproksymatora dla funkcji {fun} błędu = {max_error}".format(fun = f.__name__, max_error = max_error))
    # plt.savefig()
    plt.show()
