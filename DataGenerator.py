import numpy as np


def init_method(init_args):
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
