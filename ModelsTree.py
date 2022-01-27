

from sympy.core import function

from LinearScikit import LinearApproximator
from treelib import Tree, Node
import numpy as np


class DivisionNode:
    def __init__(self, parameters):
        self.parameters = parameters
        self.index = 0
        self.left = None
        self.right = None


class ModelSelector:
    def __init__(self):
        self.tree: Tree = Tree()

    def add_node(self, new_node: DivisionNode, parent : str):
        """
        add a new node to the binary tree holding domain division parameters in nodes and local approximators in leaves
        :param new_node: node to be added
        :param parent: identifier of parent node of the new one
        :return: identifier of the new node
        """
        parent_node = self.tree.get_node(parent)
        #  first node
        if parent == "":
            return self.tree.create_node(data=new_node).identifier
        # trying to add a child to a leaf
        if parent_node.data.__class__ == LinearApproximator:
            return -1
        node = self.tree.create_node(parent=parent, data=new_node)
        # print("Adding new node to parent: {parent_node}, with parameters {parameters}".format(parent_node= parent, parameters = new_node.parameters))
        return node.identifier

    def select_model(self, x):
        """
        select an appropriate local approximator for a given argument
        :param x: function argument for which we want to select a model
        :return: selected LinearApproximator
        """
        current_node: Node = self.tree.get_node(self.tree.root)
        while current_node.data.__class__ != LinearApproximator:
            if surface_side(t_example=x, div_surface_coordinates=current_node.data.parameters):
                current_node = self.tree.children(current_node.identifier)[0]
            else:
                current_node = self.tree.children(current_node.identifier)[1]
        return current_node.data


def step_perpendicular(training_set: list, point_index: int):
    """
    calculate parameters of a surface that will divide the domain in a given point
    :param training_set:
    :param point_index: index in training set of the division point
    :return: surface coordinates
    """
    previous_point = training_set[point_index - 1]
    break_point = training_set[point_index]

    step_vector = np.subtract(break_point, previous_point)
    surface_coordinates = []
    if not isinstance(step_vector, list) and not isinstance(step_vector, np.ndarray):
        surface_coordinates.append(step_vector)
    else:
        for cord in step_vector:
            surface_coordinates.append(cord)
    surface_coordinates.append(-(np.sum(np.multiply(step_vector, break_point))))
    # print("Creating division surface on points: {x1}, {x2}".format(x1 = previous_point, x2 = break_point))
    return surface_coordinates


def surface_side(t_example, div_surface_coordinates):
    """
    used to divide training set or select a model, determine on which side of the surface a given point is on
    :param t_example: training example
    :param div_surface_coordinates: coordinates of the surface
    :return: true if training set is below the surface or false if above
    """
    return np.sum(np.multiply(div_surface_coordinates[:-1], t_example)) + div_surface_coordinates[-1] <= 0
