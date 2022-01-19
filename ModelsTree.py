from sympy.core import function

from LinearApproximator import LinearApproximator
from LinearExtended import surface_side
from treelib import Tree, Node


class DivisionNode:
    def __init__(self, parameters):
        self.parameters = parameters
        self.index = 0
        self.left = None
        self.right = None


class ModelSelector:
    def __init__(self):
        self.tree: Tree = Tree()

    def add_node(self, new_node, parent):
        parent_node = self.tree.get_node(parent)
        if parent == "":
            return self.tree.create_node( data = new_node).identifier
        if parent_node.data.__class__ == LinearApproximator:
            return -1
        node = self.tree.create_node(parent = parent, data = new_node)
        return node.identifier

    def select_model(self, x):
        current_node: Node = self.tree.get_node(self.tree.root)
        while current_node.data.__class__ != LinearApproximator:
            if surface_side(t_example=x, div_surface_coordinates=current_node.data.parameters):
                current_node = self.tree.children(current_node.identifier)[0]
            else:
                current_node = self.tree.children(current_node.identifier)[1]
        return current_node.data
