import numpy
from models.dlm_model import DLM

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class Node:
    """
    Represents a node in the hierarchical tree structure.
    """
    def __init__(self, name: str, parent: Optional['Node'] = None):
        self.name = name
        self.parent = parent
        self.children: List['Node'] = []
        if parent:
            parent.add_child(self)

    def add_child(self, child: 'Node'):
        """Add a child node to the current node."""
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if the node is a leaf (bottom-level node)."""
        return len(self.children) == 0

    def __repr__(self):
        return f"Node(name = {self.name}, children = {[c.name for c in self.children]})"
    
    def print_tree(self, level=0):
        """Show tree structure"""

        print("\t"*level, self.name)
        for child in self.children:
            child.print_tree(level+1)

class HierarchicalTree:
    """
    Represents the hierarchical tree structure for the time series.
    """
    def __init__(self, hierarchy: Dict[str, List[str]]):
        """
        Initialize the hierarchical tree.

        :param hierarchy: A dictionary representing the hierarchy.
                          Example: {'Total': ['Region1', 'Region2'], 'Region1': ['City1', 'City2']}
        """
        self.nodes: Dict[str, Node] = {}
        self.root: Optional[Node] = None
        self._build_tree(hierarchy)

    def _build_tree(self, hierarchy: Dict[str, List[str]]):
        """Build the tree from the hierarchy dictionary."""
        for parent, children in hierarchy.items():
            if parent not in self.nodes:
                self.nodes[parent] = Node(parent)
            for child in children:
                if child not in self.nodes:
                    self.nodes[child] = Node(child, self.nodes[parent])

        # Set the root node (node with no parent)
        self.root = next(node for node in self.nodes.values() if node.parent is None)

    def get_leaves(self) -> List[Node]:
        """Get all leaf nodes (bottom-level nodes)."""
        return [node for node in self.nodes.values() if node.is_leaf()]

    def get_levels(self) -> List[List[Node]]:
        """Get all levels of the hierarchy, starting from the root."""
        levels = []
        current_level = [self.root]

        while current_level:
            levels.append(current_level)
            next_level = []
            for node in current_level:
                next_level.extend(node.children)
            current_level = next_level

        return levels
    

if __name__ == "__main__":

    pass

    


    


