from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class BaseReconciliation(ABC):
    """
    Base class for reconciliation methodologies.
    """
    @abstractmethod
    def aggregate(self, data: pd.DataFrame, tree) -> pd.DataFrame:
        """
        Aggregate or reconcile data according to the methodology.

        :param data: A DataFrame containing the time series data.
        :param tree: The hierarchical tree structure.
        :return: A DataFrame with reconciled or aggregated data.
        """
        pass


from typing import Dict, Optional
import pandas as pd

class BottomUpReconciliation(BaseReconciliation):
    """
    Bottom-up reconciliation methodology.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the bottom-up reconciler.

        :param weights: A dictionary of weights for aggregating child nodes.
                         Example: {'Region1': 0.6, 'Region2': 0.4}
        """
        self.weights = weights if weights is not None else {}

    def aggregate(self, data: pd.DataFrame, tree) -> pd.DataFrame:
        """
        Aggregate data from the bottom level to the top level using weighted sums.

        :param data: A DataFrame containing the time series data.
        :param tree: The hierarchical tree structure.
        :return: A DataFrame with aggregated data.
        """
        aggregated_data = data.copy()

        # Get all levels of the hierarchy, starting from the bottom
        levels = tree.get_levels()

        # Process each level from bottom to top
        for level in reversed(levels):
            for node in level:
                if not node.is_leaf():
                    children = [child.name for child in node.children]
                    if self.weights:
                        # Weighted sum of child nodes
                        aggregated_data[node.name] = sum(
                            aggregated_data[child] * self.weights.get(child, 1.0) for child in children
                        )
                    else:
                        # Default to simple sum if no weights are provided
                        aggregated_data[node.name] = aggregated_data[children].sum(axis=1)

        return aggregated_data
    

class TopDownReconciliation(BaseReconciliation):
    """
    Top-down reconciliation methodology.
    """
    def __init__(self, method: str = "proportional", weights: Optional[Dict[str, float]] = None):
        """
        Initialize the top-down reconciler.

        :param method: The disaggregation method ("proportional" or "average").
        :param weights: A dictionary of weights for disaggregating parent nodes.
                         Example: {'City1': 0.6, 'City2': 0.4}
        """
        self.method = method
        self.weights = weights if weights is not None else {}

    def aggregate(self, data: pd.DataFrame, tree) -> pd.DataFrame:
        """
        Disaggregate data from the top level to the bottom level using weighted proportions.

        :param data: A DataFrame containing the time series data.
        :param tree: The hierarchical tree structure.
        :return: A DataFrame with disaggregated data.
        """
        disaggregated_data = data.copy()
        
        # Get all levels of the hierarchy, starting from the top
        levels = tree.get_levels()

        # Process each level from top to bottom
        for level in levels:

            for node in level:

                if not node.is_leaf():
                    children = [child.name for child in node.children]

                    if self.weights:
                        # Weighted proportional disaggregation
                        total_weight = sum(self.weights.get(child, 1.0) for child in children)
                        for child in children:
                            disaggregated_data[child] = (
                                disaggregated_data[node.name] * (self.weights.get(child, 1.0) / total_weight)
                            )
                    else:
                        if self.method == "proportional":
                            # This needs weights -> learn?, insert?
                            pass

                        elif self.method == "average":
                            # Equal distribution among child nodes
                            for child in children:
                                disaggregated_data[child] = disaggregated_data[node.name] / len(children)

        return disaggregated_data
    

class MiddleOutReconciliation(BaseReconciliation):
    """
    Middle-out reconciliation methodology.
    """
    def __init__(self, middle_level: str, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the middle-out reconciler.

        :param middle_level: The middle level node to start reconciliation.
        :param weights: A dictionary of weights for aggregating and disaggregating nodes.
                         Example: {'Region1': 0.6, 'Region2': 0.4}
        """
        self.middle_level = middle_level
        self.weights = weights if weights is not None else {}

    def aggregate(self, data: pd.DataFrame, tree) -> pd.DataFrame:
        """
        Reconcile data using the middle-out methodology with weighted aggregation and disaggregation.

        :param data: A DataFrame containing the time series data.
        :param tree: The hierarchical tree structure.
        :return: A DataFrame with reconciled data.
        """
        reconciled_data = data.copy()

        # Bottom-up aggregation from middle level to root
        node = tree.nodes[self.middle_level]
        while node.parent:
            children = [child.name for child in node.parent.children]
            if self.weights:
                # Weighted sum of child nodes
                reconciled_data[node.parent.name] = sum(
                    reconciled_data[child] * self.weights.get(child, 1.0) for child in children
                )
            else:
                # Default to simple sum if no weights are provided
                reconciled_data[node.parent.name] = reconciled_data[children].sum(axis=1)
            node = node.parent

        # Top-down disaggregation from middle level to leaves
        node = tree.nodes[self.middle_level]
        stack = [node]
        while stack:
            current_node = stack.pop()
            if not current_node.is_leaf():
                children = [child.name for child in current_node.children]
                if self.weights:
                    # Weighted proportional disaggregation
                    total_weight = sum(self.weights.get(child, 1.0) for child in children)
                    for child in children:
                        reconciled_data[child.name] = (
                            reconciled_data[current_node.name] * (self.weights.get(child, 1.0) / total_weight)
                        )
                else:
                    # Default to equal distribution if no weights are provided
                    for child in children:
                        reconciled_data[child.name] = reconciled_data[current_node.name] / len(children)
                stack.extend(current_node.children)

        return reconciled_data