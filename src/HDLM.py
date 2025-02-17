import numpy
from dlm_model import DLM

class TreeNode:

    def __init__(self, value, name="", parent=None):
        self.value = value
        self.name = name
        self.parent = parent
        self.children = []
        self.weights = []

    def add_child(self, child, weight):
        self.children.append(child)
        child.parent = self
        self.weights.append(weight)

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def print_tree(self):
        # Print the tree structure with names of the nodes
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        print(prefix + self.name)
        if self.children:
            for child in self.children:
                child.print_tree()

    def update(self, y=None):
        # Update children first
        for child in self.children:
            child.update(y)
        
        if self.children:
            # Combine children's states into the parent's state
            weighted_theta = sum(
                w * child.value.predict()[0] for w, child in zip(self.weights, self.children)
            )
            weighted_variance = sum(
                w**2 * child.value.predict()[1] for w, child in zip(self.weights, self.children)
            )
            
            # Update parent's state with the combined state
            self.value.update(weighted_theta)
            self.value.set_variance(weighted_variance)
        else:
            # If no children, update the node's state normally
            if y is not None:
                self.value.update(y)

    def predict(self):
        # Predict the current node's state
        return self.value.predict()
    

class HDLM:

    def __init__(self, root):
        self.root = root

    def update(self, y: list): 
        self._recursive_update(self.root, y)

    def _recursive_update(self, node, y):
        # Update the current node with its data or aggregated children data
        node.update(y)

    def predict(self):
        return self._recursive_predict(self.root)
    
    def _recursive_predict(self, node):
        predictions = []
        predictions.append(node.predict())
        for child in node.children:
            predictions.extend(self._recursive_predict(child))
        return predictions
    

if __name__ == "__main__":

    stock_1_values = [60,62,65,63,59,57,55,58,59]
    stock_2_values = [30,33,35,37,39,36,38,39,40]
    etf_values = [90,91,93,88,89,90,87,85,83]

    weight_stock_1 = 0.5
    weight_stock_2 = 0.5

    F = numpy.array([[1, 1], [0, 1]])
    G = numpy.array([[1, 0]])
    V = numpy.array([[1, 0], [0, 1]])
    W = numpy.array([[1]])

    dlm_stock_1 = DLM(F, G, V, W)
    dlm_stock_2 = DLM(F, G, V, W)
    dlm_etf = DLM(F, G, V, W)

    dlm_stock_1.initialize(numpy.array([[60], [2]]), numpy.array([[1, 0], [0, 1]]))
    dlm_stock_2.initialize(numpy.array([[30], [3]]), numpy.array([[1, 0], [0, 1]]))
    dlm_etf.initialize(numpy.array([[90], [1]]), numpy.array([[1, 0], [0, 1]]))

    stock_1 = TreeNode(dlm_stock_1, "Stock 1")
    stock_2 = TreeNode(dlm_stock_2, "Stock 2")
    etf = TreeNode(dlm_etf, "ETF")

    root = TreeNode(etf, "Root")
    root.add_child(stock_1, weight_stock_1)
    root.add_child(stock_2, weight_stock_2)

    hdlm = HDLM(root)

    for stock_1_value, stock_2_value, etf_value in zip(stock_1_values, stock_2_values, etf_values):
        hdlm.update([stock_1_value, stock_2_value, etf_value])

    results = hdlm.predict()

    # plot the results
    import matplotlib.pyplot as plt

    stock_1_predictions = [result[0][0] for result in results]
    stock_1_prediction_vars = [result[1][0][0] for result in results]

    stock_2_predictions = [result[0][0] for result in results]
    stock_2_prediction_vars = [result[1][0][0] for result in results]

    etf_predictions = [result[0][0] for result in results]
    etf_prediction_vars = [result[1][0][0] for result in results]

    plt.plot(stock_1_values, label="Stock 1 True values")
    plt.plot(stock_1_predictions, label="Stock 1 Predictions")
    plt.plot(stock_2_values, label="Stock 2 True values")
    plt.plot(stock_2_predictions, label="Stock 2 Predictions")
    plt.plot(etf_values, label="ETF True values")
    plt.plot(etf_predictions, label="ETF Predictions")
    plt.legend()

    plt.show()

    


    


