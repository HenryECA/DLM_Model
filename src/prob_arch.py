import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, pinv

# Simplified Node/Tree classes without type hints
class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        if parent:
            parent.children.append(self)

class HierarchicalTree:
    def __init__(self, hierarchy):
        self.nodes = {}
        self.root = None
        self._build_tree(hierarchy)

    def _build_tree(self, hierarchy):
        for p, childs in hierarchy.items():
            parent = self.nodes.setdefault(p, Node(p))
            for c in childs:
                child = self.nodes.setdefault(c, Node(c))
                if child.parent is None:
                    child.parent = parent
                    parent.children.append(child)
        self.root = next(n for n in self.nodes.values() if n.parent is None)

class ProbabilisticHierarchicalTree:
    def __init__(self, hierarchy):
        self.det = HierarchicalTree(hierarchy)

    def _compute_summing_matrix(self, base_nodes):
        all_nodes = list(self.det.nodes.keys())
        N, M = len(all_nodes), len(base_nodes)
        S = np.zeros((N, M))
        for j, b in enumerate(base_nodes):
            curr = self.det.nodes[b]
            while curr:
                S[all_nodes.index(curr.name), j] = 1.0
                curr = curr.parent
        return S, all_nodes

    def compute_reconciliation_matrix(self, residuals, base_nodes):
            S, all_nodes = self._compute_summing_matrix(base_nodes)

            print(S)
            
            N, M = S.shape

            if set(base_nodes) == set(all_nodes):
                # full MinT using Sigma_b
                Sigma = residuals[base_nodes].cov().values  # (M×M)
                Sigma_inv = pinv(Sigma)                    # (M×M)
                G = inv(S.T @ Sigma_inv @ S) @ (S.T @ Sigma_inv)  # (M×N)
                W = S @ G                                     # (N×M)
            else:
                # OLS‐MinT: assume Sigma_b = I
                G = inv(S.T @ S)      # (M×M)
                W = S @ G             # (N×M)
            
            # now W.shape == (N, M)
            return W, all_nodes

    def simulate(self, point_forecasts, residuals, data, n_simulations=1000):
        base_nodes = list(point_forecasts.index)
        W, all_nodes = self.compute_reconciliation_matrix(residuals, base_nodes)
        M = len(base_nodes)
        N = len(all_nodes)

        # 1) draw base errors: shape (n_simulations, M)
        cov_base    = residuals[base_nodes].cov().values      # (M×M)
        errors_base = np.random.multivariate_normal(
                          mean=np.zeros(M),
                          cov=cov_base,
                          size=n_simulations
                      )                                  # → (n_simulations, M)

        # 2) build base_draws matrix (n_simulations×M)
        point_vals  = point_forecasts.values                   # shape (M,)
        base_draws  = errors_base + point_vals[np.newaxis, :]  # broadcast to (n_simulations, M)

        # Sanity check:
        assert base_draws.shape == (n_simulations, M)
        assert W.shape        == (N, M)

        # 3) reconcile into all N nodes: shape (n_simulations, N)
        sims = base_draws @ W.T
        assert sims.shape == (n_simulations, N)

        # 4) top-down split for any non-base descendants (same as before)...
        means = data[all_nodes].mean()
        pi = {}
        for p_node in self.det.nodes.values():
            for c in p_node.children:
                pi[(p_node.name, c.name)] = means[c.name] / means[p_node.name]

        levels = []
        curr = [self.det.root]
        while curr:
            levels.append([n.name for n in curr])
            curr = [child for n in curr for child in n.children]

        for lvl in levels[1:]:
            for n in lvl:
                if n not in base_nodes:
                    parent = self.det.nodes[n].parent.name
                    sims[:, all_nodes.index(n)] = sims[:, all_nodes.index(parent)] * pi[(parent, n)]

        return sims, all_nodes



if __name__ == "__main__":
    hierarchy = {
        "O": ["A", "B"],
        "A": ["A1", "A2", "A3"],
        "B": ["B1", "B2"],
        "B1": ["B11", "B12"]
    }
    df = pd.DataFrame({
        "O": [1,2,3,4],
        "A": [2,3,4,5],
        "B": [3,4,5,6],
        "A1":[1,3,5,7],
        "A2":[2,4,6,8],
        "A3":[1,4,7,10],
        "B1":[3,5,7,9],
        "B2":[4,6,8,10],
        "B11":[1,2,3,5],
        "B12":[2,3,4,6]
    })

    base_nodes = ["A", "B1", "B2"]
    residuals = df[base_nodes] - df[base_nodes].mean()
    point_forecasts = df[base_nodes].iloc[-1]

    pht = ProbabilisticHierarchicalTree(hierarchy)
    sims, nodes = pht.simulate(point_forecasts, residuals, df, n_simulations=500)

    # plot 5%–95% intervals
    q05 = np.percentile(sims, 5, axis=0)
    q95 = np.percentile(sims, 95, axis=0)
    plt.figure(figsize=(8,4))
    plt.errorbar(nodes, q05, yerr=q95-q05, fmt='o', capsize=5)
    plt.xticks(rotation=45)
    plt.title("Quantiles (5%–95%) with Top-Down Split")
    plt.tight_layout()
    plt.show()
