import numpy as np
import pandas as pd
from numpy.linalg import inv, pinv
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple, Union

from utils import seed_all
seed_all(42)

class Node:
    def __init__(self, name: str, parent: Optional['Node'] = None):
        self.name = name
        self.parent = parent
        self.children: List['Node'] = []
        if parent:
            parent.children.append(self)

class HierarchicalTree:
    def __init__(self, hierarchy: Dict[str, List[str]], method: str = 'mint'):
        """
        hierarchy: dict mapping parent -> list of children
        method: 'mint', 'ols', 'middle_out', or 'regression'
        """
        self.nodes: Dict[str, Node] = {}
        self.root: Optional[Node] = None
        self._build_tree(hierarchy)
        self.method = method
        self.W: Optional[np.ndarray] = None
        self.all_nodes: List[str] = list(self.nodes.keys())
        self.hierarchy = hierarchy

    def _build_tree(self, hierarchy: Dict[str, List[str]]):
        for parent_name, child_names in hierarchy.items():
            parent = self.nodes.setdefault(parent_name, Node(parent_name))
            for child_name in child_names:
                child = self.nodes.setdefault(child_name, Node(child_name))
                if child.parent is None:
                    child.parent = parent
                    parent.children.append(child)
        self.root = next(node for node in self.nodes.values() if node.parent is None)

    def _compute_summing_matrix(self, base_nodes: List[str]) -> np.ndarray:
        N, M = len(self.all_nodes), len(base_nodes)
        S = np.zeros((N, M))
        for j, b in enumerate(base_nodes):
            curr = self.nodes[b]
            while curr:
                i = self.all_nodes.index(curr.name)
                S[i, j] = 1.0
                curr = curr.parent
        return S

    def _fit_regression_weights(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Fit regression coefficients for each parent on its children.
        Returns dict parent -> {child: coef}
        """
        weights = {}
        for parent, children in self.hierarchy.items():
            X = data[children].values
            y = data[parent].values
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            weights[parent] = dict(zip(children, model.coef_))
        return weights

    def compute_W(
                  self,
                  base_nodes: List[str],
                  residuals: Optional[pd.DataFrame] = None,
                  data: Optional[pd.DataFrame] = None
                 ) -> np.ndarray:
        S = self._compute_summing_matrix(base_nodes)
        if self.method == 'ols':
            G = inv(S.T @ S) @ S.T
            W = S @ G
        elif self.method == 'mint':
            if residuals is None:
                raise ValueError("Residuals required for MinT method.")
            Sigma = residuals[base_nodes].cov().values
            print(S.shape)
            print(Sigma.shape)
            print(S)
            Sigma_inv = pinv(Sigma)
            G = inv(S @ Sigma_inv @ S.T) @ (S @ Sigma_inv)
            print(G.shape)
            W = S @ G
        elif self.method == 'middle_out':
            if data is None:
                raise ValueError("Data required for middle_out method.")
            means = data.mean()
            N, M = S.shape
            W = np.zeros((N, M))
            for j, b in enumerate(base_nodes):
                base = self.nodes[b]
                for i, ni in enumerate(self.all_nodes):
                    target = self.nodes[ni]
                    if self._is_ancestor(target, base):
                        path = self._path(target.name, base.name)
                        prod = np.prod([means[p] / means[c] for p, c in path])
                        W[i, j] = prod
                    elif self._is_ancestor(base, target):
                        path = self._path(base.name, target.name)
                        prod = np.prod([means[c] / means[p] for p, c in path])
                        W[i, j] = prod
            # clip negatives, normalize
            W = np.clip(W, 0, None)
            row_sums = W.sum(axis=1, keepdims=True)
            W = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums!=0)
        elif self.method == 'regression':
            if data is None:
                raise ValueError("Data required for regression method.")
            # Fit upward regression weights (parent ~ children)
            up_weights = self._fit_regression_weights(data)
            # Fit downward regression weights (children ~ parent, multi-output)
            down_weights = {}
            for parent, childs in self.hierarchy.items():
                X = data[[parent]].values  # shape (T,1)
                Y = data[childs].values    # shape (T, len(childs))
                model = LinearRegression(fit_intercept=False)
                model.fit(X, Y)
                # coef_ shape (len(childs), 1)
                for coef, child in zip(model.coef_.flatten(), childs):
                    down_weights.setdefault(parent, {})[child] = coef

            N, M = S.shape
            W = np.zeros((N, M))
            for j, b in enumerate(base_nodes):
                leaf_node = self.nodes[b]
                for i, n in enumerate(self.all_nodes):
                    if n == b:
                        W[i, j] = 1.0
                    else:
                        target_node = self.nodes[n]
                        if self._is_ancestor(target_node, leaf_node):
                            # going up: multiply parent-on-children betas
                            path = self._path(n, b)
                            prod = 1.0
                            for p, c in path:
                                prod *= up_weights[p][c]
                            W[i, j] = prod
                        elif self._is_ancestor(leaf_node, target_node):
                            # going down: multiply child-on-parent gammas
                            path = self._path(b, n)
                            prod = 1.0
                            for p, c in path:
                                prod *= down_weights[p][c]
                            W[i, j] = prod
                        # else stays 0
        else:
            raise ValueError(f"Unknown method {self.method}")
        self.W = W
        return W
    
    def reconcile(self,
                  base_forecasts: Union[pd.Series, np.ndarray, Dict[str, float]],
                  base_nodes: List[str],
                  residuals: Optional[pd.DataFrame] = None,
                  data: Optional[pd.DataFrame] = None
                 ) -> pd.Series:
        W = self.compute_W(base_nodes, residuals, data)
        if isinstance(base_forecasts, pd.Series):
            b = base_forecasts.reindex(base_nodes).values
        elif isinstance(base_forecasts, dict):
            b = np.array([base_forecasts[n] for n in base_nodes])
        else:
            b = np.asarray(base_forecasts)
        rec_vals = W @ b
        return pd.Series(rec_vals, index=self.all_nodes)

    def _is_ancestor(self, anc: Node, desc: Node) -> bool:
        curr = desc
        while curr:
            if curr is anc:
                return True
            curr = curr.parent
        return False

    def _path(self, start: str, end: str) -> List[Tuple[str, str]]:
        if start == end:
            return []
        node = self.nodes[start]
        for child in node.children:
            sub = self._path(child.name, end)
            if sub is not None:
                return [(start, child.name)] + sub
        return None

if __name__ == "__main__":
    # --- Synthetic data generation ---
    np.random.seed(42)
    n = 100
    t = np.arange(n)

    # Leaf series: different functions + noise
    noise_std = 0.5
    y_AA = 0.1*t + 2*np.sin(2*np.pi*t/20) + np.random.normal(0, noise_std, n)
    y_AB = -0.05*t + np.cos(2*np.pi*t/15) + np.random.normal(0, noise_std, n)
    y_BA = 0.2*np.log1p(t) + np.random.normal(0, noise_std, n)
    y_BB = 3*np.exp(-t/50) + np.random.normal(0, noise_std, n)

    # True noise components for residuals
    forecasts_bottom = pd.DataFrame({
        'AA': y_AA - (0.1*t + 2*np.sin(2*np.pi*t/20)),
        'AB': y_AB - (-0.05*t + np.cos(2*np.pi*t/15)),
        'BA': y_BA - (0.2*np.log1p(t)),
        'BB': y_BB - (3*np.exp(-t/50))
    })

    forecasts_middle = pd.DataFrame({
        'A': y_AA + y_AB - (0.1*t + 2*np.sin(2*np.pi*t/20) - 0.05*t + np.cos(2*np.pi*t/15)),
        'B': y_BA + y_BB - (0.2*np.log1p(t) + 3*np.exp(-t/50))
    })

    # Build full data with parents = sums + small noise
    data = pd.DataFrame({
        'AA': y_AA, 'AB': y_AB, 'BA': y_BA, 'BB': y_BB
    })
    data['A'] = (data['AA'] + data['AB'] + np.random.normal(0, 0.1, n))*4
    data['B'] = (data['BA'] + data['BB'] + np.random.normal(0, 0.1, n))/3
    data['O'] = data['A'] + data['B'] + np.random.normal(0, 0.1, n)

    # Define hierarchy
    hierarchy = {'O': ['A', 'B'], 'A': ['AA', 'AB'], 'B': ['BA', 'BB']}
    leaves = ['AA', 'AB', 'BA', 'BB']
    middle = ['A', 'B']

    # Test each method
    methods = ['regression']
    for method in methods:
        tree = HierarchicalTree(hierarchy, method=method)
        if method == 'ols':
            W_ols = tree.compute_W(base_nodes=leaves)
        
        elif method == 'middle_out':
            W_mo = tree.compute_W(base_nodes=leaves, data=data)

        elif method == 'regression':
            W_reg = tree.compute_W(base_nodes=leaves, data=data)
        
        print(f"\n=== W matrix for {method.upper()} ===")
    
    # We now calculate the residuals using the W_reg for the whole data

    # Set an array with the data from the leaves
    leaves_data = data[leaves].values                # shape (100, 4)
    # build full_data: T×7
    all_nodes = tree.all_nodes                       # ['O','A','B','AA','AB','BA','BB']
    full_data = data[all_nodes].values               # shape (100, 7)

    # predicted full series = leaves_data @ W_reg.T  (100×4 @ 4×7 → 100×7)
    preds = leaves_data @ W_reg.T

    # residuals = actual – predicted
    resids_all = (full_data - preds) + np.random.normal(0, 0.1, full_data.shape)
    resid_df_all = pd.DataFrame(resids_all, columns=all_nodes, index=data.index)
    tree = HierarchicalTree(hierarchy, method="mint")
    W_mint_full = tree.compute_W(base_nodes=leaves, residuals=forecasts_bottom)

    # Display the MinT reconciliation matrix
    df_W_mint = pd.DataFrame(W_mint_full, index=all_nodes, columns=all_nodes)
    print("=== Full MinT W Matrix (All Nodes as Base) ===")
    print(df_W_mint)

        