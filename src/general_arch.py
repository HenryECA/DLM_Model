import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from numpy.linalg import inv, pinv
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf


class Node:
    def __init__(self, name: str):
        self.name = name
        self.parent: Optional['Node'] = None
        self.children: List['Node'] = []


class HierarchicalTree:
    def __init__(self, hierarchy: Dict[str, List[str]]):
        """
        hierarchy: dict mapping parent_name -> list of child_names
        """
        self.nodes: Dict[str, Node] = {}
        self._build_tree(hierarchy)
        # identify bottom-level base nodes (no children)
        self.base_nodes: List[str] = [name for name, node in self.nodes.items() if not node.children]
        # stable ordering of all node names
        self.all_nodes: List[str] = list(self.nodes.keys())
        self.hierarchy = hierarchy

    def _build_tree(self, hierarchy: Dict[str, List[str]]):
        for parent_name, child_names in hierarchy.items():
            parent = self.nodes.setdefault(parent_name, Node(parent_name))
            for cname in child_names:
                child = self.nodes.setdefault(cname, Node(cname))
                child.parent = parent
                parent.children.append(child)

    def get_summing_matrix(self) -> np.ndarray:
        """
        Build the summing matrix S of shape (N, M) where:
        - N = total number of nodes
        - M = number of bottom-level (base) series
        S[i,j] = 1 if node i is on the path from base node j up to the root
        """
        base_nodes = self.base_nodes
        N, M = len(self.all_nodes), len(base_nodes)
        S = np.zeros((N, M), dtype=float)
        for j, bn in enumerate(base_nodes):
            curr = self.nodes[bn]
            while curr is not None:
                i = self.all_nodes.index(curr.name)
                S[i, j] = 1.0
                curr = curr.parent
        return S

    def _is_ancestor(self, anc: Node, desc: Node) -> bool:
        """True if anc is on the path from desc up to the root."""
        curr = desc
        while curr is not None:
            if curr is anc:
                return True
            curr = curr.parent
        return False

    def _path(self, start: str, end: str) -> Optional[List[Tuple[str, str]]]:
        """Find path as list of (parent, child) from start down to end."""
        if start == end:
            return []
        node = self.nodes[start]
        for child in node.children:
            sub = self._path(child.name, end)
            if sub is not None:
                return [(start, child.name)] + sub
        return None

    def _fit_regression_weights(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Fit regressions parent ~ children (no intercept),
        returning for each parent a dict of child -> beta.
        """
        weights: Dict[str, Dict[str, float]] = {}
        for parent, childs in self.hierarchy.items():
            X = data[childs].values
            y = data[parent].values
            # Drop any rows with NaN values in X or y
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            if X.shape[0] == 0 or X.shape[1] == 0:
                raise ValueError(f"No valid data for parent {parent} and children {childs}")
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            weights[parent] = dict(zip(childs, model.coef_))
        return weights

    def get_P_classic(self) -> np.ndarray:
        """
        Middle-out classical reconciliation:
        Marks ancestors of each base node with 1 and distributes base node value down equally.
        """
        base_nodes = self.base_nodes
        M = len(base_nodes)
        N = len(self.all_nodes)
        P = np.zeros((M, N), dtype=float)

        for j, bn in enumerate(base_nodes):
            leaf = self.nodes[bn]
            # bottom-up: mark ancestors
            curr = leaf
            while curr is not None:
                idx = self.all_nodes.index(curr.name)
                P[j, idx] = 1.0
                curr = curr.parent
            # top-down equal share under bn
            for i, name in enumerate(self.all_nodes):
                node_i = self.nodes[name]
                if self._is_ancestor(leaf, node_i) and name != bn:
                    path = self._path(bn, name)
                    weight = 1.0
                    for p_name, c_name in path:
                        p_node = self.nodes[p_name]
                        weight /= len(p_node.children)
                    P[j, i] = weight
        return P

    def get_P_MinT(
        self,
        method: str,
        residuals: Optional[pd.DataFrame] = None,
        initial: Optional[str] = "classic", 
        k_h: int = 1, 
        S: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the MinT reconciliation matrix P for hierarchical forecasts.

        method: 'mint_sample', 'mint_ols' 'mint_shrink', 'mint_shrink_lw', 'wls_v', 'wls_s'
        initial: None, 'zeroes', 'classic', 'regression'
        """
        # apply initial reconciliation if requested

        if method not in ('mint_ols', 'wls_s') and residuals.shape[1] != len(self.all_nodes):
            print("Residuals shape ", residuals.shape)
            raise ValueError("Residuals length does not match the number of nodes in the hierarchy")

        # select covariance
        if method == 'mint_sample':
            residuals_sub = residuals[self.all_nodes]  # N×m
            base_Sigma   = k_h * residuals_sub.cov().values
            m            = base_Sigma.shape[0]
            trace        = np.trace(base_Sigma)

            Sigma = base_Sigma.copy()
            
            # 2) try ridge alphas from 1e-6 up to 5e-2 until cond(Sigma)<1e5
            alphas = np.concatenate([
                np.logspace(-6, -3, num=10),    # from 1e-6 → 1e-3
                np.linspace(1e-3, 5e-2, num=20)  # from 1e-3 → 5e-2
            ])
            
            for alpha in alphas:
                eps   = alpha * (trace / m)
                Sigma = base_Sigma + eps * np.eye(m)
                if np.linalg.cond(Sigma) < 1e5:
                    # found a stable Σ
                    break
            else:
                # if we never broke, use the max α
                alpha = alphas[-1]
                eps   = alpha * (trace / m)
                Sigma = base_Sigma + eps * np.eye(m)

            
        elif method == 'mint_ols':
            Sigma = self.W_ols(k_h)

        elif method == 'mint_shrink':
            R = residuals[self.all_nodes]
            T, m = R.shape

            # 1) sample cov & cor
            W1 = R.cov().values
            R1 = R.corr().values
            upper_i, upper_j = np.triu_indices(m, k=1)
            r_vec = R1[upper_i, upper_j]

            # 2) analytic shrinkage intensity
            var_r_vec = ((1 - r_vec**2)**2) / (T - 1)
            num, den = var_r_vec.sum(), (r_vec**2).sum()
            λ = np.clip(num/den if den > 0 else 1.0, 0.0, 1.0)

            # 3) constant-corr target
            variances = np.diag(W1)
            avg_r     = r_vec.mean() if r_vec.size > 0 else 0.0
            T_mat     = np.outer(np.sqrt(variances), np.sqrt(variances)) * avg_r
            np.fill_diagonal(T_mat, variances)

            # 4) form shrinked cov
            W1_star = λ * T_mat + (1 - λ) * W1
            base_Sigma = k_h * W1_star

            Sigma = base_Sigma.copy()

            # 5) dynamic ridge until cond<1e5
            trace   = np.trace(base_Sigma)
            alphas  = np.concatenate([
                np.logspace(-6, -3, num=10),
                np.linspace(1e-3, 5e-2, num=20)
            ])

            for alpha in alphas:
                eps   = alpha * (trace / m)
                Sigma = base_Sigma + eps * np.eye(m)
                if np.linalg.cond(Sigma) < 1e5:
                    break
            else:
                # fallback to the largest alpha if none worked
                alpha = alphas[-1]
                eps   = alpha * (trace / m)
                Sigma = base_Sigma + eps * np.eye(m)

        elif method == 'mint_shrink_lw':
            R = residuals[self.all_nodes]  # shape = (T, m)
            lw = LedoitWolf(store_precision=False).fit(R.values)
            Sigma = k_h * lw.covariance_
            
            base_Sigma = Sigma.copy()
            trace      = np.trace(base_Sigma)
            m          = base_Sigma.shape[0]
            alphas     = np.concatenate([
                np.logspace(-6, -3, num=10),
                np.linspace(1e-3, 5e-2, num=20)
            ])

            for alpha in alphas:
                eps   = alpha * (trace / m)
                Sigma = base_Sigma + eps * np.eye(m)
                if np.linalg.cond(Sigma) < 1e5:
                    break
            else:
                # never broke → use the strongest ridge
                alpha = alphas[-1]
                eps   = alpha * (trace / m)
                Sigma = base_Sigma + eps * np.eye(m)

        elif method == 'wls_v':
            residuals = residuals[self.all_nodes]  # N×M
            vars_array = residuals.var(ddof=1).values  # shape = (N,)
            Sigma = k_h * np.diag(vars_array)

            # Add regularization
            m = Sigma.shape[0]  # number of bottom-level nodes
            avg_var = np.trace(Sigma) / Sigma.shape[0]  # average variance
            eps = 1e-2 * avg_var  # e.g. 1% of mean variance
            Sigma = Sigma + eps * np.eye(m)  # add regularization

        elif method == 'wls_s':
            if S is None:
                S = self.get_summing_matrix()  # N×M
            scales = S.sum(axis=1)
            Sigma = k_h * np.diag(scales)

            # M = S.T @ np.linalg.inv(Sigma) @ S
            # cond_number = np.linalg.cond(M)
            # print("Cond(Sᵀ Σ⁻¹ S) =", cond_number)

            # # 3) glance at Σ
            # print("Σ diagonal (first 5):", np.diag(Sigma)[:5])
            # print("Σ off-diagonal row 0    :", Sigma[0,1:])
        else:
            raise ValueError(f"Unknown MinT method: {method}")
        
        try:
            inv_Sigma = inv(Sigma)
        except np.linalg.LinAlgError as e:
            inv_Sigma = pinv(Sigma)  # Use pseudo-inverse if Sigma is singular

        try:
            op = inv(S.T @ inv_Sigma @ S)

        except np.linalg.LinAlgError as e:
            op = pinv(S.T @ inv_Sigma @ S)  # Use pseudo-inverse if the operation fails

        P = op @ (S.T @ inv_Sigma)
        return P
    
    def W_ols(self, k_h = 1) -> np.ndarray:
        """
        Ordinary Least Squares reconciliation:
            P = (S^T S)^{-1} S^T
        """
        W_h = k_h * np.eye(len(self.all_nodes))  

        return W_h

    def get_P_regression(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Regression‐based reconciliation: up‐ and down‐ regression weights.
        Arguments:
        - data: DataFrame with columns as node names and rows as time series observations.
        """
        base_nodes = self.base_nodes
        M = len(base_nodes)
        N = len(self.all_nodes)
        # up weights: parent ~ children
        up = self._fit_regression_weights(data)
        # down weights: children ~ parent
        down: Dict[str, Dict[str, float]] = {}
        for parent, childs in self.hierarchy.items():
            model = LinearRegression(fit_intercept=False)

            # Drop any nan values in the parent and childs columns
            data = data.dropna(subset=[parent] + childs)
            if data.empty:
                raise ValueError(f"No valid data for parent {parent} and children {childs}")

            model.fit(data[[parent]].values, data[childs].values)
            for coef, c in zip(model.coef_.flatten(), childs):
                down.setdefault(parent, {})[c] = coef

        P = np.zeros((M, N), dtype=float)
        for j, bn in enumerate(base_nodes):
            for i, name in enumerate(self.all_nodes):
                if name == bn:
                    P[j, i] = 1.0
                    continue
                node_i, node_j = self.nodes[name], self.nodes[bn]
                # upward path
                if self._is_ancestor(node_i, node_j):
                    prod = 1.0
                    curr = node_j
                    while curr.parent is not None:
                        p, c = curr.parent.name, curr.name
                        prod *= up[p][c]
                        curr = curr.parent
                    P[j, i] = prod
                # downward path
                elif self._is_ancestor(node_j, node_i):
                    prod = 1.0
                    curr = node_i
                    while curr.parent is not None:
                        p, c = curr.parent.name, curr.name
                        prod *= down[p][c]
                        curr = curr.parent
                    P[j, i] = prod
        return P

    def set_reconciliation_matrix(
        self,
        method: str,
        data: Optional[pd.DataFrame] = pd.DataFrame(),
        residuals: Optional[pd.DataFrame] = pd.DataFrame(),
        initial: Optional[str] = None, 
        k_h: int = 1, 
        s_matrix = "summing"
    ) -> None:
        """
        Compute and store the reconciliation operators (P and W_full).
        method: 'mint_ols', 'classic', 'regression', 'mint_sample', 'mint_shrink', 'mint_shrink_lw', 'wls_v', 'wls_s'
        """
        if not residuals.empty and residuals.shape[1] != len(self.all_nodes):
            if initial is not None:
                print("Residuals shape ", residuals.shape)
                if initial == 'zeroes':
                    zero_res = pd.DataFrame(0.0, index=residuals.index, columns=self.all_nodes)
                    zero_res[self.base_nodes] = residuals[self.base_nodes]
                    residuals = zero_res
                elif initial == 'classic':
                    P0 = self.get_P_classic()
                    residuals = residuals @ P0
                elif initial == 'regression':
                    P0 = self.get_P_regression(data)
                    residuals = residuals @ P0
                else:
                    raise ValueError(f"Unknown initial method: {initial}")
            else:
                raise ValueError("Residuals length does not match the number of nodes in the hierarchy. Requires argument initial")

        if s_matrix == "regression" and data is not None:
            S = self.get_P_regression(data).T

        else:
            S = self.get_summing_matrix()

        method = method.lower()                # M×N
        if method in ('classic', 'regression'):
            if method == 'regression' and data is None:
                raise ValueError("Data required for regression method")

            if method == 'regression':
                P = self.get_P_regression(data).T
            elif method == 'classic':
                P = self.get_P_classic().T
            
            # Now, we need to fill with zeroes the columns that correspond with non-base nodes
            G = np.zeros((len(self.all_nodes), len(self.all_nodes)), dtype=float)

            for i, node in enumerate(self.all_nodes):
                if node in self.base_nodes:
                    G[:, i] = P[:, self.base_nodes.index(node)]
                else:
                    G[:, i] = 0.0

        elif method.startswith('mint_') or method.startswith('wls_'):
            if method in ("mint_ols", "wls_s"):
                P = self.get_P_MinT(method=method, k_h=k_h, S=S)
            elif method in ('mint_sample', 'mint_shrink', 'wls_v', 'mint_shrink_lw'):
                if residuals is None:
                    raise ValueError(f"Residuals required for {method} method")
                P = self.get_P_MinT(residuals=residuals, method=method, k_h=k_h, S=S)

            G = S @ P   
        else:
            raise ValueError(f"Unknown reconciliation method: {method}")
              

        self.P = P                             
        self.G = G                  

        return P, G

    def forward(
        self,
        base_forecasts: Union[pd.Series, np.ndarray, Dict[str, float]],
        base_vars: Optional[Union[pd.Series, np.ndarray, Dict[str, float]]] = None,
        method: str = "point",
        n_samples: int = 1000, 
        quantiles: Optional[List[float]] = 0.2
    ) -> Union[pd.Series, Dict[str, Any], pd.DataFrame]:
        """
        Generate reconciled forecasts (point or probabilistic).

        - 'point': roll up bottom-level only (no reconciliation) or apply W_full if set.
        - 'gaussian': propagate diag(base_vars) → full covariance analytically.
        - 'montecarlo': draw bottom-level ensemble and reconcile each draw.
        """
        # Each row of base_forecasts should correspond to a b vector
        b = base_forecasts.values.T
                           
        method = method.lower()
        # --- point forecast ---
        if method == 'point':
            if hasattr(self, 'G'):
                y_rec = self.G @ b
                return pd.DataFrame(y_rec.T, columns=self.all_nodes)
            else:
                # no reconciliation, return naive
                raise ValueError("Set a G matrix.")

        # for probabilistic, base_vars must be provided
        if base_vars is None:
            raise ValueError("base_vars required for probabilistic methods")

        T = base_forecasts.shape[0]  # number of time steps
        m = len(self.all_nodes)  # number of bottom-level nodes
        base_forecasts = base_forecasts.to_numpy().astype(float)  # ensure float type
        base_vars = base_vars.to_numpy().astype(float)  # ensure float type
        mu_rec   = np.zeros((T, m))  # reconciled means
        var_rec  = np.zeros((T, m))  # reconciled variances (diagonal of reconciled covariance)
        # --- gaussian propagation ---
        if method == 'gaussian':
            for t in range(T):
                # 1) Extract means and variances for time t
                mu_t  = base_forecasts[t, :]   # shape (m,)
                var_t = base_vars[t, :]    # shape (m,)
                W     = np.diag(var_t)           # shape (m, m)
                
                G = self.G

                mu_rec[t, :] = G @ mu_t  # shape (N,)
                W_rec = G @ W @ G.T  # shape (N, N)
                var_rec[t, :] = np.diag(W_rec)

            return pd.DataFrame(mu_rec, columns=self.all_nodes), pd.DataFrame(var_rec, columns=self.all_nodes)

        # --- montecarlo propagation ---
        elif method == 'montecarlo':
            rec_mean_mc = np.zeros((T, m))
            rec_var_mc  = np.zeros((T, m))

            G = self.G  # precomputed reconciliation matrix of shape (m×m)

            for t in range(T):
                mu_t  = base_forecasts[t, :]   # (m,)
                var_t = base_vars[t, :]        # (m,)

                # 1) Draw K independent Gaussian samples (m × n_samples)
                draws = np.random.normal(
                    loc  = mu_t.reshape(m, 1),
                    scale= np.sqrt(var_t).reshape(m, 1),
                    size = (m, n_samples)
                )

                # 2) Reconcile each draw: (m×m) @ (m×n_samples) → (m×n_samples)
                reconciled_draws = G.dot(draws)

                # 3) Summarize:
                rec_mean_mc[t, :] = np.mean(reconciled_draws, axis=1)
                rec_var_mc[t, :]  = np.sqrt(np.var(reconciled_draws, axis=1, ddof=1))

            # <-- now that all T steps are done, return the full matrices as DataFrames:
            return (
                pd.DataFrame(rec_mean_mc, columns=self.all_nodes),
                pd.DataFrame(rec_var_mc,  columns=self.all_nodes),
            )

        else:
            raise ValueError("Unknown method: use 'point', 'gaussian', or 'montecarlo'.")
