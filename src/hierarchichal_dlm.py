from models.dlm_model import DLM
from general_arch import HierarchicalTree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_group_priors(data_dict):
    """
    data_dict: dict of the form { leaf_name: 1D np.ndarray }
    Returns group-level priors:
      - 'v_delta': mean variance of Δy
      - 'v_delta2': mean variance of Δ²y
      - 'w': mean variance of y
    """
    v_deltas, v_deltas2, w_vars = [], [], []
    for y in data_dict.values():
        v_deltas.append(  np.var(np.diff(y, 1)) )
        v_deltas2.append( np.var(np.diff(y, 2)) )
        w_vars.append(    np.var(y)         )
    return {
        'v_delta':  np.mean(v_deltas),
        'v_delta2': np.mean(v_deltas2),
        'w':        np.mean(w_vars)
    }

def build_leaf_dlm(name, y, spec, priors, shrink=0.5, ci=0.95):
    """
    Construct and initialize a DLM for leaf `name` with empirical Bayes shrinkage.
      - y: 1D array of observations
      - spec: dict specifying components for DLM.from_spec
      - priors: dict with 'v_delta','v_delta2','w'
      - shrink: lambda in [0,1]
      - ci: confidence level
    """
    # local estimates
    v_d  = np.var(np.diff(y, 1))
    v_d2 = np.var(np.diff(y, 2))
    w0   = np.var(y)

    # shrinkage
    V_lvl = np.array([[ shrink*v_d   + (1-shrink)*priors['v_delta'] ]])
    V_tr  = np.diag([ shrink*v_d2  + (1-shrink)*priors['v_delta2'], 0. ])
    W_obs = np.array([[ shrink*w0    + (1-shrink)*priors['w'] ]])

    # build DLM with custom noise blocks
    return DLM.from_spec(
        spec, data=y, ci=ci,
        custom_V_lvl=V_lvl,
        custom_V_tr=V_tr,
        custom_W_obs=W_obs
    )

if __name__ == "__main__":
    np.random.seed(0)

    # 1) Simulate synthetic data for each leaf
    n_total = 200
    leaves = ['AA','AB','BA','BB']
    data = {}
    for leaf in leaves:
        t = np.arange(n_total)
        data[leaf] = (
            0.05 * t
            + 5 * np.sin(2*np.pi*t/7 + hash(leaf) % 10)
            + np.random.normal(0,1.0,n_total)
        )

    # 2) Train/test split
    h = 10
    n_train = n_total - h
    train_data = {leaf: data[leaf][:n_train] for leaf in leaves}

    # 3) Compute group priors on train
    priors = compute_group_priors(train_data)
    print("=== Empirical Bayes group priors ===")
    for k,v in priors.items():
        print(f"  {k}: {v:.4f}")
    print()

    # 4) DLM specification and shrinkage
    spec = {'level': True, 'trend': True, 'seasonal': {'periods':[7]}, 'ar':0}
    shrink = 0.5
    ci = 0.95

    def metrics(true, pred):
        e = pred - true
        return np.mean(np.abs(e)), np.sqrt(np.mean(e**2))
    
    # --- 5) Fit each leaf, forecast & store results ---
    results = {}
    for leaf in leaves:
        y_train = train_data[leaf]
        y_test  = data[leaf][n_train:]

        model = build_leaf_dlm(leaf, y_train, spec, priors, shrink=shrink, ci=ci)
        y_pred, y_low, y_up = model.forecast(h)

        mae, rmse = metrics(y_test, y_pred)
        results[leaf] = {
            'true': y_test,
            'pred': y_pred,
            'lower': y_low,
            'upper': y_up,
            'MAE': mae,
            'RMSE': rmse
        }

    # --- 6) Build DataFrame of true & predicted for all nodes ---
    # True test series:
    df = pd.DataFrame({leaf: results[leaf]['true'] for leaf in leaves},
                      index=np.arange(n_train, n_total))
    # Leaf predictions:
    for leaf in leaves:
        df[f"{leaf}_pred"] = results[leaf]['pred']
    # Intermediate nodes True and Naive-pred:
    df['A_true'] = df['AA'] + df['AB']
    df['B_true'] = df['BA'] + df['BB']
    df['O_true'] = df['A_true'] + df['B_true']
    df['A_pred_naive'] = df['AA_pred'] + df['AB_pred']
    df['B_pred_naive'] = df['BA_pred'] + df['BB_pred']
    df['O_pred_naive'] = df['A_pred_naive'] + df['B_pred_naive']

    # --- 7) Reconcile via HierarchicalTree ---
    df_hist = pd.DataFrame(train_data)
    df_hist['A'] = df_hist['AA'] + df_hist['AB']
    df_hist['B'] = df_hist['BA'] + df_hist['BB']
    df_hist['O'] = df_hist['A']  + df_hist['B']
    tree = HierarchicalTree({'O':['A','B'], 'A':['AA','AB'], 'B':['BA','BB']})
    W_mat, all_nodes = tree.compute_transition_matrix(df_hist, leaves)

    print("\n=== Reconciliation matrix W ===")

    # Plot the reconciliation matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(W_mat, cmap='Blues', aspect='auto')
    plt.colorbar(label='Weight')
    plt.xticks(range(len(all_nodes)), all_nodes, rotation=45)
    plt.yticks(range(len(all_nodes)), all_nodes)
    plt.title('Reconciliation Matrix W')
    plt.tight_layout()
    plt.show()


    base_means = np.vstack([results[lf]['pred'] for lf in leaves])

    print(W_mat)
    print(base_means)
    rec = W_mat @ base_means
    # map reconciliation into df
    for node in ['A','B','O']:
        idx = all_nodes.index(node)
        df[f"{node}_pred_rec"] = rec[idx]

    # --- 8) Compute errors for every node ---
    error_rows = []
    for node in ['AA','AB','BA','BB','A','B','O']:
        true_col = node if node in leaves else f"{node}_true"
        for method in ['naive','rec']:
            if node in leaves and method=='rec':
                continue  # no rec for leaves
            pred_col = f"{node}_pred" if node in leaves else f"{node}_pred_{method}"
            mae, rmse = metrics(df[true_col].values, df[pred_col].values)
            error_rows.append({
                'node': node,
                'method': method,
                'MAE': mae,
                'RMSE': rmse
            })
    errors_df = pd.DataFrame(error_rows)

    # --- 9) Display results ---
    print("\n=== Time-series table (first rows) ===")
    print(df.head(5).round(2))
    print("\n=== Forecast Errors by Node and Method ===")
    print(errors_df.pivot(index='node', columns='method', values=['MAE','RMSE']).round(3))

