#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.dlm_model import DLM
from prob_arch import ProbabilisticHierarchicalTree   # assumes you saved that class here

def compute_group_priors(data_dict):
    v_deltas, v_deltas2, w_vars = [], [], []
    for y in data_dict.values():
        v_deltas.append(np.var(np.diff(y,1)))
        v_deltas2.append(np.var(np.diff(y,2)))
        w_vars.append(np.var(y))
    return {
        'v_delta':  np.mean(v_deltas),
        'v_delta2': np.mean(v_deltas2),
        'w':        np.mean(w_vars)
    }

def build_leaf_dlm(name, y, spec, priors, shrink=0.5, ci=0.95):
    v_d  = np.var(np.diff(y,1))
    v_d2 = np.var(np.diff(y,2))
    w0   = np.var(y)
    V_lvl = np.array([[shrink*v_d   + (1-shrink)*priors['v_delta']]])
    V_tr  = np.diag([shrink*v_d2  + (1-shrink)*priors['v_delta2'], 0.])
    W_obs = np.array([[shrink*w0    + (1-shrink)*priors['w']]])
    return DLM.from_spec(
        spec, data=y, ci=ci,
        custom_V_lvl=V_lvl,
        custom_V_tr=V_tr,
        custom_W_obs=W_obs
    )

def metrics(true, pred):
    e = pred - true
    return np.mean(np.abs(e)), np.sqrt(np.mean(e**2))

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
            + 5 * np.sin(2*np.pi*t/7 + hash(leaf)%10)
            + np.random.normal(0,1.0,n_total)
        )

    # 2) Train/test split
    h = 10
    n_train = n_total - h
    train_data = {leaf: data[leaf][:n_train] for leaf in leaves}
    test_data  = {leaf: data[leaf][n_train:] for leaf in leaves}

    # 3) Empirical Bayes priors
    priors = compute_group_priors(train_data)

    # 4) DLM spec & fit leaves + collect in-sample residuals
    spec   = {'level':True,'trend':True,'seasonal':{'periods':[7]},'ar':0}
    shrink = 0.5
    ci     = 0.95

    # store residuals on train and point_forecasts for test
    resid_df = pd.DataFrame(index=np.arange(n_train))
    point_fc = {}

    for leaf in leaves:
        y_tr = train_data[leaf]
        dlm = build_leaf_dlm(leaf, y_tr, spec, priors, shrink, ci)
        # collect one-step-ahead residuals on train
        fitted = []
        dlm.reset()                 # ensure fitted only on train
        for y in y_tr:
            m, _ = dlm.predict_once()
            fitted.append(m.item())
            dlm.update(np.array([[y]]))
        resid_df[leaf] = y_tr - np.array(fitted)

        # now forecast h steps
        m, _, _ = dlm.forecast(h)
        point_fc[leaf] = m

    # 5) Build ProbabilisticHierarchicalTree, simulate
    hierarchy = {'O':['A','B'],'A':['AA','AB'],'B':['BA','BB']}
    pht = ProbabilisticHierarchicalTree(hierarchy)

    # Assemble point_forecasts and residuals tables
    pf_series = pd.Series(point_fc)
    sims, all_nodes = pht.simulate(
        pf_series,
        resid_df,
        pd.DataFrame({**train_data, **test_data}),
        n_simulations=2000
    )

    # 6) Extract median forecast for each node
    sim_med = np.median(sims, axis=0)
    fc_df   = pd.DataFrame([sim_med], columns=all_nodes)

    # 7) Build true test series for all nodes
    test_df = pd.DataFrame({leaf: test_data[leaf] for leaf in leaves})
    test_df['A'] = test_df['AA'] + test_df['AB']
    test_df['B'] = test_df['BA'] + test_df['BB']
    test_df['O'] = test_df['A']  + test_df['B']

    # 8) Compute errors for every node
    error_rows = []
    for node in all_nodes:
        true_vals = test_df[node].values
        pred_vals = np.repeat(fc_df[node].values, h)  # same median at all h steps
        mae, rmse = metrics(true_vals, pred_vals)
        error_rows.append({'node':node,'MAE':mae,'RMSE':rmse})

    err_df = pd.DataFrame(error_rows).set_index('node')
    print("\n=== Forecast Errors (Median simulation) ===")
    print(err_df.round(3))

    # 9) Plot 5–95% intervals
    q05 = np.percentile(sims, 5, axis=0)
    q95 = np.percentile(sims, 95, axis=0)
    plt.figure(figsize=(8,4))
    plt.errorbar(all_nodes, q05, yerr=q95-q05, fmt='o', capsize=5)
    plt.xticks(rotation=45)
    plt.title("5%–95% Prediction Intervals (Probabilistic Reconciliation)")
    plt.tight_layout()
    plt.show()
