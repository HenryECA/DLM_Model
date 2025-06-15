import numpy as np
import pandas as pd
import time
import os
import gc

from general_arch import HierarchicalTree
from synthetic_series import read_synthetic_series
from snp_data import read_snp500_data
from model_runner import ModelRunner
from utils import seed_all, run_metrics

# Set random seed for reproducibility
seed_all(42)

# Constants
PCT_80 = 1.282
RECONCILIATION_METHODS = [
    'classic', 'regression', 'mint_ols', 'mint_sample',
    'mint_shrink', 'mint_shrink_lw', 'wls_v', 'wls_s'
]
VARIATIONAL_METHODS = ['gaussian', 'montecarlo']
ALPHA = 0.2
MAX_HORIZON = 30
WINDOW_SIZE = 10


def main(hierarchy, series, results_path, train_ratio=0.8):
    os.makedirs(results_path, exist_ok=True)
    metrics_csv = os.path.join(results_path, "dataframe_evaluation_metrics.csv")
    sem_csv     = os.path.join(results_path, "dataframe_evaluation_metrics_se.csv")
    computing_time_csv = os.path.join(results_path, "computing_time.csv")
    recon_json = os.path.join(results_path, "reconciliation_results.json")

    # Prepare output files with headers
    pd.DataFrame().to_csv(metrics_csv, index=False)
    pd.DataFrame().to_csv(sem_csv, index=False)
    computing_time_records = []
    recon_results = {}

    # Instantiate hierarchy and models
    tree = HierarchicalTree(hierarchy)
    models = {
        name: ModelRunner(series[name], train_ratio=train_ratio,
                          max_horizon=MAX_HORIZON, window_size=WINDOW_SIZE)
        for name in series
    }

    # Fit all models
    for name, model in models.items():
        print(f"Fitting model for {name}...")
        model.run_all(active_models=["DLM"], plot=False)

    # Precompute residuals storage by horizon will be streamed within loop

    # Compute reconciliation matrices timing
    for method in RECONCILIATION_METHODS:
        start = time.time()
        # For methods requiring data or first horizon setup
        if method == 'classic':
            tree.set_reconciliation_matrix(method=method)
        elif method == 'mint_ols':
            tree.set_reconciliation_matrix(method=method, k_h=1)
        # others will be called each horizon
        elapsed = time.time() - start
        # record placeholder, will update average later
        for var in VARIATIONAL_METHODS:
            computing_time_records.append({
                'rec_method': method,
                'var_method': var,
                'matrix': elapsed,
                'forward_pass': None
            })

    # Horizon-by-horizon processing
    for h in range(MAX_HORIZON):
        print(f"Processing horizon {h+1}/{MAX_HORIZON}")
        # Build base forecasts, variances, residuals for this horizon
        preds_df = pd.DataFrame({
            name: models[name].predictions['test']['DLM'][h][: -h or None]
            for name in tree.all_nodes
        }).astype(np.float32)

        # handle variances if available
        vars_dict = models[next(iter(models))].std
        if vars_dict:
            vars_df = pd.DataFrame({
                name: models[name].std['test']['DLM'][h][: -h or None]
                for name in tree.all_nodes
            }).astype(np.float32)
        else:
            vars_df = None

        # residuals: preds - actual
        actuals = pd.DataFrame({
            name: models[name].test_unscaled[h:][:len(preds_df)]
            for name in tree.all_nodes
        }).astype(np.float32)
        resid_df = preds_df - actuals

        # Compute & update reconciliation matrices per horizon
        for idx, method in enumerate(RECONCILIATION_METHODS):
            t0 = time.time()
            if method in ('mint_sample', 'mint_shrink', 'mint_shrink_lw', 'wls_v'):
                tree.set_reconciliation_matrix(method=method, residuals=resid_df, k_h=1)
            elif method == 'regression':
                tree.set_reconciliation_matrix(method=method, data=pd.DataFrame(series).astype(np.float32))
            elif method == 'wls_s':
                tree.set_reconciliation_matrix(method=method, k_h=1)
            # classic and mint_ols already set
            compute_mat_time = time.time() - t0
            # accumulate matrix time to records
            base_idx = idx * len(VARIATIONAL_METHODS)
            for var_i, var in enumerate(VARIATIONAL_METHODS):
                computing_time_records[base_idx + var_i]['matrix'] += compute_mat_time

        # Forward pass and metrics
        for idx, method in enumerate(RECONCILIATION_METHODS):
            P, G = tree.P, tree.G
            for var in VARIATIONAL_METHODS:
                t0 = time.time()
                rec = tree.forward(base_forecasts=preds_df,
                                   base_vars=vars_df,
                                   method=var)
                fwd_time = time.time() - t0
                # update forward_pass time
                rec_idx = next(i for i, r in enumerate(computing_time_records)
                               if r['rec_method']==method and r['var_method']==var)
                computing_time_records[rec_idx]['forward_pass'] += fwd_time

                # store reconciled results per horizon-node
                if h not in recon_results:
                    recon_results[h] = {}
                recon_results[h][(method, var)] = rec

                # compute and stream metrics
                records, sems = [], []
                for node in tree.all_nodes:
                    y = models[node].test_unscaled[h:][:len(preds_df)]
                    # base metric
                    m_base, se_base = run_metrics(y, preds_df[node].values,
                                                  None, alpha=ALPHA)
                    records.append({'horizon':h+1, 'node':node,
                                    'rec_method':'base', 'var_method':'base', **m_base})
                    sems.append({'horizon':h+1, 'node':node,
                                 'rec_method':'base', 'var_method':'base', **se_base})
                    # reconciled metric
                    mu_rec = rec[0][node].values
                    std_rec = rec[1][node].values if rec[1] is not None else None
                    m_rec, se_rec = run_metrics(y, mu_rec, std_rec, alpha=ALPHA)
                    records.append({'horizon':h+1, 'node':node,
                                    'rec_method':method, 'var_method':var, **m_rec})
                    sems.append({'horizon':h+1, 'node':node,
                                 'rec_method':method, 'var_method':var, **se_rec})

                # append to CSV
                pd.DataFrame(records).to_csv(metrics_csv, mode='a', header=False, index=False)
                pd.DataFrame(sems).to_csv(sem_csv,    mode='a', header=False, index=False)

        # Cleanup per-horizon data
        del preds_df, vars_df, resid_df, actuals
        gc.collect()

    # Finalize computing_time: average over horizons
    ct_df = pd.DataFrame(computing_time_records)
    # divide by number of horizons
    ct_df['matrix'] /= MAX_HORIZON
    ct_df['forward_pass'] /= MAX_HORIZON
    ct_df.to_csv(computing_time_csv, index=False)

    # dump reconciliation results
    import json
    with open(recon_json, 'w') as f:
        json.dump(recon_results, f, indent=4)

    print("All files written successfully.")


if __name__ == "__main__":
    hierarchy, series = read_snp500_data(folder_path="data/snp500_data/yfinance")
    main(
        hierarchy=hierarchy,
        series=series,
        results_path=r"D:\Documentos\ICAI\TFG\Code\DLM Model\snp500_results\reconciliation",
        train_ratio=0.8
    )
