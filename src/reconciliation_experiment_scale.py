import numpy as np
import pandas as pd
import time
import os

from general_arch import HierarchicalTree
from synthetic_series import read_synthetic_series
from snp_data import read_snp500_data
from model_runner_scaled import ModelRunner
from utils import seed_all, run_metrics
from plotter import MetricsTablePlotter
import json

import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed_all(42)

PCT_80 = 1.282

import warnings
warnings.filterwarnings("ignore")

RECONCILIATION_METHODS = ['classic', 'regression', 'mint_ols', 'mint_sample', 'mint_shrink', 'mint_shrink_lw', 'wls_v', 'wls_s']
VARIATIONAL_METHODS = ['gaussian', 'montecarlo']

def main(hierarchy, series, results_path, train_ratio=0.8, s_matrix="summing"):
    data_nans = {}

    # Fin max length
    max_length = max(len(series[key]) for key in series.keys())

    for key in series.keys():
        if len(series[key]) < max_length:
            array_nans = np.nan * np.ones(max_length - len(series[key]))
            data_nans[key] = np.concatenate((array_nans, series[key]))
        else:
            data_nans[key] = series[key]

    data = pd.DataFrame(data_nans)  # Transpose to have series as columns

    # Instantiate the hierarchical tree
    tree = HierarchicalTree(hierarchy)
    leaves = tree.base_nodes  # bottom-level nodes
    # Sort data by tree.all_nodes
    data = data[tree.all_nodes]

    models = {name: ModelRunner(series[name], train_ratio=train_ratio, max_horizon=30, window_size=10) for name in series}
    horizon_predictions = {h: pd.DataFrame(columns=tree.all_nodes) for h in range(30)}
    horizon_variances = {h: pd.DataFrame(columns=tree.all_nodes) for h in range(30)}
    horizon_residuals = {h: pd.DataFrame(columns=tree.all_nodes) for h in range(30)}
    
    k_h = 1

    min_len = 0

    for name, model in models.items():
        print(f"Fitting model for {name}...")
        model.run_all(active_models=["DLM"], plot=False)


    for h in range(30):
        for name in tree.all_nodes:
            # Get predictions and variances for each horizon
            preds = models[name].predictions["test"]["DLM"][h]
            vars = models[name].std["test"]["DLM"][h] if models[name].std else None
            horizon_predictions[h][name] = preds[:len(preds)-h]
            horizon_variances[h][name] = vars[:len(preds)-h]
            horizon_residuals[h][name] = preds[:len(preds)-h] - models[name].test[h:]

    print("Data preparation complete. Starting reconciliation...")

    reconciliation_matrices = {h: {rec: [] for rec in RECONCILIATION_METHODS} for h in range(30)}

    computing_time = pd.DataFrame(columns=["var_method", "rec_method", "matrix", "forward_pass"])

    for method in RECONCILIATION_METHODS:
        times = []
        for h in range(30):
            if method == 'mint_ols':
                start_time = time.time()
                if s_matrix == "regression":
                    tree.set_reconciliation_matrix(method=method, data=data, k_h=k_h, s_matrix=s_matrix)
                else:
                    tree.set_reconciliation_matrix(method=method, k_h=k_h)
            elif method == 'classic':
                start_time = time.time()
                tree.set_reconciliation_matrix(method=method)
            elif method == 'regression':
                start_time = time.time()
                tree.set_reconciliation_matrix(method=method, data=data)
            elif method == 'mint_sample':
                start_time = time.time()
                if s_matrix == "regression":
                    tree.set_reconciliation_matrix(method=method, data=data, residuals=horizon_residuals[h], k_h = k_h, s_matrix=s_matrix)
                else:
                    tree.set_reconciliation_matrix(method=method, residuals=horizon_residuals[h], k_h = k_h)
            elif method == 'mint_shrink':
                start_time = time.time()
                if s_matrix == "regression":
                    tree.set_reconciliation_matrix(method=method, data=data, residuals=horizon_residuals[h], k_h = k_h, s_matrix=s_matrix)
                else:
                    tree.set_reconciliation_matrix(method=method, residuals= horizon_residuals[h], k_h=k_h)
            elif method == 'mint_shrink_lw':
                start_time = time.time()
                if s_matrix == "regression":
                    tree.set_reconciliation_matrix(method=method, data=data, residuals=horizon_residuals[h], k_h = k_h, s_matrix=s_matrix)
                else:
                    tree.set_reconciliation_matrix(method=method, residuals=horizon_residuals[h], k_h=k_h)
            elif method == 'wls_v':
                start_time = time.time()
                if s_matrix == "regression":
                    tree.set_reconciliation_matrix(method=method, data=data, residuals=horizon_residuals[h], k_h = k_h, s_matrix=s_matrix)
                else:
                    tree.set_reconciliation_matrix(method=method, residuals=horizon_residuals[h], k_h=k_h)
            elif method == 'wls_s':
                start_time = time.time()
                if s_matrix == "regression":
                    tree.set_reconciliation_matrix(method=method, data=data, k_h=k_h, s_matrix=s_matrix)
                else:
                    tree.set_reconciliation_matrix(method=method, k_h=k_h)
            
            times.append(time.time() - start_time)
            reconciliation_matrices[h][method] = [tree.P, tree.G]
        
        computing_time.loc[len(computing_time)] = {
            "var_method": "gaussian",  # Assuming base for variational methods
            "rec_method": method,
            "matrix": np.mean(times),
            "forward_pass": None  # Placeholder for forward pass time
        }
        computing_time.loc[len(computing_time)] = {
            "var_method": "montecarlo",  # Assuming base for variational methods
            "rec_method": method,
            "matrix": np.mean(times),
            "forward_pass": None  # Placeholder for forward pass time
        }


    print("Reconciliation matrices computed.")

    results = {h: {method: {var: {} for var in VARIATIONAL_METHODS} for method in RECONCILIATION_METHODS} for h in range(30)}

    # Now, we execute the forward pass for each horizon and method
    for method in RECONCILIATION_METHODS:
        for var in VARIATIONAL_METHODS:
            times = []
            for h in range(30):
                P, G = reconciliation_matrices[h][method]
                tree.P = P
                tree.G = G
                start_time = time.time()
                reconciled_results = tree.forward(
                    base_forecasts=horizon_predictions[h],
                    base_vars=horizon_variances[h] if horizon_variances[h] is not None else None,
                    method=var
                )

                times.append(time.time() - start_time)

                for node in tree.all_nodes:
                    if node not in results[h][method][var]:
                        results[h][method][var][node] = {}
                    if reconciled_results is not None:
                        mu_scale = reconciled_results[0][node].values.reshape(1, -1)
                        mu = models[node].scaler.inverse_transform(mu_scale).flatten().tolist() if mu_scale is not None else None
                        std_scale = (reconciled_results[1][node].values.reshape(1, -1) * models[node].scaler.scale_[0]).flatten().tolist() if reconciled_results[1] is not None else None
                        results[h][method][var][node] = {
                            'mu': mu,
                            'std': std_scale
                        }
            
            # Get index of row with the current method and variational method
            index = computing_time[(computing_time['rec_method'] == method) & (computing_time['var_method'] == var)].index[0]
            computing_time.at[index, 'forward_pass'] = np.mean(times)  # Update the forward pass time

    # Save results

    base_test = {h: {node: {} for node in tree.all_nodes} for h in range(30)}

    for h in range(30):
        for node in tree.all_nodes:
            base_test[h][node]['base'] = {}
            # Add the base to results dict
            mu = models[node].scaler.inverse_transform(horizon_predictions[h][node].values.reshape(1, -1)).flatten().tolist()
            std = (horizon_variances[h][node].values.reshape(1, -1) * models[node].scaler.scale_[0]).flatten().tolist() if horizon_variances[h][node] is not None else None
            base_test[h][node]['base']['base'] = {
                'y': list(models[node].test_unscaled[h:]),
                'mu': mu,
                'std': std
            }

    with open(os.path.join(results_path, "reconciliation_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(results_path, "base_test_results.json"), 'w') as f:
        json.dump(base_test, f, indent=4)

    # Clear base test results
    del base_test

    print("Forward pass completed for all horizons and methods.")

    evaluation_metrics = pd.DataFrame(columns=['node', 'horizon', 'rec_method', 'var_method','alpha', 'RMSE', 'MAE', 'NLL', 'CRPS', 'ECE'])
    evaluation_metrics_se = pd.DataFrame(columns=['node', 'horizon', 'rec_method', 'var_method','alpha', 'RMSE', 'MAE', 'NLL', 'CRPS', 'ECE'])
    alpha = 0.2
    for h in range(30):
        for node in tree.all_nodes:
            for method in RECONCILIATION_METHODS:
                for var in VARIATIONAL_METHODS:
                    if results[h][method][var] is not None:
                        y = models[node].test_unscaled[h:]
                        run_results = results[h][method][var]
                        mu = np.array(run_results[node]['mu'])
                        std = np.array(run_results[node]['std']) if run_results[node]['std'] is not None else None

                        metrics = run_metrics(y, mu, std, alpha=alpha)

                        new_row = {
                            'node':       node,
                            'horizon':    h + 1,
                            'rec_method': method,
                            'var_method': var,
                            'alpha':      alpha,
                            **metrics[0]      # assuming `metrics` is a dict of other columns→values
                        }

                        new_row_se = {
                            'node':       node,
                            'horizon':    h + 1,
                            'rec_method': method,
                            'var_method': var,
                            'alpha':      alpha,
                            **metrics[1]      # assuming `metrics` is a dict of other columns→values
                        }

                        # Assign to the next integer index:
                        evaluation_metrics.loc[len(evaluation_metrics)] = new_row

                        evaluation_metrics_se.loc[len(evaluation_metrics_se)] = new_row_se

                        # Delete the results for the current method and variational method
                        del results[h][method][var][node]

            # Now add the results for the base forecasts (no reconciliation)
            y = models[node].test_unscaled[h:]
            mu = models[node].scaler.inverse_transform(horizon_predictions[h][node].values.reshape(1, -1)).flatten()
            std = (horizon_variances[h][node].values.reshape(1, -1) * models[node].scaler.scale_[0]).flatten() if horizon_variances[h][node] is not None else None

            met = run_metrics(y, mu, std, alpha=alpha)

            new_row = {
                'node':       node,
                'horizon':    h + 1,
                'rec_method': 'base',
                'var_method': 'base',
                'alpha':      alpha,
                **met[0]  # assuming `run_metrics` returns a dict
            }

            new_row_se = {
                'node':       node,
                'horizon':    h + 1,
                'rec_method': 'base',
                'var_method': 'base',
                'alpha':      alpha,
                **met[1]  # assuming `run_metrics` returns a dict
            }

            evaluation_metrics.loc[len(evaluation_metrics)] = new_row
            evaluation_metrics_se.loc[len(evaluation_metrics_se)] = new_row_se



    print("Evaluation metrics computed.")

    # Save the results to a CSV file

    evaluation_metrics.to_csv(os.path.join(results_path, "dataframe_evaluation_metrics.csv"), index=False)
    evaluation_metrics_se.to_csv(os.path.join(results_path, "dataframe_evaluation_metrics_se.csv"), index=False)
    computing_time.to_csv(os.path.join(results_path, "computing_time.csv"), index=False)
    



def parse_predictions_data(test, results, variances, horizon):
    """
    Adjusts the data to be used with the reconciliation matrices
    """

    horizon_data = {}

    for h in range(horizon):
        x_values = []
        y_values = []
        lower_values = []
        upper_values = []

        for i in range(len(test)):
            if len(results[i]) >= h:
                x_values.append(test[i])
                y_values.append(results[i][-h])
                if len(variances) > 0:
                    lower_values.append(variances[i][-h][0])
                    upper_values.append(variances[i][-h][1])

        horizon_data[h] = {
            'x': np.array(x_values),
            'y': np.array(y_values),
            'lower': np.array(lower_values) if lower_values else None,
            'upper': np.array(upper_values) if upper_values else None
        }

    return horizon_data

        

    


if __name__ == "__main__":
    # hierarchy, series = read_synthetic_series(
    #     experiment_path="data/synthetic_series/",
    #     folder_name="example"
    # )

    # main(
    #     hierarchy=hierarchy,
    #     series=series,
    #     results_path=r"C:\Users\202109355\Documents\DLM_Model\synth_results\reconciliation_scale", 
    #     s_matrix="regression",  # Change to "summing" for classic reconciliation
    # )


    # hierarchy, series = read_snp500_data(folder_path="data/snp500_data/yfinance")

    # main(
    #     hierarchy=hierarchy,
    #     series=series,
    #     results_path=r"D:\Documentos\ICAI\TFG\Code\DLM Model\snp500_results\reconciliation", 
    #     train_ratio=100
    # )

    hierarchy, series = read_snp500_data(folder_path="data/snp500_data/yfinance")
    main(
        hierarchy=hierarchy,
        series=series,
        results_path=r"C:\Users\202109355\Documents\DLM_Model\snp500_results\reconciliation3", 
        train_ratio=200,
        s_matrix="regression"
    )