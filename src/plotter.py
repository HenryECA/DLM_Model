import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import dataframe_image as dfi
import os
from matplotlib import colors

from utils import seed_all, run_metrics

seed_all(42)

class MetricsTablePlotter:

    def create_dataframes(all_metrics, group=None):
        models = list(all_metrics["err"].keys())
        base_horizon_length = len(all_metrics["err"][models[0]]['RMSE'])
        horizons = list(range(1, base_horizon_length + 1))

        def group_and_average(data, group_size):
            """Group data into chunks of group_size and return averages"""
            if len(data) == 0:  # Handle empty data case
                return []
            return [np.mean(data[i:i+group_size]) 
                    for i in range(0, len(data), group_size)]

        # Common validation for metrics2 models
        def is_valid_model(model):
            return all(
                metric in all_metrics["err"][model] and 
                len(all_metrics["err"][model][metric]) == base_horizon_length
                for metric in metrics2
            )

        if group is not None:
            group = int(group)
            # Create grouped horizon labels (e.g., "1-5", "6-10")
            n_groups = (base_horizon_length + group - 1) // group  # Ceiling division
            horizons_grouped = [
                f"{i*group + 1}-{min((i+1)*group, base_horizon_length)}" 
                for i in range(n_groups)
            ]
            
            # Process metrics1
            metrics1 = ['RMSE', 'MAE']
            df1_data = {
                (metric, model): group_and_average(all_metrics["err"][model][metric], group)
                for metric in metrics1 
                for model in models
                if len(all_metrics["err"][model][metric]) == base_horizon_length
            }

            df1_data_se = {
                (metric, model): group_and_average(all_metrics["err_se"][model][metric], group)
                for metric in metrics1
                for model in models
                if len(all_metrics["err_se"][model][metric]) == base_horizon_length
            }

            df1 = pd.DataFrame(df1_data, index=horizons_grouped)
            df1.index.name = 'Horizon'
            df1 = df1.T

            df1_se = pd.DataFrame(df1_data_se, index=horizons_grouped)
            df1_se.index.name = 'Horizon'
            df1_se = df1_se.T
            
            # Process metrics2 with strict validation
            metrics2 = ['CRPS', 'NLL', 'ECE']
            valid_models = [m for m in models if is_valid_model(m)]
            
            if valid_models:
                df2_data = {
                    (metric, model): group_and_average(all_metrics["err"][model][metric], group)
                    for metric in metrics2 
                    for model in valid_models
                }
                # Additional length validation
                df2_data = {k: v for k, v in df2_data.items() if len(v) == len(horizons_grouped)}

                df2_data_se = {
                    (metric, model): group_and_average(all_metrics["err_se"][model][metric], group)
                    for metric in metrics2 
                    for model in valid_models
                }
                
                # Additional length validation
                df2_data_se = {k: v for k, v in df2_data_se.items() if len(v) == len(horizons_grouped)}

                
                df2 = pd.DataFrame(df2_data, index=horizons_grouped)
                df2.index.name = 'Horizon'
                df2 = df2.T
                df2_se = pd.DataFrame(df2_data_se, index=horizons_grouped)
                df2_se.index.name = 'Horizon'
                df2_se = df2_se.T

            else:
                df2 = pd.DataFrame()
                df2_se = pd.DataFrame()

        else:
            # Original non-grouped implementation with strict validation
            metrics1 = ['RMSE', 'MAE']
            df1 = pd.DataFrame(
                {(metric, model): all_metrics["err"][model][metric] 
                for metric in metrics1 for model in models
                if len(all_metrics["err"][model][metric]) == base_horizon_length},
                index=horizons
            )
            df1.index.name = 'Horizon'
            df1 = df1.T

            df1_se = pd.DataFrame(
                {(metric, model): all_metrics["err_se"][model][metric] 
                for metric in metrics1 for model in models
                if len(all_metrics["err_se"][model][metric]) == base_horizon_length},
                index=horizons
            )
            df1_se.index.name = 'Horizon'
            df1_se = df1_se.T

            metrics2 = ['CRPS', 'NLL', 'ECE']
            valid_models = [m for m in models if is_valid_model(m)]
            
            df2 = pd.DataFrame(
                {(metric, model): all_metrics["err"][model][metric] 
                for metric in metrics2 for model in valid_models},
                index=horizons
            ).T if valid_models else pd.DataFrame()
            
            df2.index.name = 'Horizon'
            df2_se = pd.DataFrame(
                {(metric, model): all_metrics["err_se"][model][metric] 
                for metric in metrics2 for model in valid_models},
                index=horizons
            ).T if valid_models else pd.DataFrame()


        return df1, df2, df1_se, df2_se



    def plot_table_as_image(df, title, save_path=None):
        if df.empty:
            print(f"No data to plot for {title}.")
            return
        sns.set_style("white")
        n_rows, n_cols = df.shape
        fig_width = max(8, n_cols * 0.8)
        fig_height = max(4, n_rows * 0.5)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')

        # Prepare cell colors for zebra striping
        row_colors = []
        for i in range(n_rows):
            row_colors.append('#f9f9f9' if i % 2 == 0 else 'white')

        # Create table
        table = ax.table(
            cellText= [[f"{val:.3f}" for val in row] for row in df.values],
            rowLabels=[f'{m[0]} | {m[1]}' for m in df.index],
            colLabels=[f'H{h}' for h in df.columns],
            cellLoc='center',
            rowColours=row_colors,
            colColours=['#d3d3d3'] * n_cols,
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        # Header formatting
        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == -1:
                cell.set_text_props(weight='bold')
                cell.set_edgecolor('black')
            cell.set_linewidth(0.5)

        ax.set_title(title, fontsize=16, pad=20)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


class ForecastingPlotter:

    def __init__(self, path, results_path):
        self.path = path
        self.results_path = results_path

        # Read json
        with open(os.path.join(self.path, "all_metrics_forecasting.json"), 'r') as f:
            self.all_metrics = json.load(f)

        # Take out GARCH
        self.all_metrics["err"].pop("GARCH", None)
        self.all_metrics["err_se"].pop("GARCH", None)
        
        with open(os.path.join(self.path, "params_forecasting.json"), 'r') as f:
            self.params = json.load(f)

        # Take out GARCH
        self.params.pop("GARCH", None)
        

        with open(os.path.join(self.path, "predictions_test_forecasting.json"), 'r') as f:
            self.predictions = json.load(f)

        # Take out GARCH
        self.predictions.pop("GARCH", None)

        with open(os.path.join(self.path, "std_test_forecasting.json"), 'r') as f:
            self.std = json.load(f)

        # Take out GARCH
        self.std.pop("GARCH", None)

        with open(os.path.join(self.path, "times_forecasting.json"), 'r') as f:
            self.times = json.load(f)

        # Take out GARCH
        self.times.pop("GARCH", None)

        # Read numpy arrays
        self.y_test = np.load(os.path.join(self.path, "test_set_forecasting.npy"))

        # Define the colors for the models

        self.color = {
            "DLM":    "#1f77b4",
            "SMA":    "#ff7f0e",
            "ARIMA":  "#2ca02c",
            "LSTM":   "#d62728",
            "Conv1D": "#9467bd",
            "MLP":    "#8c564b",
            "EMA":    "#17becf",
        }

    def horizon_metric_models(self, error, show=False):
        """
        Given an error metric, plots the horizon vs the error for each model.
        """

        # Create a DataFrame for the error metrics
        df = pd.DataFrame({
            model: self.all_metrics["err"][model][error] 
            for model in self.all_metrics["err"]
        })

        df_se = pd.DataFrame({
            model: self.all_metrics["err_se"][model][error] 
            for model in self.all_metrics["err"]
        })

        # Plotting
        plt.figure(figsize=(12, 6))
        for model in df.columns:
            plt.errorbar(df.index + 1, df[model], yerr=df_se[model], 
                         label=model, color=self.color.get(model, 'black'), 
                         capsize=5, fmt='-o', markersize=4)

        plt.title(f"Horizon vs {error} for each model")
        plt.xlabel("Horizon")
        plt.ylabel(error)
        # Add legend below and horizontal
        plt.legend(title="Models", loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f"horizon_vs_{error}.png"), dpi=600)
        if show:
            plt.show()
        plt.close()


    def line_horizons_models(self, models, horizons, limit=None, start=0, show=False):
        """
        Given a list of model names and forecast horizons, makes a plot
        for each model with one line per horizon (offset by h), plus the
        true series in thick black. Shades ±1 std around each forecast.
        """
        n_models = len(models)
        fig, axes = plt.subplots(n_models, 1,
                                figsize=(12, 4 * n_models),
                                sharex=True)

        # ensure axes is a list
        if n_models == 1:
            axes = [axes]

        max_h = max(horizons) if horizons else 0

        y_true = self.y_test[max_h + start:]
        if limit is not None:
            y_true = y_true[:limit]
        x_true = np.arange(len(y_true))

        for ax, model in zip(axes, models):
            # 1) plot true values
            ax.plot(x_true, y_true,
                    color='black',
                    linewidth=2,
                    label='True')

            # 2) plot each horizon
            preds_h = self.predictions[model]
            std_h   = self.std[model]

            for h in horizons:
                pred = np.asarray(preds_h[str(h)])
                stdev = np.asarray(std_h[str(h)]) if std_h.get(str(h)) is not None else None

                pred = pred[max_h - h + start:][:len(y_true)]
                if stdev is not None:
                    stdev = stdev[max_h - h + start:][:len(y_true)]

                x_pred = np.arange(len(pred))

                # plot the mean forecast
                ax.plot(x_true, pred,
                        label=f'h={h}')

                # shaded ±1 std
                if stdev is not None:
                    stdev = stdev[:len(x_true)]
                    ax.fill_between(x_true, pred - stdev,
                                    pred + stdev,
                                    alpha=0.2)

            ax.set_title(f"Forecasts by {model}")
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.set_xticklabels([])
        axes[-1].set_xlabel("Time index")
        plt.tight_layout()

        # one legend for all subplots
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels,
                loc='lower center',
                ncol=len(horizons) + 1,    # +1 for 'True'
                bbox_to_anchor=(0.5, -0.05))

        # save and/or show
        outpath = os.path.join(self.results_path, "horizons_models.png")
        plt.savefig(outpath, dpi=600, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    
    def line_point_horizon_models(self, folder_path, horizons=None, limit=None, start=0, show=False):
        
        # If horizons is None, then create individual plots for each horizon
        if horizons is not None:
            one_plot = True
        else:
            one_plot = False
            horizons = self.predictions["DLM"].keys()
            # caast to int
            horizons = [int(h) for h in horizons]
        
        h_max = max(horizons) if horizons else 29

        if one_plot:

            # Plot the different horizons continuosly with subplots
            fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 4 * len(horizons)), sharex=True)
            if len(horizons) == 1:  
                axes = [axes]

            for ax, h in zip(axes, horizons):

                y_true = self.y_test[h_max + start:]
                if limit is not None:
                    y_true = y_true[:limit]
                x_true = np.arange(len(y_true))

                # 1) plot true values
                ax.plot(x_true, y_true,
                        color='black',
                        linewidth=2,
                        label='True')
                
                # 2) plot each model
                for model in self.predictions:
                    if str(h) not in self.predictions[model]:
                        continue
                    
                    pred = np.asarray(self.predictions[model][str(h)])

                    pred = pred[h_max - h + start:][:len(y_true)]

                    x_pred = np.arange(len(pred))

                    # plot the mean forecast
                    ax.plot(x_pred, pred,
                            label=f'{model}', color=self.color.get(model, 'black'))
                    
            
                # finish figure
                ax.set_title(f"Forecasts h = {h+1}")
                ax.set_ylabel("Value")
                ax.grid(True)
                ax.set_xticklabels([])
            axes[-1].set_xlabel("Time index")
            plt.tight_layout()
            # one legend for all subplots
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels,
                    loc='lower center',
                    ncol=len(self.predictions) + 1,    # +1 for 'True'
                    bbox_to_anchor=(0.5, -0.05))
            # save and/or show
            outpath = os.path.join(self.results_path, "point_forecast_horizons_models.png")
            plt.savefig(outpath, dpi=600, bbox_inches='tight')
            if show:
                plt.show()

        else:
            # Plot a different plot for each horizon and store them indifidually in folder_path
            # check if folder_path exists, if not create it
            complete_path = os.path.join(self.results_path, folder_path)
            if not os.path.exists(complete_path):
                os.makedirs(complete_path)

            for h in horizons:
                y_true = self.y_test[h_max + start:]
                if limit is not None:
                    y_true = y_true[:limit]
                x_true = np.arange(len(y_true))

                plt.figure(figsize=(12, 4))

                # 1) plot true values
                plt.plot(x_true, y_true,
                        color='black',
                        linewidth=2,
                        label='True')

                # 2) plot each model
                for model in self.predictions:
                    if str(h) not in self.predictions[model]:
                        continue
                    
                    pred = np.asarray(self.predictions[model][str(h)])

                    pred = pred[h_max - h + start:][:len(y_true)]

                    x_pred = np.arange(len(pred))

                    # plot the mean forecast
                    plt.plot(x_pred, pred,
                            label=f'{model}', color=self.color.get(model, 'black'))
                
                plt.title(f"Forecasts h = {h+1}")
                plt.ylabel("Value")
                plt.grid(True)
                plt.xticks([])
                plt.xlabel("Time index")
                plt.tight_layout()
                
                # one legend for all subplots
                handles, labels = plt.gca().get_legend_handles_labels()
                plt.legend(handles, labels,
                        loc='lower center',
                        ncol=len(self.predictions) + 1,    # +1 for 'True'
                        bbox_to_anchor=(0.5, -0.05))
                
                # save and/or show
                outpath = os.path.join(complete_path, f"horizon_{h}.png")
                plt.savefig(outpath, dpi=600, bbox_inches='tight')
                plt.close()

    
    def horizons_amp_models(self, models, start=0, std=True, show=False):
        """
        This plot will take the start index and project the forecsts towards the future for the given models. Detect if there are std
        """
        h_max = int(max([int(h) for h in self.predictions[models[0]]])) if models else 29
        # Create one figure
        fig, ax = plt.subplots(figsize=(12, 3))
        y_true = self.y_test[start:h_max + start + 1]
        x_true = np.arange(1, len(y_true)+1)

        # 1) plot true values
        ax.plot(x_true, y_true,
                color='black',
                linewidth=2,
                label='True')
        

        # 2) plot each model
        for model in models:
            preds = self.predictions[model]
            stds = self.std[model] if std else None

            values = []
            stds_values = []

            for h in preds:
                values.append(np.asarray(preds[h])[start])
                if h in stds and len(stds[h]) > 0:
                    stds_values.append(np.asarray(stds[h])[start])

            values = np.array(values)
            if stds_values:
                stds_values = np.array(stds_values) 

        
            # plot the mean forecast
            ax.plot(x_true, values,
                    label=f'{model}', color=self.color.get(model, 'black'))
            # shaded ±1 std
            if stds_values is not None and len(stds_values) > 0:
                ax.fill_between(x_true,
                                values - stds_values,
                                values + stds_values,
                                alpha=0.2, 
                                color=self.color.get(model, 'black'))
                
        ax.set_title(f"Forecasts for {', '.join(models)}")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.set_xticks(np.arange(1, len(values)+1))
        ax.set_xlabel("Horizon")
        plt.tight_layout()
        # one legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels,
                loc='lower center',
                ncol=len(models) + 1,    # +1 for 'True'
                bbox_to_anchor=(0.5, -0.05))
        # save and/or show
        outpath = os.path.join(self.results_path, "amp_forecast_models.png")
        plt.savefig(outpath, dpi=600, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)


    def table_error_metrics(self, group=1):
        """
        Create and save a nested JSON of error metrics grouped by horizons.
        
        Parameters
        ----------
        group : int
            Number of equal‐sized horizon‐groups to form.
        filename : str
            Name of the JSON file to write under self.results_path.
        """
        # 1) extract all horizons (assume same for every model/metric)
        err_dict = self.all_metrics["err"]
        sample_model = next(iter(err_dict))
        sample_metric = next(iter(err_dict[sample_model]))
        # list of errors per horizon for that metric:
        horizon_count = len(err_dict[sample_model][sample_metric])
        horizons = list(range(1, horizon_count+1))
        
        # 2) split into `group` bins
        # numpy.array_split will make groups as equal as possible
        bins = np.array_split(horizons, group)
        # name them by their range, e.g. "1-3"
        bin_names = [f"{int(b[0])}-{int(b[-1])}" for b in bins]

        # 3) prepare output structure
        out = {}

        # 4) for each metric (e.g. "MAE", "RMSE", etc.)
        for metric in err_dict[sample_model].keys():
            out_metric = {}
            # pull the parallel se‐dict
            err_dict = self.all_metrics["err"]
            se_dict = self.all_metrics["err_se"]

            for bin_name, bin_horizons in zip(bin_names, bins):
                grp_results = {}
                for model in err_dict.keys():
                    # check if model has this metric
                    if metric not in err_dict[model] or len(err_dict[model][metric]) == 0:
                        continue
                    # get the raw lists
                    errs = np.array(err_dict[model][metric])
                    ses  = np.array(se_dict[model][metric])
                    # select this group’s horizons (index = horizon-1)
                    idx = [h-1 for h in bin_horizons]
                    print(metric, model, bin_name, idx, len(errs), len(ses))
                    vals = errs[idx]
                    vals_se = ses[idx]

                    # compute mean and standard‐error‐of‐the‐mean
                    mean_val = float(vals.mean())
                    # SEM: sqrt(mean of variances / n)
                    sem_val = float(np.sqrt((vals_se**2).mean())/np.sqrt(len(vals_se)))

                    grp_results[model] = [mean_val, sem_val]

                out_metric[bin_name] = grp_results

            out[metric] = out_metric

        # 5) write to JSON
        path = os.path.join(self.results_path, "error_metrics_table.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"Wrote grouped error‐metrics JSON to {path}")

    
    def table_parameters_time(self):

        df = pd.DataFrame(columns=["model", "# parameters", "init_time", "forecast_time", "update_time"])

        # We will use the the self.params and self.times dicts for the task

        for model in self.times.keys():
            new_row = {
                "model" : model, 
                "# parameters" : self.params[model]["num_params"],
                "init_time" : self.times[model]["training"],
                "forecast_time" : self.times[model]["forecast"], 
                "update_time" : self.times[model]["update"]
            }

            df.loc[len(df)] = new_row

        # We save the df into a csv file

        df.to_csv(os.path.join(self.results_path, "table_params_times.csv"), index=False)




class ReconciliationPlotter:
    def __init__(self, metrics_path, results_path):
        self.results_path = results_path
        self.metrics_path = metrics_path

        self.metrics = pd.read_csv(os.path.join(self.metrics_path, "dataframe_evaluation_metrics.csv"))
        self.metrics_se = pd.read_csv(os.path.join(self.metrics_path, "dataframe_evaluation_metrics_se.csv"))
        self.computing_times = pd.read_csv(os.path.join(self.metrics_path, "computing_time.csv"))
        self.base = "base"

        self.nodes = sorted(self.metrics['node'].unique())
        self.horizons = sorted(self.metrics['horizon'].unique())
        self.var_methods = sorted(
            set(self.metrics['var_method'].unique()) - {self.base}
        )
        self.rec_methods = sorted(
            set(self.metrics['rec_method'].unique()) - {self.base}
        )
        
        with open(os.path.join(self.metrics_path, "reconciliation_results.json"), 'r') as f:
            self.results = json.load(f)

        print("ReconciliationPlotter initialized with results and metrics data.")     
        
        '''
        Results has the following structure:
        {
            "h": {
                "node": {
                    "method": {
                        "var": {
                            "y": []
                            "mu": [],
                            "std": []

        Metrics is a df with the following columns:
            node, horizon, rec_method, var_method, alpha, RMSE, MAE, CRPS, NLL, ECE
        '''

    def table_times(self):

        # Take the self.times dataframe and make a  json: {var_method, rec_method: {matrix_creation_time: val, forward_time: val}}

        # var_method,rec_method,matrix,forward_pass

        times_dict = {}
        for _, row in self.computing_times.iterrows():
            var_method = row['var_method']
            rec_method = row['rec_method']
            matrix_creation_time = row['matrix']
            forward_time = row['forward_pass']

            if var_method not in times_dict:
                times_dict[var_method] = {}
            if rec_method not in times_dict[var_method]:
                times_dict[var_method][rec_method] = {}
            times_dict[var_method][rec_method] = {
                'matrix_creation_time': matrix_creation_time,
                'forward_time': forward_time
            }

        # Save to json
        times_path = os.path.join(self.results_path, "times_reconciliation.json")
        with open(times_path, 'w') as f:
            json.dump(times_dict, f, indent=2)



    def models_ranking(self, group=None, metric='RMSE'):
        """
        Generate a ranking DataFrame for a single error metric and save as CSV.

        Parameters
        ----------
        group : int, optional
            Bin width for horizons; horizons are grouped in ranges of this size (e.g., 1-6, 7-12).
            If None, each horizon is its own bin.
        metric : str, default 'RMSE'
            Error metric column to rank (e.g., 'RMSE', 'MAE').
        save_path : str, optional
            Path to save the resulting DataFrame as CSV.

        Returns
        -------
        pandas.DataFrame
            Ranking table (rows: (var_method, rec_method), cols: horizon bins).
        """
        import os
        import numpy as np
        import pandas as pd

        # 1) Copy metrics
        df = self.metrics.copy()

        # 2) Compute horizon bins as contiguous ranges
        if isinstance(group, int) and group > 0:
            # Compute low and high bounds for each horizon
            low = ((df['horizon'] - 1) // group) * group + 1
            high = low + group - 1
            df['horizon_bin'] = low.astype(str) + '-' + high.astype(str)
        else:
            df['horizon_bin'] = df['horizon'].astype(str)

        # 3) Compute rank for each bin
        displays = []
        for bin_label, part in df.groupby('horizon_bin'):
            combos = pd.MultiIndex.from_frame(
                part[['var_method', 'rec_method']].drop_duplicates()
            )
            nodes = sorted(part['node'].unique())

            rank_df = pd.DataFrame(index=combos, columns=nodes, dtype=float)

            # Rank the mean metric across all horizons in the bin, for each node
            for node in nodes:
                tmp = part[part['node'] == node]
                mean_metric = (
                    tmp.groupby(['var_method', 'rec_method'])[metric]
                    .mean()
                )
                rank_df.loc[mean_metric.index, node] = mean_metric.rank(
                    method='average', ascending=True
                )

            # Average the ranks across nodes
            mean_rank = rank_df.mean(axis=1)
            disp = mean_rank.map(lambda x: f"{x:.2f}")
            disp.name = bin_label
            displays.append(disp)

        # 4) Combine bins into final table
        result = pd.concat(displays, axis=1)

        # 5) Reorder rows so 'base' appears first if present
        vm_order = list(result.index.get_level_values('var_method').unique())
        if self.base in vm_order:
            vm_order = [self.base] + [vm for vm in vm_order if vm != self.base]

        ordered_index = []
        for vm in vm_order:
            recs = result.loc[vm].index.get_level_values('rec_method')
            for rec in recs:
                ordered_index.append((vm, rec))
        result = result.reindex(ordered_index)

        result.to_csv(os.path.join(self.results_path, f"reconciliation_rankings_{metric}.csv"), index=True)

        return result

    def errors_by_group(self, group=None, metric='RMSE', se=False):
        """
        Compute mean error metric by (var_method, rec_method) across horizon‐bins.

        Parameters
        ----------
        group : int or None
            Bin width for horizons; horizons are grouped in ranges of this size
            (e.g. 1–6, 7–12). If None each horizon is its own bin.
        metric : str, default 'RMSE'
            Column name of the error metric to aggregate (e.g. 'RMSE', 'MAE').

        Returns
        -------
        pandas.DataFrame
            Table of mean errors: rows are MultiIndex (var_method, rec_method),
            columns are horizon‐bin labels, values are the average of `metric`.
        """
        import pandas as pd

        # 1) copy
        if se:
            df = self.metrics_se.copy()
        else:
            df = self.metrics.copy()

        # 2) define bins
        if isinstance(group, int) and group > 0:
            low  = ((df['horizon'] - 1) // group) * group + 1
            high = low + group - 1
            df['horizon_bin'] = low.astype(str) + '-' + high.astype(str)
        else:
            df['horizon_bin'] = df['horizon'].astype(str)

        # 3) pivot to get mean(metric) per var_method, rec_method, horizon_bin
        result = (
            df
            .groupby(['var_method', 'rec_method', 'horizon_bin'])[metric]
            .mean()
            .unstack('horizon_bin')
        )

        # 4) reorder so that 'base' var_method (if present) comes first
        vm_order = list(result.index.get_level_values('var_method').unique())
        if hasattr(self, 'base') and self.base in vm_order:
            vm_order = [self.base] + [vm for vm in vm_order if vm != self.base]

        ordered_index = []
        for vm in vm_order:
            recs = result.loc[vm].index
            for rec in recs:
                ordered_index.append((vm, rec))
        result = result.reindex(ordered_index)

        # Save the result to a CSV file
        if not se:
            result.to_csv(os.path.join(self.results_path, f"reconciliation_errors_{metric}.csv"), index=True)
        else:
            result.to_csv(os.path.join(self.results_path, f"reconciliation_errors_{metric}_se.csv"), index=True)


    def plot_covariance_comparison(self, method, var, h, save_path=None):
        """
        Plot covariance (correlation) comparison between base and reconciled forecasts.

        Parameters
        ----------
        method : str
            Key for the reconciliation method in self.results[h][node].
        var : str
            Key for the variance method under the chosen reconciliation method.
        h : int
            Horizon (1-based).
        save_path : str, optional
            A format string path to save the figures, with placeholders {h}, {method}, {var}.
        """
        # zero-index horizon
        h_idx = h - 1
        str_h = str(h_idx)

        # list of nodes
        nodes = sorted(self.results[str_h].keys())

        # --- build base_matrix
        base_matrix = []
        for node in nodes:
            rec = self.results[str_h][node].get('base', {}).get('base', {})
            mu = np.array(rec['mu'])
            y  = np.array(rec['y'])
            base_matrix.append(mu - y)
        base_matrix = np.vstack(base_matrix)

        # --- build recon_matrix
        recon_matrix = []
        for node in nodes:
            rec = self.results[str_h][node].get(method, {}).get(var, {})
            mu = np.array(rec['mu'])
            y  = np.array(rec['y'])
            recon_matrix.append(mu - y)
        recon_matrix = np.vstack(recon_matrix)

        # --- compute correlations
        corr_base      = np.corrcoef(base_matrix)
        corr_reconciled = np.corrcoef(recon_matrix)

        # mask for upper triangle only
        mask = np.triu(np.ones_like(corr_base, dtype=bool), k=1)

        # --- FIGURE 1: difference heatmap
        diff = corr_reconciled - corr_base
        plt.figure(figsize=(6,5))
        sns.heatmap(diff, mask=mask, annot=True, fmt=".2f", 
                    xticklabels=nodes, yticklabels=nodes, 
                    cmap='vlag', center=0, cbar_kws={'label':'Δ corr'})
        plt.title(f"Δ Correlation ({method}, {var}) at Horizon {h}")
        plt.tight_layout()
        save_path = save_path or os.path.join(self.results_path, "reconciliation_covariance_diff.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # --- FIGURE 2: side-by-side base vs reconciled
        fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)
        sns.heatmap(corr_base, mask=mask, annot=True, fmt=".2f",
                    xticklabels=nodes, yticklabels=nodes, 
                    cmap='coolwarm', cbar_kws={'label':'corr'}, ax=axes[0])
        axes[0].set_title(f"Base Correlation (h={h})")
        sns.heatmap(corr_reconciled, mask=mask, annot=True, fmt=".2f",
                    xticklabels=nodes, yticklabels=nodes, 
                    cmap='coolwarm', cbar_kws={'label':'corr'}, ax=axes[1])
        axes[1].set_title(f"Reconciled ({method}, {var})")
        plt.tight_layout()
        save_path = os.path.join(self.results_path, "reconciliation_covariance_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def error_metrics_grouped_horizons(self, group: int = None, reference_path: str = None, save_path: str = None):
        """
        Show RMSE and Calibration errors (format: RMSE(Calibration))
        with top-level columns: index, industry, sector, stock, global;
        second-level columns: horizon bins. Rows: var_method, rec_method.
        """
        df = self.metrics.copy()
        # Compute horizon bins
        if isinstance(group, int) and group > 0:
            df['horizon_bin'] = ((df['horizon'] - 1) // group) * group + group
        else:
            df['horizon_bin'] = df['horizon']

        # Identify nodes per category
        index_nodes = ["^GSPC"]
        sectors = [f.replace('_series.csv','') for f in os.listdir(os.path.join(reference_path, 'sector'))]
        industries = [f.replace('_series.csv','') for f in os.listdir(os.path.join(reference_path, 'industry'))]
        stocks = [f.replace('_history.csv','') for f in os.listdir(os.path.join(reference_path, 'historical_data'))]
        # Category keys
        categories = ['index', 'sector', 'industry', 'stock']
        # Map node to category
        node_cat = {}
        for node in df['node'].unique():
            if node in index_nodes:
                node_cat[node] = 'index'
            elif node in industries:
                node_cat[node] = 'industry'
            elif node in sectors:
                node_cat[node] = 'sector'
            elif node in stocks:
                node_cat[node] = 'stock'
            else:
                node_cat[node] = None

        # Gather horizon bins
        bins = sorted(df['horizon_bin'].unique())

        # Helper to build frame per DataFrame
        def _build_frame(sub_df):
            """Return DataFrame indexed by (var_method,rec_method) with bins as columns."""
            combos = pd.MultiIndex.from_frame(sub_df[['var_method','rec_method']].drop_duplicates())
            frame = pd.DataFrame(index=combos)
            for hb in bins:
                grp = sub_df[sub_df['horizon_bin']==hb]
                meanm = grp.groupby(['var_method','rec_method'])[['RMSE','CAL']].mean()
                disp = meanm['RMSE'].map(lambda x: f"{x:.2f}") + "(" + meanm['CAL'].map(lambda x: f"{x:.2f}") + ")"
                disp.name = hb
                frame = frame.join(disp)
            return frame

        # Build per-category frames
        frames = {}
        for cat in categories:
            sub = df[df['node'].map(node_cat)==cat]
            frames[cat] = _build_frame(sub)
        # Global across all nodes
        frames['global'] = _build_frame(df)

        # Concatenate along columns
        result = pd.concat(frames.values(), axis=1, keys=frames.keys())

        # Reorder rows: ensure 'base' var_method at top
        idx = result.index
        var_methods = list(dict.fromkeys(idx.get_level_values(0)))
        if 'base' in var_methods:
            var_methods = ['base'] + [m for m in var_methods if m != 'base']
        new_idx = []
        for vm in var_methods:
            recs = idx[idx.get_level_values(0)==vm].get_level_values(1)
            for rec in recs:
                new_idx.append((vm, rec))
        result = result.reindex(new_idx)

        print(result.head())

        # save dataframe to json
        if save_path:
            print(type(result))
            # Check the file extension
            if not save_path.endswith('.json'):
                save_path = os.path.splitext(save_path)[0] + '.json'
            
            # generate the JSON text
            json_str = result.to_json(orient='split', index=True)
            print(json_str)  # Print first 1000 characters for debugging
            # now write it explicitly as UTF-8 text
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

        # Shade by var_method
        shades = np.linspace(0.9, 0.5, len(var_methods))
        palette = {vm: colors.to_hex((s,)*3) for vm, s in zip(var_methods, shades)}
        styled = result.style.apply(lambda row: [f'background-color: {palette.get(row.name[0], "#FFFFFF")}']*result.shape[1], axis=1)

        # Optionally save
        if save_path:
            if dfi is None:
                raise ImportError("Please install 'dataframe_image' to save styled tables.")
            dfi.export(styled, save_path)

        return styled
        
    

    def plot_levels(self, stock, sector, industry, index):
        '''
        In this plot, we will be given a stock, sector, industry and index and we will plot 
        have a 2 column 4 row plot with the following structure:
        1. Stock level: each of the var method and all rec_method
        2. Sector level: each of the var method and all rec_method
        3. Industry level: each of the var method and all rec_method
        4. Index level: each of the var method and all rec_method
        The values will be the first forecast for all horizons for the given stock, sector, industry and index.
        '''
        pass

if __name__ == "__main__":
    # synth_plotter = ReconciliationPlotter(r"D:\Documentos\ICAI\TFG\Code\DLM Model\results\reconciliation_results_synthetic_series.csv", 
    #                       r"D:\Documentos\ICAI\TFG\Code\DLM Model\results\reconciliation_results_synthetic_series.json")
    
    # # Example usage
    # styled_table = synth_plotter.models_ranking(group=5, save_path=r"D:\Documentos\ICAI\TFG\Code\DLM Model\results\reconciliation_results_models_ranking_synthetic_series.png")
    # synth_plotter.covariance_errors(method='mint_shrink', var='montecarlo', h=1, save_path=r"D:\Documentos\ICAI\TFG\Code\DLM Model\results\reconciliation_results_covariance_errors.png")

    # fins_plotter = ReconciliationPlotter(
    #     r"D:\Documentos\ICAI\TFG\Code\DLM Model\results\reconciliation_results_snp500_data.csv", 
    #     r"D:\Documentos\ICAI\TFG\Code\DLM Model\results\reconciliation_results_snp500_data.json")
    
    # # Example usage
    # styled_table = fins_plotter.error_metrics_grouped_horizons(group=10, 
    #                                                            reference_path=r"D:\Documentos\ICAI\TFG\Code\DLM Model\data\snp500_data\yfinance",
    #                                                            save_path=r"D:\Documentos\ICAI\TFG\Code\DLM Model\results\reconciliation_results_models_ranking_snp500_data.png")

    pass